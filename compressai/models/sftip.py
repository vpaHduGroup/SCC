import math

import numpy as np
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from .utils import conv, deconv, update_registered_buffers, sobel_dx, sobel_dy, laplace
from compressai.ops import ste_round
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from compressai.layers.layers import NESFT, NESFTResblk
from .base import CompressionModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import compressai.transforms.functional as function
from torchvision import transforms as transforms


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SFTIPPre(CompressionModel):
    """Train without Mask"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5
        self.prior_nc = 64

        self.get_lap = laplace(1, 1, stride=1, kernel_size=3)

        self.get_lap.weight.requires_grad = False
        self.get_lap.bias.requires_grad = False

        # g_a,structure
        self.structure_feature_g1 = nn.Sequential(
            conv(3 + 3, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_g2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_g3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_g4 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # h_a,structure
        self.structure_feature_h0 = nn.Sequential(
            conv(M + 3, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_h1 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_h2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_h3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # f_c
        self.structure_feature_gs0 = nn.Sequential(
            deconv(N, N, 3),
            nn.GELU(),
            deconv(N, N, 3),
            nn.GELU(),
            conv(N, N, 3, 1)
        )

        # g_s,structure
        self.structure_feature_gs1 = nn.Sequential(
            conv(M + N, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_gs2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_gs3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_gs4 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # compression networks
        # g_a

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a2 = GDN(N)

        self.g_a3 = conv(N, N, kernel_size=5, stride=2)
        self.g_a4 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a5 = GDN(N)

        self.g_a6 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)
        self.g_a7 = conv(N, N, kernel_size=5, stride=2)
        self.g_a8 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a9 = GDN(N)

        self.g_a10 = conv(N, M, kernel_size=5, stride=2)
        self.g_a11 = Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2)
        self.g_a12 = NESFTResblk(M, self.prior_nc, kernel_size=3)

        # g_s

        self.g_s0 = NESFTResblk(M, self.prior_nc, kernel_size=3)
        self.g_s1 = Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2)
        self.g_s2 = deconv(M, N, kernel_size=5, stride=2)

        self.g_s3 = GDN(N, inverse=True)
        self.g_s4 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s5 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s6 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)

        self.g_s7 = GDN(N, inverse=True)
        self.g_s8 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s9 = deconv(N, N, kernel_size=5, stride=2)

        self.g_s10 = GDN(N, inverse=True)
        self.g_s11 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s12 = deconv(N, 3, kernel_size=5, stride=2)

        # h_a

        self.h_a0 = conv3x3(320, 320)
        self.h_a1 = NESFT(320, self.prior_nc, kernel_size=3)
        self.h_a2 = nn.GELU()
        self.h_a3 = conv3x3(320, 288)
        self.h_a4 = NESFT(288, self.prior_nc, kernel_size=3)
        self.h_a5 = nn.GELU()
        self.h_a6 = conv3x3(288, 256, stride=2)
        self.h_a7 = NESFT(256, self.prior_nc, kernel_size=3)
        self.h_a8 = nn.GELU()
        self.h_a9 = conv3x3(256, 224)
        self.h_a10 = NESFTResblk(224, self.prior_nc, kernel_size=3)
        self.h_a11 = nn.GELU()
        self.h_a12 = conv3x3(224, 192, stride=2)

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def g_a(self, x, structure):
        structure = self.structure_feature_g1(torch.cat([structure, x], dim=1))
        x = self.g_a0(x)
        x = self.g_a1(x, structure)
        x = self.g_a2(x)

        structure = self.structure_feature_g2(structure)
        x = self.g_a3(x)
        x = self.g_a4(x, structure)
        x = self.g_a5(x)

        structure = self.structure_feature_g3(structure)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, structure)
        x = self.g_a9(x)

        structure = self.structure_feature_g4(structure)
        x = self.g_a10(x)
        x = self.g_a11(x)
        x = self.g_a12(x, structure)
        return x


    def h_a(self, x, structure):
        structure = F.adaptive_avg_pool2d(structure, x.size()[2:])
        structure = self.structure_feature_h0(torch.cat([structure, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, structure)
        x = self.h_a2(x)

        structure = self.structure_feature_h1(structure)
        x = self.h_a3(x)
        x = self.h_a4(x, structure)
        x = self.h_a5(x)

        structure = self.structure_feature_h2(structure)
        x = self.h_a6(x)
        x = self.h_a7(x, structure)
        x = self.h_a8(x)

        structure = self.structure_feature_h3(structure)
        x = self.h_a9(x)
        x = self.h_a10(x, structure)
        x = self.h_a11(x)

        x = self.h_a12(x)

        return x

    def g_s(self, x, z_hat):
        structure = self.structure_feature_gs0(z_hat)
        structure = F.adaptive_avg_pool2d(structure, x.size()[2:])
        w = self.structure_feature_gs1(torch.cat([structure, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)

        w = self.structure_feature_gs2(w)
        x = self.g_s4(x, w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)

        w = self.structure_feature_gs3(w)
        x = self.g_s8(x, w)
        x = self.g_s9(x)
        x = self.g_s10(x)

        w = self.structure_feature_gs4(w)
        x = self.g_s11(x, w)
        x = self.g_s12(x)

        return x

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        lap1 = self.get_lap(x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap2 = self.get_lap(x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap3 = self.get_lap(x[:, 2, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        structure_b = torch.cat([lap1, lap2, lap3], dim=1)
        y = self.g_a(x, structure_b)
        y_shape = y.shape[2:]
        z = self.h_a(y, structure_b)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        lap1 = self.get_lap(x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap2 = self.get_lap(x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap3 = self.get_lap(x[:, 2, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        structure_b = torch.cat([lap1, lap2, lap3], dim=1)

        y = self.g_a(x, structure_b)

        y_shape = y.shape[2:]

        z = self.h_a(y, structure_b)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class SFTIP(CompressionModel):
    """Train with Mask"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5
        self.prior_nc = 64

        self.get_lap = laplace(1, 1, stride=1, kernel_size=3)
        self.get_lap.weight.requires_grad = False
        self.get_lap.bias.requires_grad = False

        # g_a,structure
        self.structure_feature_g1 = nn.Sequential(
            conv(3 + 3, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_g2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_g3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_g4 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # h_a,structure
        self.structure_feature_h0 = nn.Sequential(
            conv(M + 3, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_h1 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_h2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_h3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # f_c
        self.structure_feature_gs0 = nn.Sequential(
            deconv(N, N, 3),
            nn.GELU(),
            deconv(N, N, 3),
            nn.GELU(),
            conv(N, N, 3, 1)
        )

        # g_s,structure
        self.structure_feature_gs1 = nn.Sequential(
            conv(M + N, self.prior_nc * 4, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 4, self.prior_nc * 2, 3, 1),
            nn.GELU(),
            conv(self.prior_nc * 2, self.prior_nc, 3, 1)
        )
        self.structure_feature_gs2 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_gs3 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )
        self.structure_feature_gs4 = nn.Sequential(
            conv(self.prior_nc, self.prior_nc, 1, 1),
            nn.GELU(),
            deconv(self.prior_nc, self.prior_nc, 3),
            nn.GELU(),
            conv(self.prior_nc, self.prior_nc, 1, 1)
        )

        # compression networks
        # g_a

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a2 = GDN(N)

        self.g_a3 = conv(N, N, kernel_size=5, stride=2)
        self.g_a4 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a5 = GDN(N)

        self.g_a6 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)
        self.g_a7 = conv(N, N, kernel_size=5, stride=2)
        self.g_a8 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_a9 = GDN(N)

        self.g_a10 = conv(N, M, kernel_size=5, stride=2)
        self.g_a11 = Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2)
        self.g_a12 = NESFTResblk(M, self.prior_nc, kernel_size=3)

        # g_s

        self.g_s0 = NESFTResblk(M, self.prior_nc, kernel_size=3)
        self.g_s1 = Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2)
        self.g_s2 = deconv(M, N, kernel_size=5, stride=2)

        self.g_s3 = GDN(N, inverse=True)
        self.g_s4 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s5 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s6 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)

        self.g_s7 = GDN(N, inverse=True)
        self.g_s8 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s9 = deconv(N, N, kernel_size=5, stride=2)

        self.g_s10 = GDN(N, inverse=True)
        self.g_s11 = NESFT(N, self.prior_nc, kernel_size=3)
        self.g_s12 = deconv(N, 3, kernel_size=5, stride=2)

        # h_a

        self.h_a0 = conv3x3(320, 320)
        self.h_a1 = NESFT(320, self.prior_nc, kernel_size=3)
        self.h_a2 = nn.GELU()
        self.h_a3 = conv3x3(320, 288)
        self.h_a4 = NESFT(288, self.prior_nc, kernel_size=3)
        self.h_a5 = nn.GELU()
        self.h_a6 = conv3x3(288, 256, stride=2)
        self.h_a7 = NESFT(256, self.prior_nc, kernel_size=3)
        self.h_a8 = nn.GELU()
        self.h_a9 = conv3x3(256, 224)
        self.h_a10 = NESFTResblk(224, self.prior_nc, kernel_size=3)
        self.h_a11 = nn.GELU()
        self.h_a12 = conv3x3(224, 192, stride=2)

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def g_a(self, x, structure):
        structure = self.structure_feature_g1(torch.cat([structure, x], dim=1))
        x = self.g_a0(x)
        x = self.g_a1(x, structure)
        x = self.g_a2(x)

        structure = self.structure_feature_g2(structure)
        x = self.g_a3(x)
        x = self.g_a4(x, structure)
        x = self.g_a5(x)

        structure = self.structure_feature_g3(structure)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, structure)
        x = self.g_a9(x)

        structure = self.structure_feature_g4(structure)
        x = self.g_a10(x)
        x = self.g_a11(x)
        x = self.g_a12(x, structure)
        return x


    def h_a(self, x, structure):
        structure = F.adaptive_avg_pool2d(structure, x.size()[2:])
        structure = self.structure_feature_h0(torch.cat([structure, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, structure)
        x = self.h_a2(x)

        structure = self.structure_feature_h1(structure)
        x = self.h_a3(x)
        x = self.h_a4(x, structure)
        x = self.h_a5(x)

        structure = self.structure_feature_h2(structure)
        x = self.h_a6(x)
        x = self.h_a7(x, structure)
        x = self.h_a8(x)

        structure = self.structure_feature_h3(structure)
        x = self.h_a9(x)
        x = self.h_a10(x, structure)
        x = self.h_a11(x)

        x = self.h_a12(x)

        return x

    def g_s(self, x, z_hat):
        structure = self.structure_feature_gs0(z_hat)
        structure = F.adaptive_avg_pool2d(structure, x.size()[2:])
        w = self.structure_feature_gs1(torch.cat([structure, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)

        w = self.structure_feature_gs2(w)
        x = self.g_s4(x, w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)

        w = self.structure_feature_gs3(w)
        x = self.g_s8(x, w)
        x = self.g_s9(x)
        x = self.g_s10(x)

        w = self.structure_feature_gs4(w)
        x = self.g_s11(x, w)
        x = self.g_s12(x)

        return x

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, mask):
        lap1 = self.get_lap(x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap2 = self.get_lap(x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap3 = self.get_lap(x[:, 2, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        structure_b = torch.cat([lap1, lap2, lap3], dim=1)
        structure = (mask + 2) * structure_b / 3
        y = self.g_a(x, structure)
        y_shape = y.shape[2:]
        z = self.h_a(y, structure)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, mask):
        lap1 = self.get_lap(x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap2 = self.get_lap(x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        lap3 = self.get_lap(x[:, 2, :, :].view(x.size(0), 1, x.size(2), x.size(3)))
        structure_b = torch.cat([lap1, lap2, lap3], dim=1)

        structure = (mask + 2) * structure_b / 3

        y = self.g_a(x, structure)

        y_shape = y.shape[2:]

        z = self.h_a(y, structure)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)

        return {"x_hat": x_hat}