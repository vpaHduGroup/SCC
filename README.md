# Enhanced Screen Content Image Compression: A Synergistic Approach for Structural Fidelity and Text Integrity Preservation
Pytorch implementation of the paper "Enhanced Screen Content Image Compression: A Synergistic Approach for Structural Fidelity and Text Integrity Preservation".
This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI). We kept scripts for training and evaluation, and removed other components. The major changes are provided in `compressai/models`. For the official code release, see the [CompressAI](https://github.com/InterDigitalInc/CompressAI).


## Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n compress python=3.7
conda activate compress
pip install compressai
pip install pybind11
cd SFTIP_SCC
pip install -e .
pip install -e '.[dev]'
```

> **Note**: wheels are available for Linux and MacOS.

## Usage

### Training
An examplary training script with a rate-distortion loss is provided in
`train.py`. 
And examplary training script with text region mask is provided in 
`mask/trainwithmask.py`. 

Training a pre-trained model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py -d /path/to/nature/image/dataset/ -e 1000 --batch-size 16 --save --save_path /path/to/save/ -m sftip --cuda --lambda 0.0035
e.g., CUDA_VISIBLE_DEVICES=0,1 python train.py -d openimages -e 1000 --batch-size 16 --save --save_path ckpt/cnn_0035.pth.tar -m sftip-pre --cuda --lambda 0.0035
```
Training a final model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python mask/trainwithmask.py -d /path/to/screen/image/dataset/ -e 1000 --batch-size 16 --save --save_path /path/to/save/ -m sftip --cuda --lambda 0.0035
```


### Evaluation

To evaluate a trained model on your own dataset, the evaluation script is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d /path/to/image/folder/ -r /path/to/reconstruction/folder/ -a sftip-pre -p /path/to/checkpoint/ --cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 python -m mask/eval_mask.py -d /path/to/image/folder/ -r /path/to/reconstruction/folder/ -a sftip -p /path/to/checkpoint/ --cuda
```


### Dataset
The script for downloading [OpenImages](https://github.com/openimages) is provided in `downloader_openimages.py`. Please install [fiftyone](https://github.com/voxel51/fiftyone) first.
And the `JPEGAI` dataset needs to applied to the JPEGAI official.

### Proposed methods in the code
`NESFT` and `structure_feature` corresponds to SEM in the paper. The rate distortion loss function corresponds to `PixelwiseRateDistortionLoss` of `mask/trainwithmask.py`.


## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Swin-Transformer: https://github.com/microsoft/Swin-Transformer
 * Tensorflow compression library by Ballé et al.: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from Fabian 'ryg' Giesen: https://github.com/rygorous/ryg_rans
 * Kodak Images Dataset: http://r0k.us/graphics/kodak/
 * Open Images Dataset: https://github.com/openimages
 * fiftyone: https://github.com/voxel51/fiftyone
 * CLIC: https://www.compression.cc/
 * STF: https://github.com/Googolxx/STF


