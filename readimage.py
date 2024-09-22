import torch
from PIL import Image
from torchvision import transforms

def check_largee_values(image: torch.Tensor, threshold: float) -> bool:
    return bool((image > threshold).any())


transform = transforms.Compose([transforms.ToTensor()])
image_path ='/home/huangxf/project/STF/openimages/train/data/09db2087f6e4d803.jpg'
image = Image.open(image_path)
image_t = transform(image)
print(check_largee_values(image_t,1))