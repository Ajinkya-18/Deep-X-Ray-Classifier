import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .train import val_transforms
from .utils import test_model, load_model


test_dir = '../data/test'
test_dataset = ImageFolder(root=test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset)


model = load_model('../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt', 'cpu')

test_model(model, test_loader)
print("Test loop finished!")



