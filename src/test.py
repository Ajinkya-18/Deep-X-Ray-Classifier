import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from train import val_transforms
from utils import test_model


test_dir = '../data/test'
test_dataset = ImageFolder(root=test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset)


model = torch.load('../models/x_ray_classifier_resnet_18.pt')

test_model(model, test_loader)
print("Test loop finished!")




