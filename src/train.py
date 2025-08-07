import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from utils import initialize_resnet18, train_model


train_dir = "../data/chest_xray/train"
val_dir = "../data/chest_xray/test"


train_transforms = v2.Compose([
    v2.Grayscale(num_output_channels=3),
    v2.Resize(size=(256, 256)), 
    # v2.RandomResizedCrop(size=(256, 256)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    # v2.GaussianNoise(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = v2.Compose([
    v2.Grayscale(num_output_channels=3),
    v2.Resize(size=(256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])


BATCH_SIZE=16

train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=False, prefetch_factor=1, 
                          persistent_workers=False, in_order=False)

val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=False, prefetch_factor=1, 
                        persistent_workers=False, in_order=True)


model = initialize_resnet18(1)

trained_model = train_model(model, train_loader, val_loader, BATCH_SIZE, 50, 1e-3)
torch.save(trained_model, '../models/x_ray_classifier_resnet_18.pt')

print("Training completed!")





