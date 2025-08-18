import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .utils import initialize_model, train_model, train_transforms, val_transforms


train_dir = "../data/chest_xray/train"
val_dir = "../data/chest_xray/test"


BATCH_SIZE=16

train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=False, prefetch_factor=1, 
                          persistent_workers=False, in_order=False)

val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=False, prefetch_factor=1, 
                        persistent_workers=False, in_order=True)


device = torch.device('cpu')
model = initialize_model(1).to(device)

trained_model = train_model(model, train_loader, val_loader, BATCH_SIZE, 50, 1e-3)
# torch.save(trained_model, '../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v2.pt')

print("Training completed!")



