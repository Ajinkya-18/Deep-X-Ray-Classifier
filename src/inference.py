import torch
from train import val_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

infer_dir = '../data/infer'
infer_dataset = ImageFolder(root=infer_dir, transform=val_transforms)
data_loader = DataLoader(infer_dataset)


model = torch.load('../models/x_ray_classifier_resnet_18.pt', map_location=device)

model.eval()
model.to(device)

class_names = infer_dataset.classes

with torch.no_grad():
    progress_bar = tqdm(data_loader, 'Inference', leave=True)

    for x, y in progress_bar:
        y_hat = model(x)
        probs = torch.sigmoid(y_hat)
        preds = (probs > 0.5).int()
        for i in range(x.size(0)):
            print(f"Prediction: {class_names[int(preds[i].item())]} | Confidence: {probs[i].item():.3f}")









