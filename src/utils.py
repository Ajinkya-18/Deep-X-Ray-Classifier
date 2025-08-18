from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2
import torch


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
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = v2.Compose([
    v2.Grayscale(num_output_channels=3),
    v2.Resize(size=(256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#--------------------------------------------------------------------------------------------------------------------------------------

def initialize_model(num_classes=1, model=models.resnet18(weights=ResNet18_Weights.DEFAULT)):
    from torch import nn

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer-4.1
    # for param in model.layer4[1].parameters():
    #     param.requires_grad = True

    # Replace original fc layer with custom fc layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

    return model

#-------------------------------------------------------------------------------------------------------------------------------------

def model_summary(model):
    print('--------------------------------------------Model Summary----------------------------------------------\n')
    for name, param in model.named_parameters():
        status = 'Trainable' if param.requires_grad else 'Non-Trainable (Frozen)'
        print(f'{name:30} | {status}')

    print('\n======================================================================================================\n')

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}\n")
    
#--------------------------------------------------------------------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0

#--------------------------------------------------------------------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, batch_size=16, epochs=50, learning_rate=1e-3, log_dir='../reports/exp2_resnet18-layer4-fc-unfrozen'):
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import torch
    from torch import nn

    device = torch.device('cpu')
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopping = EarlyStopping(min_delta=0.001)
    
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n------------------------------------------------------------------------------------")
        
        # Training
        
        model.train()
        
        total_train_loss = 0.0
    
        train_progress_bar = tqdm(train_loader, desc='Training', leave=True)
        
        for batch, (x, y) in enumerate(train_progress_bar):
            optimizer.zero_grad()
            
            y_pred = model(x)
            
            train_loss = loss_fn(y_pred, y.unsqueeze(1).float())
            train_loss.backward()
            
            optimizer.step()
            
            total_train_loss += train_loss.item()
            
            train_progress_bar.set_postfix({'Batch Loss': f"{train_loss.item():.3f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        
        
        # Validation
        model.eval()
        total_val_loss, correct = 0.0, 0
        total = 0
    
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc='Validation', leave=True)
            
            for x, y in val_progress_bar:
                y_pred = model(x)
                
                val_loss = loss_fn(y_pred, y.unsqueeze(1).float())
                total_val_loss += val_loss.item()
                
                val_progress_bar.set_postfix({'Val Loss': f"{val_loss.item():.3f}"})
                
                y_pred_labels = (torch.sigmoid(y_pred) > 0.5).int()
                
                correct += (y_pred_labels == y.unsqueeze(1).int()).sum().item()
                total += y.size(0)
    
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, '../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt')

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(f"Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Acccuracy: {(accuracy*100):.2f} % \n")

        # LR Scheduling
        scheduler.step(avg_val_loss)

        # Early Stopping Check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    
    writer.close()
    
    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_model(model, test_loader):
    from tqdm import tqdm
    from torch import nn
    import torch

    loss_fn = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(test_loader,  'testing', leave=True)

    total_test_loss, correct = 0.0, 0
    total = 0

    model.eval()

    for x, y in progress_bar:
        y_hat = model(x)
        loss = loss_fn(y_hat, y.unsqueeze(1).float())
        total_test_loss += loss.item()

        progress_bar.set_postfix({'Test Loss': f"{loss.item():.3f}"})

        y_hat_preds = (torch.sigmoid(y_hat) > 0.5).int()

        correct += (y_hat_preds == y.unsqueeze(1).int()).sum().item()
        total += y.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total

        print(f"Accuracy: {(accuracy*100):.2f}  | Average Test Loss: {avg_test_loss:.3f}\n")

#-----------------------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    import torch

    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
    model.eval()

    return model

#-----------------------------------------------------------------------------------------------------------------------




