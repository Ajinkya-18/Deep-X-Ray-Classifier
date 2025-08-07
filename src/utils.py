def initialize_resnet18(num_classes=1):
    from torchvision import models
    from torch import nn

    weights = models.ResNet18_Weights
    model = models.resnet18(weights=weights)

    # Freeze initial layers
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

    return model


def train_model(model, train_loader, val_loader, batch_size=16, epochs=50, learning_rate=1e-3, log_dir='../reports/exp1'):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from torch import nn

    device = torch.device('cpu')
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter(log_dir=log_dir)
    
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

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(f"Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Accuracy: {(accuracy*100):.2f} % \n")
        
    writer.close()

    return model


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






