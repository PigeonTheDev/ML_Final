import os
import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from leishClass import train_loader, val_loader

# Custom collate function
#unet denenebilir veya yolo
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

# Function to find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

# Define the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2  # 1 class (Leishmania) + background
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # Lowered learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Load the latest checkpoint if available
checkpoint_dir = '.'
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
start_epoch = 0
if latest_checkpoint:
    print(f"Loading checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch + 1}")

# Function to compute loss
def compute_loss(model, images, targets):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    return losses

# Training loop
def train_model(num_epochs, model, train_loader, val_loader, optimizer, lr_scheduler, device, start_epoch=0):
    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        batch_losses = []

        # Training phase
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            for target in targets:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)

            try:
                losses = compute_loss(model, images, targets)
                if torch.isnan(losses) or torch.isinf(losses):
                    print("NaN or Inf detected in loss, stopping training")
                    return  # Stop training if NaN or Inf is detected

                batch_losses.append(losses.item())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {losses.item()}")
            except Exception as e:
                print(f"Error during model training: {e}")
                raise

        lr_scheduler.step()
        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1} completed. Average Train Loss: {avg_train_loss}, Time: {epoch_time:.2f} seconds")

        # Validation phase
        model.train()  # Use train mode to compute loss
        val_batch_losses = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = list(image.to(device) for image in images)
                for target in targets:
                    target['boxes'] = target['boxes'].to(device)
                    target['labels'] = target['labels'].to(device)

                try:
                    # Compute the loss
                    loss_dict = model(images, targets)
                    if not isinstance(loss_dict, dict):
                        raise TypeError(f"Expected loss_dict to be a dict, but got {type(loss_dict)}")
                    
                    print(f"Validation Loss dict keys: {loss_dict.keys()}")  # Debugging: Print keys of loss_dict
                    losses = sum(loss for loss in loss_dict.values())
                    val_batch_losses.append(losses.item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    print(f"Loss dict: {loss_dict}")  # Debugging: Print the loss_dict content
                    raise

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses) if val_batch_losses else float('nan')
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} completed. Average Val Loss: {avg_val_loss}")

        model.train()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }
        torch.save(checkpoint, f"model_epoch_{epoch + 1}.pth")

    print("Model training completed for PyTorch.")
    return train_losses, val_losses

if __name__ == "__main__":
    num_epochs = 10
    # Assuming train_loader and val_loader are already defined
    train_losses, val_losses = train_model(num_epochs, model, train_loader, val_loader, optimizer, lr_scheduler, device, start_epoch)
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
