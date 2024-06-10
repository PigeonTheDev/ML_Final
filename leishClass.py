import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class LeishmaniaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = [f for f in os.listdir(root) if f.lower().endswith('.jpg')]
        self.labels = [f.replace('.jpg', '.json').replace('.JPG', '.json') for f in self.imgs]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        label_path = os.path.join(self.root, self.labels[idx])
        
        img = Image.open(img_path).convert("RGB")
        with open(label_path) as f:
            label_data = json.load(f)
        

        
        original_width, original_height = img.size
        boxes = []
        labels = []
        for shape in label_data['shapes']:
            if 'points' in shape and len(shape['points']) == 2:
                xmin, ymin = shape['points'][0]
                xmax, ymax = shape['points'][1]
                # Ensure correct ordering of coordinates
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(shape['label']))  # Convert label to int

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform is not None:
            img, target = self.transform(img, target, original_width, original_height)
        
        img = transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# Define transformations with resizing and scaling boxes
def resize_transform(img, target, original_width, original_height, new_size=(512, 512)):
    img = img.resize(new_size)
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height

    for box in target['boxes']:
        box[0] *= scale_x
        box[1] *= scale_y
        box[2] *= scale_x
        box[3] *= scale_y
    
    return img, target

# Paths to train and test directories
train_dir = '04-object detection Leishmania/train'  # Update this path
test_dir = '04-object detection Leishmania/test'  # Update this path


# Custom collate function
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

# Prepare data loaders
transform = resize_transform
full_dataset = LeishmaniaDataset(train_dir, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_dataset = LeishmaniaDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
