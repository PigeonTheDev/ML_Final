import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from leishClass import test_dataset  # Adjust the import path as necessary
from torchvision.ops import box_iou
from collections import defaultdict

# Custom collate function
#debene
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
def get_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # 1 class (Leishmania) + background
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    results = []
    all_targets = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            results.extend(outputs)
            all_targets.extend(targets)
    return results, all_targets

def calculate_map(results, targets, iou_threshold=0.5):
    all_ious = []
    for res, target in zip(results, targets):
        if len(res['boxes']) == 0 or len(target['boxes']) == 0:
            continue

        ious = box_iou(res['boxes'], target['boxes'])
        max_iou, _ = ious.max(dim=1)
        all_ious.extend(max_iou)

    all_ious = torch.tensor(all_ious)
    precision = (all_ious > iou_threshold).float().mean().item()

    return precision

def filter_results(results, score_threshold=0.5):
    filtered_results = []
    for result in results:
        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']

        keep = scores >= score_threshold

        filtered_results.append({
            'boxes': boxes[keep],
            'labels': labels[keep],
            'scores': scores[keep]
        })
    
    return filtered_results

def visualize_predictions(images, results):
    for i in range(len(images)):
        fig, ax = plt.subplots(1)
        image = images[i]
        img = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        for box, label, score in zip(results[i]['boxes'], results[i]['labels'], results[i]['scores']):
            print(f"Box: {box.detach().cpu().numpy()}, Score: {score.detach().cpu().numpy()}")  # Debugging: Print the boxes, labels, and scores
            x_min, y_min, x_max, y_max = box.detach().cpu().numpy()
            if(score.detach().cpu().numpy() > 0.5):
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        ax.axis('off')
        plt.show()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model()
    model.to(device)

    # Load the latest checkpoint if available
    checkpoint_dir = '.'
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError("No checkpoint found. Please train the model first.")

    # Assuming test_dataset is already defined
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Evaluate the model and get results
    results, all_targets = evaluate_model(model, test_loader, device)
    print("Evaluation completed.")

    # Filter results based on score threshold
    filtered_results = filter_results(results, 0.5)

    # Calculate mAP scores
    precision = calculate_map(filtered_results, all_targets, iou_threshold=0.5)
    print(f'Precision: {precision:.4f}')

    # Visualize some predictions
    images_to_visualize = []
    results_to_visualize = []
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        results = model(images)
        filtered_results = filter_results(results, score_threshold=0.5)
        images_to_visualize.extend(images)
        results_to_visualize.extend(filtered_results)
        if len(images_to_visualize) >= 8:
            break  # Load only enough images for visualization

    visualize_predictions(images_to_visualize, results_to_visualize)
