import os
import random
import shutil

# Paths
dataset_dir = '04-object detection Leishmania'
image_dir = os.path.join(dataset_dir, 'leishmania')
label_dir = os.path.join(dataset_dir, 'labels-json')
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Check if directories exist
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
if not os.path.exists(label_dir):
    raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

# List all image files and corresponding label files
all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
all_label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.json')]

# Create a dictionary to map image filenames to label filenames
image_to_label = {os.path.splitext(f)[0]: os.path.splitext(f)[0] + '.json' for f in all_image_files}

# Debugging: Print counts and sample file names
print(f"Number of image files: {len(all_image_files)}")
print(f"Number of label files: {len(all_label_files)}")

print("Sample image files:", all_image_files[:5])
print("Sample label files:", all_label_files[:5])

# Ensure the number of images and labels match
assert len(all_image_files) == len(all_label_files), "Number of images and labels do not match."

# Set random seed for reproducibility
random.seed(47)
random.shuffle(all_image_files)

# Split the data
split_index = int(len(all_image_files) * 0.8)
train_image_files = all_image_files[:split_index]
test_image_files = all_image_files[split_index:]

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to safely copy files
def safe_copy(src, dst):
    try:
        shutil.copy(src, dst)
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Move files to train and test directories
for file in train_image_files:
    safe_copy(os.path.join(image_dir, file), train_dir)
    label_file = image_to_label[os.path.splitext(file)[0]]
    safe_copy(os.path.join(label_dir, label_file), train_dir)

for file in test_image_files:
    safe_copy(os.path.join(image_dir, file), test_dir)
    label_file = image_to_label[os.path.splitext(file)[0]]
    safe_copy(os.path.join(label_dir, label_file), test_dir)

print("Dataset prepared. Train files:", len(train_image_files), "Test files:", len(test_image_files))
