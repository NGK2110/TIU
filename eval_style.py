import os
import argparse
import timm
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# Suppress FutureWarning if desired
warnings.filterwarnings("ignore", category=FutureWarning)

# Custom Dataset for Single Class
class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_label=1):
        self.root_dir = root_dir
        self.transform = transform
        self.target_label = target_label
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.target_label  # Assign a fixed label
        return image, label

def evaluate_style(model, dataset_path, device):
    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Create custom dataset
    dataset = SingleClassDataset(dataset_path, transform=image_transform, target_label=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    total_images = 0
    correct_predictions = 0
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_images += labels.size(0)
    accuracy = correct_predictions / total_images
    return accuracy

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model.to(device)
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    # Evaluate model
    accuracy = evaluate_style(model, args.dataset_path, device)
    print(f"Accuracy: {accuracy:.4f}")