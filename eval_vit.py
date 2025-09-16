import os
import argparse
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Custom Dataset for Single Class
class SingleClassDataset(Dataset):
    def __init__(self, root_dir, target_label=1):
        self.root_dir = root_dir
        self.target_label = target_label
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.target_label  # Assign a fixed label
        return image, label, img_path  # Return img_path for tracking predictions

# Custom collate function to handle batches of PIL images
def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels), list(paths)

# Evaluation function
def evaluate(model, processor, dataset_path, device, batch_size=8):
    # Load dataset using the custom dataset class
    dataset = SingleClassDataset(root_dir=dataset_path, target_label=1)  # Assuming 1 is the target class
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Set model to evaluation mode
    model.eval()
    all_preds = [] 
    all_labels = []
    total_images = len(dataset)
    positive_images = []  # Store paths of images predicted with label 1

    # print(f"\nEvaluating folder: {os.path.basename(dataset_path)}")

    # Run inference
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating", disable=True):
            # Process images using ViTImageProcessor (no prior transforms needed)
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            labels = labels.to(model.device)

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Track images predicted as label 1
            for i, pred in enumerate(preds):
                if pred.item() == 1:  # Label 1 prediction
                    positive_images.append(paths[i])

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # print(f"Images predicted with Unlearnt concept back: {len(positive_images)} out of {total_images}")

    return accuracy

def main(model_path, dataset_path, evaluate_all_folders=False):
    # Load the model from the specified path
    model = ViTForImageClassification.from_pretrained(model_path)

    # Load processor from the Hugging Face Hub for the base model
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
    
    # Ensure model is on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # List to store accuracy results for sorting
    accuracy_results = []

    # Evaluate each folder individually if the flag is set
    if evaluate_all_folders:
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):  # Check if it's a directory
                accuracy = evaluate(model, processor, folder_path)
                accuracy_results.append((int(folder), accuracy))  # Store results for sorting

        # Sort results by folder name in increasing order
        accuracy_results.sort(key=lambda x: x[0])
        
        print("\nEvaluation results:")
        # Print sorted accuracies
        for folder, accuracy in accuracy_results:
            print(f"Concept {folder+1} - {accuracy:.4f}")
    else:
        # Evaluate only the specified dataset_path
        accuracy = evaluate(model, processor, dataset_path)
        print(f"Accuracy for {os.path.basename(dataset_path)}: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT model on a dataset.")
    parser.add_argument("model_path", type=str, help="Path to the trained model.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset for evaluation.")
    parser.add_argument("--evaluate_all_folders", action="store_true", help="Evaluate all subfolders in the dataset_path separately.")
    args = parser.parse_args()
    
    main(args.model_path, args.dataset_path, args.evaluate_all_folders)