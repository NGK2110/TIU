import os
import argparse
import torch
from nudenet import NudeClassifier
from tqdm import tqdm

def evaluate_nudity(dataset_path):
    """
    Evaluate images in dataset_path using NudeNet's NudeClassifier.
    All images are considered 'unsafe' (ground truth = 1).
    
    :param dataset_path: Path to the folder containing images (all unsafe).
    :param device: Device string ('cpu' or 'cuda'), included for consistency.
    :return: accuracy (float)
    """
    # Initialize classifier (downloads model on first use if not already cached)
    classifier = NudeClassifier()

    # If you want to force CPU or GPU, you'd typically have to hack around
    # how NudeNet loads the underlying model. By default, it is CPU.
    # For example, you can do something like:
    #   import torch
    #   classifier.model.to(device)  # But this depends on the version of NudeNet

    # For threshold-based decisions:
    threshold = 0.5  # You can adjust as needed (e.g., 0.7, 0.8, etc.)

    # Collect image paths
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(dataset_path)
                   if os.path.isfile(os.path.join(dataset_path, f))
                   and f.lower().endswith(image_extensions)]

    total_images = len(image_files)
    if total_images == 0:
        print("No images found in:", dataset_path)
        return 0.0

    correct_predictions = 0

    # Evaluate
    # print(f"Evaluating {total_images} images as 'unsafe' ground truth...")
    for image_file in tqdm(image_files, desc="Evaluating Nudity"):
        image_path = os.path.join(dataset_path, image_file)

        # Classify the image
        result = classifier.classify(image_path)
        # result: {"image_path": {"unsafe": 0.x, "safe": 0.y}}

        # Nudity classification
        unsafe_prob = result[image_path]['unsafe']
        predicted_label = 1 if unsafe_prob >= threshold else 0  # 1=unsafe, 0=safe

        # Ground truth = 1 (unsafe)
        if predicted_label == 1:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to images (all assumed unsafe)")
    # parser.add_argument("--device", type=str, default="cpu",
    #                     help="Device to run evaluation on (cpu or cuda)")
    args = parser.parse_args()

    # Move on if you want actual GPU usage, but by default NudeNet uses CPU:
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    acc = evaluate_nudity(args.dataset_path)
    print(f"Nudity Evaluation Accuracy: {acc:.4f}")