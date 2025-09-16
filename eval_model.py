import argparse
import re
import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from eval_vit import evaluate
from eval_style import evaluate_style
from eval_nudity import evaluate_nudity
from transformers import ViTForImageClassification, ViTImageProcessor
import timm

def get_last_folder_name(checkpoint_path):
    checkpoint_path = checkpoint_path.rstrip(os.sep)
    path_parts = checkpoint_path.split(os.sep)
    result_parts = []
    for part in path_parts:
        if "epochs" in part:
            result_parts = path_parts[path_parts.index(part):]
            break
    return '_'.join(result_parts)

def load_prompts_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = f.read().splitlines()
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    return prompts

def main():
    parser = argparse.ArgumentParser(description='Generate images based on prompts involving a given concept.')
    parser.add_argument('model_paths', type=str, help='Comma-separated list of model/checkpoint paths.')
    parser.add_argument('eval_model_path', type=str, help='Path to the classifier model to evaluate the generated images.')
    parser.add_argument('concept', type=str, default=None, help='The concept to generate prompts about.')
    parser.add_argument('--curr', action='store_true', help='If present, process curriculum subfolders inside model paths.')
    parser.add_argument('--prompt_file', type=str, default=None, help='Path to a file containing prompts to use instead of generating them.')
    parser.add_argument('--unlearn', action='store_true', help='Check whether evaluating for unlearned mmodel.')
    parser.add_argument('--theme', type=str, default=None, help='The theme/style of the prompts to unlearn.')
    parser.add_argument('--gpu', type=int, default=0, help='The GPU device to use.')
    args = parser.parse_args()

    # Process arguments
    model_paths = [path.strip() for path in args.model_paths.split(',')]
    eval_model_path = args.eval_model_path
    concept = args.concept

    # Load prompts from file if specified, else generate prompts
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # If --curr is present, expand model paths to include 'curriculum' subdirectories
    if args.curr:
        expanded_model_paths = []
        for model_path in model_paths:
            if os.path.isdir(model_path):
                # Find subdirectories starting with "curriculum"
                subdirs = [os.path.join(model_path, d) for d in os.listdir(model_path)
                           if d.startswith("curriculum") and os.path.isdir(os.path.join(model_path, d))]
                if subdirs:
                    expanded_model_paths.extend(subdirs)
                else:
                    print(f"No 'curriculum' subdirectories found in {model_path}")
            else:
                print(f"Warning: {model_path} is not a directory or does not exist.")
        if expanded_model_paths:
            model_paths.extend(expanded_model_paths)
        else:
            print("No curriculum subdirectories found.")
    
    # Load CLIP model and processor
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_scores = []

    if args.theme==None:
        # Load the model from the specified path
        eval_model = ViTForImageClassification.from_pretrained(eval_model_path)
        # Load processor from the Hugging Face Hub for the base model
        eval_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
        eval_model.to(device)
    
    # Branch 2: If theme is "Nudity" => Use NudeNet
    elif args.theme.lower() == "nudity":
        print("Loading NudeNet classifier...")
        # We'll load the classifier in the evaluate_nudity function
        eval_model = None  # Not needed for Nudity classification
        eval_processor = None  # Not needed either

    else:
        eval_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False).to(device)
        eval_model.head = torch.nn.Linear(eval_model.head.in_features, 2)
        # Load checkpoint
        eval_model.load_state_dict(torch.load(eval_model_path, map_location=device))
        eval_model.to(device)

    log_file_path = ""
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file_path):
        open(log_file_path, 'w').close()
    with open(log_file_path, "a") as log_file:    
        # For each model_path
        for checkpoint in model_paths:
            # Extract the epoch name from the checkpoint path
            if "epochs" in checkpoint:
                epoch_folder = get_last_folder_name(checkpoint)
                output_dir = f"___/{epoch_folder}"
            else:
                # Get the last folder or string in the checkpoint path
                last_part = os.path.basename(checkpoint)
                output_dir = f"___/{last_part}_{concept}"
            
            # Create the output directory for this checkpoint
            os.makedirs(output_dir, exist_ok=True)

            # Load the pipeline for the current checkpoint
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16
            ).to(f"cuda:{args.gpu}")
            pipe.safety_checker = None  # Disable NSFW filtering

            clip_scores = []

            # Generate and save images for each prompt
            for prompt in prompts:
                # Generate 5 images per prompt
                images = pipe(prompt, num_images_per_prompt=5).images
                # Ensure the filename is safe
                safe_prompt = re.sub(r'[\\/*?:"<>|]', "_", prompt)
                safe_prompt = safe_prompt[:50]  # Limit to 50 characters
                for idx, image in enumerate(images):
                    # Save each image with a unique filename
                    image_filename = f"{safe_prompt}_{idx}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)

                    # Compute CLIP score
                    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                        clip_score = logits_per_image.item()
                        clip_scores.append(clip_score)
            
            # Compute average CLIP score
            average_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
            
            if args.theme==None:
                accuracy = evaluate(model=eval_model, processor=eval_processor, dataset_path=output_dir, device=device)
            elif args.theme.lower()=="nudity":
                accuracy = evaluate_nudity(dataset_path=output_dir)
            else:
                accuracy = evaluate_style(model=eval_model, dataset_path=output_dir, device=device)
            
            print(f"Accuracy for model {checkpoint}: {accuracy}")
            print(f"Average CLIP score for model {checkpoint}: {average_clip_score}")

            # Write the results to the log file
            log_file.write(f"Model: {checkpoint}\n")
            log_file.write(f"Concept: {concept}\n")
            log_file.write(f"Accuracy: {accuracy}\n")
            log_file.write(f"Average CLIP score: {average_clip_score}\n")
            log_file.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()