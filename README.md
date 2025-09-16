# The Illusion of Unlearning: The Unstable Nature of Machine Unlearning in Text-to-Image Diffusion Models (Accepted in CVPR 2025)

## Getting Started

Follow these steps to set up the project locally on your system.

### Prerequisites

Make sure you have the following installed on your system:
- Git
- Anaconda or Miniconda

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Create a new conda environment using the provided environment.yml file:
   ```bash
   conda env create -f environment.yml

3. Activate the newly created environment:
   ```bash
   conda activate finetune

## Running the `finetuning_individual_CLIP.py` Script

The script `finetuning_individual_CLIP.py` is used for fine-tuning an individual CLIP model. Follow the steps below to prepare and execute the script:

### Prerequisites

1. **Pretrained Model**: 
   - The `--pretrained_model_name_or_path` argument should point to the location of the unlearned model in the **Diffusers** format.

2. **Training Data**: 
   - Training data can be generated using the instructions provided in the supplementary document.
   - Once the data is generated, provide the path to the training data using the `--train_data_dir` argument.

3. **Validation Prompts**: 
   - Validation prompts are provided in the supplementary material.
   - Save them in a `.txt` file and specify the path to this file using the `--validation_prompts` argument.

4. **Additional Arguments**:
   - Other arguments are explained in detail in the script. Refer to the comments in the code for more information.

---
   
### Command to Run the Script

Use the following command to execute the script:

   ```bash
   accelerate launch finetuning_individual_CLIP.py \
     --pretrained_model_name_or_path="<path_to_unlearned_model>" \
     --train_data_dir="<path_to_train_data>" \
     --caption_column="prompt" \
     --use_ema \
     --resolution=512 \
     --center_crop \
     --random_flip \
     --gradient_checkpointing \
     --mixed_precision="fp16" \
     --learning_rate=1e-05 \
     --max_grad_norm=1 \
     --lr_scheduler="constant" \
     --lr_warmup_steps=0 \
     --train_batch_size=10 \
     --gradient_accumulation_steps=1 \
     --curriculum="50,100,150,200,250,300,350,400,450,500" \
     --clip_threshold=0.31 \
     --validation_prompts="<path_to_validation_prompts_txt>" \
     --log_file="<path_to_log_file>" \
     --save_model=True \
     --output_dir="<path_to_output_dir>"
   ```

## Running Fine-Tuning Scripts

### `finetuning_individual_Classifier.py`

This script is similar to `finetuning_individual_CLIP.py`, except that it uses a classifier instead of CLIP for fine-tuning. Unfortunately, due to size constraints, the classifier weights are not included in this repository. However, detailed instructions to replicate the classifier weights are provided in the supplementary material. These weights will be made available soon.

The arguments used are similar to `finetuning_individual_CLIP.py`, with the following modification:
- Replace `--clip_threshold` with `--bc_threshold=0.3`.

### `finetuning_sequential.py`

This script allows fine-tuning with both constraints (CLIP and classifier) applied sequentially. Using this script, you can determine the **Revival Point** (discussed in the paper). Tables 7 and 8 in the supplementary material were generated using this code.

### Command to Run `finetuning_sequential.py`

Use the following command to run the script:

```bash
accelerate launch finetuning_sequential.py \
  --pretrained_model_name_or_path="<path_to_unlearned_model>" \
  --train_data_dir="<path_to_train_data>" \
  --caption_column="prompt" \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --train_batch_size=10 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=8 \
  --output_dir="<path_to_output_dir>" \
  --curriculum="50,100,150,200,250,300,350,400,450,500" \
  --curriculum_checkpoints="4,7" \
  --model_path="<path_to_model>" \
  --log_file="<path_to_log_file>" \
  --clip_threshold=0.31 \
  --bc_threshold=0.3 \
  --validation_prompts="<path_to_validation_prompts_txt>"
```

---

## Evaluation script -

```bash
python eval_model.py \
  "<checkpoint path>" \
  "<Model path>" \
  "<concept>" \
  --prompt_file="<path_to_validation_prompts_txt>" \
  --theme="" \    # No need of this argument for objects/celebrity. In case of style just --theme. In Nudity --theme="Nudity".
  --gpu <GPU_ID>
```

---

## Citation
To cite our work you can use the following:
```
@inproceedings{Naveen2025Illusion,
  title     = {The Illusion of Unlearning: The Unstable Nature of Machine Unlearning in Text-to-Image Diffusion Models},
  author    = {Naveen George, Karthik Nandan Dasaraju , Rutheesh Reddy Chittepu, Konda Reddy Mopuri},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {July}
  year      = {2025}
}


```
