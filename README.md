# TIU

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

