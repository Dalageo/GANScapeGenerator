# Import the required dependencies
import os
from PIL import Image

# Dataset conversion settings
img_resolution="256x256"    # Image resolution for the dataset
img_num="2000"              # Number of images to process

# Training settings
# gamma and mapping layers were set based on https://github.com/justinpinkney/awesome-pretrained-stylegan3
gpu = "1"                   # Number of GPUs
snap = "1"                  # Snapshot interval (Every 1 tick)
kimg_per_tick = "5"         # Number of kimages per tick
batch = "16"                # Batch size for training
batch_gpu = "16"            # Batch size per GPU
gamma = "2"                 # Gamma value for learning rate
mapping_layers = "2"        # Number of mapping layers
kimg = "50"                 # Total number of kimages for training
cfg = "stylegan3-t"         # Configuration ("stylegan3-t")
metric = ""                 # Metric setting for evaluation(empty in this case)

# Some available options for pretrained StyleGAN3 models can be found at:
# 1. NVIDIA NGC Catalog:
#    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files
# 2. Hugging Face (Search for StyleGAN3 models):
#    https://huggingface.co/models?search=stylegan-3

# Dataset and output directories
original_data = r'<path_to_your_original_data>'                                # Path to original data
train_data = r"<path_to_your_training_data>"                                   # Path to train data
stylegan3_dir=r"<path_to_the_cloned_stylegan3_directory>"                      # Path to cloned stylegan3 directory (https://github.com/NVlabs/stylegan3)
out_dir = r"<path_to_the_directory_where_training_results_will_be_saved"       # Path to output directory for saving results
pretrained_model = r"<path_to_model_checkpoint>"                               # Path to the StyleGAN3 model checkpoint

# Print current directory
os.chdir(stylegan3_dir)
print(f"Current directory: {os.getcwd()}")

# Optionally command to display help for converting an image dataset into a format compatible with StyleGAN3
# os.system("python dataset_tool.py --help")

# Convert the dataset into a compatible format
os.system(f'python dataset_tool.py --source={original_data} '
          f'--dest={train_data} ' 
          f'--resolution={img_resolution} ' 
          f'--max-images={img_num} '
          )

# Dry-run the training script
os.system(f'python train.py --outdir={out_dir} '
          f'--data={train_data} '
          f'--cfg={cfg} --gpus={gpu} --batch={batch} --gamma={gamma} '
          f'--batch-gpu={batch_gpu} --map-depth={mapping_layers} --snap={snap} --kimg={kimg} --tick={kimg_per_tick} '
          f'--resume={pretrained_model} '
          '--dry-run'
          )

# Start training
# If you encounter any errors or if the training process stalls, go to
# C:\Users\<your_username>\AppData\Local\torch_extensions and delete the 'torch_extensions' folder.
os.system(f'python train.py --outdir={out_dir} '
          f'--data={train_data} '
          f'--cfg={cfg} --gpus={gpu} --batch={batch} --gamma={gamma} '
          f'--batch-gpu={batch_gpu} --snap={snap} --kimg={kimg} --tick={kimg_per_tick} '
          f'--resume={pretrained_model} '
          f'--metrics={metric} '
          )

# Image Generation Settings
finetuned_model = r"<path_to_fine_tuned_model_checkpoint>"                     # Model fine-tuned on your dataset, starting from the pretrained checkpoint
gen_outdir = r"<path_to_generated_images_directory>"                           # Directory to save generated images
truncation_psi = "0.5"                                                         # Controls truncation (higher values lead to more diverse images)
noise_mode = "const"                                                           # Noise mode ('const', 'random', 'none')
rotate = "0"                                                                   # Rotation parameter (if applicable)
seeds = "0-50"                                                                 # Range of seeds to generate images (e.g., from seed 0 to seed 50)

# Generate the Images
os.system(f'python gen_images.py '
          f'--network={finetuned_model} '
          f'--outdir={gen_outdir} '
          f'--trunc={truncation_psi} '
          f'--noise-mode={noise_mode} '
          f'--rotate={rotate} '
          f'--seeds={seeds}')