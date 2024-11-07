import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import deepgaze_pytorch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

# model eats imgs and 4 points

dataset_name = "EOYS_images"
# Specify the folder containing the images
image_folder = f"Datasets/{dataset_name}/ALLSTIMULI"

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load the selected image using PIL
# if selected_image_file:
image_path = os.path.join(image_folder, image_files[0])
image = Image.open(image_path)

image_np = np.array(image)
try:
    centerbias_template = np.load('centerbias_mit1003.npy')
except FileNotFoundError:
    centerbias_template = np.zeros((1024, 1024))

# Rescale the centerbias to match the image size
centerbias = zoom(
    centerbias_template,
    (image_np.shape[0] / centerbias_template.shape[0],
        image_np.shape[1] / centerbias_template.shape[1]),
    order=0, mode='nearest'
)
centerbias -= logsumexp(centerbias)  # Renormalize

# 获取图片的尺寸
image_width, image_height = image.size
print(image_width, image_height)
image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
image_width, image_height = image.size
center_x = image_width // 2
center_y = image_height // 2

points = [(center_x, center_y), (center_x, center_y), (center_x, center_y), (center_x, center_y)]
fixation_history_x = np.array([p[0] for p in points])
fixation_history_y = np.array([p[1] for p in points])
x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)
log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]




breakpoint()
