import sys; sys.path.insert(0, "..")
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
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

dataset_name = "EOYS_images"
# Specify the folder containing the images
image_folder = f"../Datasets/{dataset_name}/ALLSTIMULI"

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create a dropdown menu with the image file names
selected_image_file = st.selectbox("Select an Image", image_files)

# Load the selected image using PIL
if selected_image_file:
    image_path = os.path.join(image_folder, selected_image_file)
    image = Image.open(image_path)

    # Display the selected image
    st.image(image, caption=selected_image_file)

    image_np = np.array(image)

    # Load or create the centerbias template
    try:
        centerbias_template = np.load('../centerbias_mit1003.npy')
    except FileNotFoundError:
        st.warning("Centerbias file not found, using a uniform centerbias.")
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
    center_x = image_width // 2
    center_y = image_height // 2

    # 使用图像正中心作为固定的四个点坐标
    points = [(center_x, center_y), (center_x, center_y), (center_x, center_y), (center_x, center_y)]

    # 提取最后四个点的x和y坐标
    fixation_history_x = np.array([p[0] for p in points])
    fixation_history_y = np.array([p[1] for p in points])

    # Convert the inputs to tensors
    DEVICE = "cpu"  # or "cuda" if available
    image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
    x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)

    # Add a button to run the model
    if st.button("Run Model"):
        # Generate the log density prediction if model is defined
        if model is not None:
            log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
            predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
            
            f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
            axs[1].imshow(image)
            history_points = np.array(points)
            axs[1].plot(history_points[:, 0], history_points[:, 1], 'o-', color='red')
            axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
            axs[1].set_axis_off()
            axs[0].imshow(image_np, alpha=0.5)
            heatmap = axs[0].imshow(predicted_heatmap, cmap='jet', alpha=0.6) 
            axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
            axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
            axs[0].set_axis_off()

            st.pyplot(f)  # Display the plot in Streamlit
        else:
            st.warning("Model is not defined. Please provide a valid model.")
