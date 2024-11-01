import sys; sys.path.insert(0, "..")
import streamlit as st
from streamlit_drawable_canvas import st_canvas
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

#     # Display the selected image
#     st.image(image, caption=selected_image_file)

# # Load the example image
# image = Image.fromarray(face())
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
    print(image_width, image_height)

    # 设置 Canvas，确保宽高和图片匹配
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # 半透明填充
        stroke_width=3,
        stroke_color="red",
        background_image=image,  # 使用 PIL 图像作为背景
        update_streamlit=True,
        height=image_height/3,  # 使用原始图片高度
        width=image_width/3,    # 使用原始图片宽度
        drawing_mode="point",  # 点模式
        key="canvas",
    )

    # Session state to store points
    # if "points" not in st.session_state:
    #     st.session_state.points = []
    st.session_state.points = []
    # Collect points from the canvas
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                st.session_state.points.append((obj["left"]*3, obj["top"]*3))

    # Prepare model inputs if exactly four points are selected
    if len(st.session_state.points) >= 4:
        # Extract x and y coordinates
        fixation_history_x = np.array([p[0] for p in st.session_state.points[-4:]])
        fixation_history_y = np.array([p[1] for p in st.session_state.points[-4:]])

        # Convert the inputs to tensors
        
        image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
        x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
        y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)

        # Generate the log density prediction
        log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
        
        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        axs[1].imshow(image)
        history_points = np.array(st.session_state.points)
        print(history_points)
        print(history_points[:,0], history_points[:,1])
        axs[1].plot(history_points[:, 0], history_points[:,1], 'o-', color='red')
        axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
        axs[1].set_axis_off()
        axs[0].imshow(image_np, alpha=0.5)
        heatmap = axs[0].imshow(predicted_heatmap, cmap='jet', alpha=0.6) 
        # axs[0].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
        axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
        axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
        axs[0].set_axis_off()

        st.pyplot(f)  # Display the plot in Streamlit

    else:
        st.warning("Add more points until you have exactly four.")
