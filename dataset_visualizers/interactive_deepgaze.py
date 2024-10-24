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

        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        axs[1].imshow(image)
        history_points = np.array(st.session_state.points)
        print(history_points)
        print(history_points[:,0], history_points[:,1])
        axs[1].plot(history_points[:, 0], history_points[:,1], 'o-', color='red')
        axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
        axs[1].set_axis_off()
        axs[0].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
        axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
        axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
        axs[0].set_axis_off()

        st.pyplot(f)  # Display the plot in Streamlit

    else:
        st.warning("Add more points until you have exactly four.")
# import sys; sys.path.insert(0, "..")
# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# import numpy as np
# from scipy.misc import face
# from scipy.ndimage import zoom
# from scipy.special import logsumexp
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# from io import BytesIO
# import os
# import base64
# import json

# import deepgaze_pytorch

# DEVICE = 'cpu'

# # Initialize the model
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

# # Function to convert a PIL image to base64 string
# def pil_to_base64(img):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return img_str

# # Set the base folder for datasets
# base_folder = '../Datasets/'

# # Load dataset configurations
# with open('dataset_config.json', 'r') as file:
#     dataset_configs = json.load(file)

# # Sidebar configuration to select datasets and images
# datasets = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))][::-1]
# selected_dataset = st.sidebar.selectbox("Dataset", datasets)

# stimuli_folder = os.path.join(base_folder, selected_dataset, "ALLSTIMULI")
# image_files = [f for f in os.listdir(stimuli_folder) if f.endswith(('.jpeg', '.jpg'))]

# # Implement pagination for the gallery
# batch_size = 10  # Number of thumbnails to load at a time
# num_batches = (len(image_files) + batch_size - 1) // batch_size

# # Sidebar slider to select the current batch
# batch_index = st.sidebar.slider("Select Batch", 0, num_batches - 1, 0)
# start_idx = batch_index * batch_size
# end_idx = min(start_idx + batch_size, len(image_files))

# # Load thumbnails for the current batch
# thumbnails = []
# for img_file in image_files[start_idx:end_idx]:
#     img = Image.open(os.path.join(stimuli_folder, img_file))
#     img.thumbnail((100, 100))
#     thumbnails.append(img)

# # Sidebar for selecting an image
# st.sidebar.write("### Image Thumbnails (Scroll)")
# thumbnail_cols = st.sidebar.columns(5)

# if "selected_image_idx" not in st.session_state:
#     st.session_state.selected_image_idx = start_idx  # Initialize with the first image

# for i, (img, img_file) in enumerate(zip(thumbnails, image_files[start_idx:end_idx])):
#     with thumbnail_cols[i % 5]:
#         img_base64 = pil_to_base64(img)
#         st.markdown(
#             f"""
#             <div style='border: 1px solid #ccc; width: 100px; height: 100px;'>
#                 <img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100px;"/>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#         if st.button(f"Select {i}", key=f"btn_{start_idx + i}"):
#             st.session_state.selected_image_idx = start_idx + i
#             st.rerun()

# selected_image_idx = st.session_state.selected_image_idx
# selected_image_path = os.path.join(stimuli_folder, image_files[selected_image_idx])
# selected_image = Image.open(selected_image_path)

# st.write(f"## Selected Image: {image_files[selected_image_idx]}")

# # Load the selected image into a numpy array for processing
# image_np = np.array(selected_image)

# # Set up the centerbias template
# try:
#     centerbias_template = np.load('../centerbias_mit1003.npy')
# except FileNotFoundError:
#     st.warning("Centerbias file not found, using a uniform centerbias.")
#     centerbias_template = np.zeros((1024, 1024))

# # Rescale the centerbias to match the image size
# centerbias = zoom(
#     centerbias_template,
#     (image_np.shape[0] / centerbias_template.shape[0],
#      image_np.shape[1] / centerbias_template.shape[1]),
#     order=0, mode='nearest'
# )
# centerbias -= logsumexp(centerbias)  # Renormalize

# # Display the selected image in a canvas
# print(selected_image.size)
# image_width, image_height = selected_image.size

# st.subheader("Interactive Canvas with Polyline Overlay")
# canvas_result = st_canvas(
#     fill_color="rgba(255, 0, 0, 0.3)",  
#     stroke_width=3,
#     stroke_color="red",
#     background_image=selected_image,  # Use selected image
#     update_streamlit=True,
#     height=image_height/3,
#     width=image_width/3,
#     drawing_mode="point",
#     key="canvas",
# )

# # Initialize or update points in session state
# # if "points" not in st.session_state:
# st.session_state.points = []

# # Collect points from the canvas
# if canvas_result.json_data is not None:
#     for obj in canvas_result.json_data["objects"]:
#         if obj["type"] == "circle":
#             st.session_state.points.append((obj["left"]*3, obj["top"]*3))

# # Keep only the latest four points
# # if len(st.session_state.points) > 4:
# #     st.session_state.points = st.session_state.points[-4:]

# # Generate polyline overlay if exactly four points are selected
# if len(st.session_state.points) >= 4:
#     fixation_history_x = np.array([p[0] for p in st.session_state.points[-4:]])
#     fixation_history_y = np.array([p[1] for p in st.session_state.points[-4:]])

#     # Convert the inputs to tensors
#     image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
#     centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
#     x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
#     y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)

#     # Generate the log density prediction
#     log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

#     f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
#     axs[0].imshow(selected_image)
#     history_points = np.array(st.session_state.points)
#     print(history_points)
#     print(history_points[:,0], history_points[:,1])
#     axs[0].plot(history_points[:, 0], history_points[:,1], 'o-', color='red')
#     axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
#     axs[0].set_axis_off()
#     axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
#     axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
#     axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
#     axs[1].set_axis_off()

#     st.pyplot(f)

# # Button to clear points
# if st.button("Clear Points"):
#     st.session_state.points = []
#     canvas_result.background_image= selected_image
#     st.rerun()
