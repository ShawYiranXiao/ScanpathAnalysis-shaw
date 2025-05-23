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
import pandas as pd

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

# 全局历史数据记录
if "history_rows" not in st.session_state:
    st.session_state.history_rows = []
if "heatmaps" not in st.session_state:
    st.session_state.heatmaps = []
if "predicted_points" not in st.session_state:
    st.session_state.predicted_points = []

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

    # 从 canvas 实时读取点坐标（不使用 session state 累加）
    clicked_points = []
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                clicked_points.append((obj["left"]*3, obj["top"]*3))

    idx = len(st.session_state.predicted_points)

    # Prepare model inputs if至少五个点
    if len(clicked_points) >= idx + 5:
        # Extract x and y coordinates from the 4-point history
        history_points = clicked_points[idx:idx+4]
        fixation_history_x = np.array([p[0] for p in history_points])
        fixation_history_y = np.array([p[1] for p in history_points])
        real_point = clicked_points[idx + 4]

        # Convert the inputs to tensors
        image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
        x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
        y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)

        # Generate the log density prediction
        log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]

        # 找到预测点
        pred_y, pred_x = np.unravel_index(np.argmax(predicted_heatmap), predicted_heatmap.shape)
        pred_point = (float(pred_x), float(pred_y))

        # 记录历史、预测、真实点
        row = {
            "Index": idx,
            "Predicted_x": pred_point[0],
            "Predicted_y": pred_point[1],
            "Real_x": real_point[0],
            "Real_y": real_point[1],
            "Image_Width": image_width,
            "Image_Height": image_height
        }

        # Add historical points P1~P4 as flat columns
        for i, (hx, hy) in enumerate(history_points):
            row[f"H{i+1}_x"] = hx
            row[f"H{i+1}_y"] = hy

        st.session_state.history_rows.append(row)
        st.session_state.heatmaps.append(predicted_heatmap)
        st.session_state.predicted_points.append(pred_point)

        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        axs[1].imshow(image)
        points_to_plot = clicked_points[idx:idx+5]
        history_points_np = np.array(points_to_plot)
        print(history_points_np)
        print(history_points_np[:,0], history_points_np[:,1])
        axs[1].plot(history_points_np[:, 0], history_points_np[:,1], 'o-', color='red')
        axs[1].scatter(real_point[0], real_point[1], 10, color='yellow', zorder=100)
        axs[1].set_axis_off()
        axs[0].imshow(image_np, alpha=0.5)
        heatmap = axs[0].imshow(predicted_heatmap, cmap='jet', alpha=0.6) 
        axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
        axs[0].scatter(pred_point[0], pred_point[1], 10, color='blue', zorder=100)
        axs[0].scatter(real_point[0], real_point[1], 10, color='yellow', zorder=101)
        axs[0].set_axis_off()

        st.pyplot(f)  # Display the plot in Streamlit

    else:
        st.warning("Add more points until you have at least 5 to start generating predictions.")

# 导出函数定义
def export_csv_files():
    rows = st.session_state.get("history_rows", [])
    heatmaps = st.session_state.get("heatmaps", [])
    if not rows:
        st.warning("No data to export.")
        return

    # 直接将结构化 row 数据写入 DataFrame
    df = pd.DataFrame(rows)

    columns = [
        "Index", "Predicted_x", "Predicted_y", "Real_x", "Real_y",
        "Image_Width", "Image_Height",
        "H1_x", "H1_y", "H2_x", "H2_y", "H3_x", "H3_y", "H4_x", "H4_y"
    ]
    df = pd.DataFrame(rows, columns=columns)
    
    df.to_csv("predictions_log.csv", index=False)
    st.success("Saved predictions_log.csv")

    # 导出每个 heatmap 为 CSV 文件
    for i, heatmap in enumerate(heatmaps):
        np.savetxt(f"heatmap_{i}.csv", heatmap, delimiter=",")
    st.success("Saved all heatmaps as CSV.")


# 导出按钮（始终显示在页面底部）
if st.button("\U0001F4BE Export CSV files"):
    export_csv_files()
