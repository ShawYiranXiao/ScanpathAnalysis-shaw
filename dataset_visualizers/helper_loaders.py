import pandas as pd
import scipy
import numpy as np
from PIL import Image

def preprocess_mit1003(data_path, image_path):
    subject_data = scipy.io.loadmat(data_path)
    try:
        selected_key = ''
        # Extract and filter the fixation data
        for key in subject_data.keys():
            if key.startswith('__'):
                continue
            else:
                selected_key = key
                break

        data_array = subject_data[selected_key][0][0]
        points = data_array[data_array.dtype.names.index('DATA')][0][0][2] if 'DATA' in data_array.dtype.names else -1
    except IndexError:
        # st.warning(f"Index error while accessing {subject} scanpath data. Skipping this entry.")
        points = np.empty((0, 2))  # Fallback for missing data

    # Extract x, y coordinates and filter valid points
    filtered_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
    return filtered_points[:, 0], filtered_points[:, 1]
        # x, y = filtered_points[:, 0], filtered_points[:, 1]

def preprocess_eoys(data_path, image_path):

    # print(data_path, image_path)
    # Load the image to get its width
    image = Image.open(image_path)
    image_width = image.width

    # Fixed height for the transformation
    image_height = 1080

    # Calculate the screen width using a 16:9 aspect ratio
    screen_width = image_height * 16.0 / 9.0

    # Calculate starting x offset for the image in the larger screen area
    starting_x = (screen_width - image_width) / 2
    left_ratio = starting_x / screen_width

    # Load CSV file without headers
    data = pd.read_csv(data_path, header=None)

    # Filter the points based on ratios relative to the image boundaries
    filtered_data = data[(data[1] >= left_ratio) & (data[1] <= 1 - left_ratio) & (data[2] >= 0) & (data[2] <= 1)]

    # Transform x and y to align with image pixel dimensions
    transformed_x = filtered_data[1].values * screen_width - starting_x
    transformed_y = filtered_data[2].values * image_height

    # print(transformed_y)
    return transformed_x, transformed_y
    
    
data_loaders = {
    "preprocess_mit1003": preprocess_mit1003,
    "preprocess_eoys": preprocess_eoys
}
