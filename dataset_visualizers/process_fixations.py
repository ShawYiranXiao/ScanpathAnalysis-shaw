import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import zoom
from scipy.special import logsumexp

# Add the deepgaze_pytorch path
sys.path.insert(0, "/Users/shaw/ScanpathAnalysis/deepgaze_pytorch")
print(sys.path)  # Add this line for debugging purposes
import deepgaze_pytorch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

def process_image(image_path, output_dir, num_points=4, num_iterations=10):
    # Load the image
    image = Image.open(image_path)
    image_np = np.array(image)
    image_width, image_height = image.size

    # Create a uniform centerbias (all zeros)
    centerbias = np.zeros((image_np.shape[0], image_np.shape[1]))

    # List to hold heatmaps for averaging
    heatmaps = []

    for _ in range(num_iterations):
        # Generate random fixation points that cover the entire image
        points = [(np.random.randint(0, image_width), np.random.randint(0, image_height)) for _ in range(num_points)]

        # Extract x and y coordinates of the fixation points
        fixation_history_x = np.array([p[0] for p in points])
        fixation_history_y = np.array([p[1] for p in points])

        # Convert the inputs to tensors
        image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)
        x_hist_tensor = torch.tensor([fixation_history_x]).float().to(DEVICE)
        y_hist_tensor = torch.tensor([fixation_history_y]).float().to(DEVICE)

        # Generate the log density prediction
        log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
        heatmaps.append(predicted_heatmap)

        # Save intermediate scanpath and result images
        intermediate_dir = os.path.join(output_dir, 'intermediate_results')
        os.makedirs(intermediate_dir, exist_ok=True)

        # Save one sample scanpath on the image
        sample_scanpath_path = os.path.join(intermediate_dir, f'scanpath_iteration_{_}.jpg')
        plt.imshow(image)
        points_np = np.array(points)
        plt.plot(points_np[:, 0], points_np[:, 1], 'o-', color='red')
        plt.scatter(fixation_history_x[-1], fixation_history_y[-1], 10, color='yellow', zorder=100)
        plt.axis('off')
        plt.savefig(sample_scanpath_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save the prediction result (feature map layered on the original image)
        sample_prediction_path = os.path.join(intermediate_dir, f'prediction_iteration_{_}.jpg')
        plt.imshow(image_np, alpha=0.5)
        plt.imshow(predicted_heatmap, cmap='jet', alpha=0.6)
        plt.axis('off')
        plt.savefig(sample_prediction_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Average the heatmaps to reduce the influence of individual fixation points
    final_heatmap = np.mean(heatmaps, axis=0)

    # Save original image
    original_image_path = os.path.join(output_dir, 'original_image.jpg')
    image.save(original_image_path)

    # Save final feature map as image
    final_heatmap_path = os.path.join(output_dir, 'final_feature_map.png')
    plt.imshow(final_heatmap, cmap='jet')
    plt.axis('off')
    plt.savefig(final_heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save original image with feature map overlay
    overlay_image_path = os.path.join(output_dir, 'original_with_feature_map.jpg')
    plt.imshow(image_np, alpha=0.5)
    plt.imshow(final_heatmap, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.savefig(overlay_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save final feature map raw values (log density -> array) as text file
    raw_values_path = os.path.join(output_dir, 'final_feature_map_values.txt')
    np.savetxt(raw_values_path, final_heatmap)

def main():
    parser = argparse.ArgumentParser(description='Batch process images for fixation heatmap generation.')
    parser.add_argument('--dirs', nargs='+', required=True, help='Directories containing images to process')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save results')
    args = parser.parse_args()

    # Loop over each input directory
    for input_dir in args.dirs:
        # Extract folder name to create corresponding output sub-folder
        folder_name = os.path.basename(input_dir.rstrip('/'))
        output_folder = os.path.join(args.output, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Get list of images in the current input directory
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            output_dir = os.path.join(output_folder, os.path.splitext(image_file)[0])
            os.makedirs(output_dir, exist_ok=True)

            # Process each image using Method 3: Random Fixation Points Averaging
            process_image(image_path, output_dir)

if __name__ == '__main__':
    main()
