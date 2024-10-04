
import mmcv
import numpy as np
import torch
import torch.nn as nn
def get_all_scenes(path):

    data =mmcv.load(path)    
    scenes = np.unique([d["scene_token"] for d in data["infos"]]).tolist()

    return scenes


def add_layer_channel_correction(model, output_channels=256, state_dict_path="pretrained/bevfusion-seg.pth"):
    # Get the current layers of the downsample module
    current_layers = list(model.encoders.camera.vtransform.downsample.children())
    
    # Get the number of input channels from the last convolutional layer
    input_channels = current_layers[-3].out_channels  # Assuming the last Conv2d is 3 positions from the end
    
    # Create the new layers you want to add
    new_conv = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    new_relu = nn.ReLU(inplace=True)
    
    # Add the new conv layer at index 9 (after the last ReLU)
    current_layers.insert(9, new_conv)
    
    # Add the new ReLU at index 10
    current_layers.insert(10, new_relu)
    
    # Create a new Sequential module with the updated layers
    new_downsample = nn.Sequential(*current_layers)
    
    model.encoders.camera.vtransform.downsample = new_downsample
    
    # Load the state dict if a path is provided
    if state_dict_path:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict, strict=False)
    
    # Initialize the new conv layer if it's not in the loaded state dict
    if 'encoders.camera.vtransform.downsample.9.weight' not in model.state_dict():
        model_state_dict = model.state_dict()
        model_state_dict['encoders.camera.vtransform.downsample.9.weight'] = new_conv.weight
        model.load_state_dict(model_state_dict)
    
    return model

def filter_and_save_first_10_scenes(input_path, output_path):
    # Load the original data
    data = mmcv.load(input_path)
    
    # Get the first 10 unique scene tokens
    first_10_scenes = np.unique([d["scene_token"] for d in data["infos"]])[:10].tolist()
    
    # Filter the data to include only the first 10 scenes
    filtered_infos = [info for info in data["infos"] if info["scene_token"] in first_10_scenes]
    
    # Overwrite the 'infos' key with the filtered data
    data["infos"] = filtered_infos
    
    # Save the filtered data
    mmcv.dump(data, output_path)
    
    print(f"Filtered dataset with the first 10 scenes saved to {output_path}")
    print(f"Number of samples in the filtered dataset: {len(filtered_infos)}")


#filter_and_save_first_10_scenes("/root/quick/Kopie von nuscenes_infos_val.pkl", "/root/bevfusion/data/nuscenes/nuscenes_infos_test_10.pkl")
#filter_and_save_first_10_scenes("/root/bevfusion/data/nuscenes/nuscenes_infos_val.pkl", "/root/bevfusion/data/nuscenes/nuscenes_infos_test_10.pkl")

import json
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_and_save_dicts(dict1, dict2, label1, label2, title, xlabel, ylabel, output_file):
    # Extract keys and values from both dictionaries
    x1, y1 = zip(*sorted(dict1.items()))
    x2, y2 = zip(*sorted(dict2.items()))

    # Convert x-values to float (assuming they're numeric)
    x1 = [float(x) for x in x1]
    x2 = [float(x) for x in x2]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label=label1, marker='o')
    plt.plot(x2, y2, label=label2, marker='s')

    # Customize the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved as {output_file}")

# Load data from JSON files
results_dict_img = load_json('results_dict_img.json')
results_dict_points = load_json('results_dict_points.json')

# Plot the data and save the figure
plot_and_save_dicts(
    results_dict_img, results_dict_points, 
    "Image Results", "Points Results", 
    "Comparison of Image and Points Results", 
    "X-axis Label", "Y-axis Label",
    "comparison_plot.png"  # Output file name
)