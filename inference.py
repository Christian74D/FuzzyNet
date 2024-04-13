from model import load_model
import matplotlib.pyplot as plt
from data import X, no_classes
from constants import inference_img_path
import torch
import numpy as np


print("Loading saved model..")
device = 'cpu'
model = load_model(device)


maxv, minv = [max(i) for i in X],  [min(i) for i in X]

"""
for i, network_list in enumerate(model.networks):
    print(f"Decision maker {i}:")
    for j, submodule in enumerate(network_list):
        #print(f"  Submodule {j}: {submodule}")
        print(submodule(torch.tensor([1.0])))

"""

# Prepare inputs within specified ranges [minv[i], maxv[i]] for each submodule
num_samples = 100  # Number of samples within each range
input_ranges = [(minv[i], maxv[i]) for i in range(len(minv))]  # List of input ranges for each submodule

# Define number of classes (assuming each submodule outputs class probabilities)
num_classes = no_classes

# Plotting the outputs of each submodule in separate graphs (subplots)
num_decisions = len(model.networks)
num_submodules_per_decision = max(len(network_list) for network_list in model.networks)

plt.figure(figsize=(12, 4 * num_decisions))  # Adjust figure size based on the number of decisions

for i, network_list in enumerate(model.networks):
    dm_title = f"dm {i}"
    print(f"Decision maker {i} adding to plot..")
    num_submodules = len(network_list)

    for j, submodule in enumerate(network_list):
        subplot_idx = i * num_submodules_per_decision + j + 1

        plt.subplot(num_decisions, num_submodules_per_decision, subplot_idx)
        plt.title(f"{dm_title}, {j}")
        plt.xlabel('ip')
        plt.ylabel('prob')

        input_range = input_ranges[j]
        inputs = torch.linspace(input_range[0], input_range[1], num_samples)  # Generate input values

        # Compute class probabilities for each input value
        class_probabilities = []
        for input_value in inputs:
            output = submodule(input_value.unsqueeze(0))  # Pass single input value to submodule
            class_probabilities.append(output.detach().squeeze().numpy())  # Convert output tensor to numpy array

        class_probabilities = np.array(class_probabilities)  # Convert list of numpy arrays to a numpy array

        # Plot class probability curves for each class
        for class_idx in range(num_classes):
            plt.plot(inputs.numpy(), class_probabilities[:, class_idx], label=f'{class_idx}')




plt.savefig(inference_img_path)  
print(f"Plot saved as: {inference_img_path}")
plt.tight_layout()
plt.show()
