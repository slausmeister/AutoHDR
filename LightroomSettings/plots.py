import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

labels = ["Exposure", "Contrast", "Highlights", "Shadows", "Whites", "Blacks", "Vibrance", "Saturation"]

def plot_label_histograms(data_loader, label_names):
    # Validate all label names
    for label_name in label_names:
        if label_name not in labels:
            raise ValueError(f"Label name {label_name} is not in the list of known labels.")

    num_labels = len(label_names)
    
    # Prepare subplots with 2 rows and 4 columns
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    axes = axes.flatten()  # Flatten the 2D array to 1D for easier indexing
    
    # Collect the specified label values for each label
    for i, label_name in enumerate(label_names):
        label_index = labels.index(label_name)
        label_values = []
        for img_tensors, img_labels in data_loader:
            label_values.extend(img_labels[:, label_index].numpy())
        
        # Plot the histogram on the appropriate subplot
        axes[i].hist(label_values, bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Histogram of {label_name}")
        axes[i].set_xlabel(label_name)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)
        axes[i].set_xlim(-1, 1)
    
    # Hide any unused subplots
    for j in range(num_labels, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def bar_plot(array):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), array, color='blue', alpha=0.7)
    plt.ylabel("Standard Deviation")
    plt.xticks(range(len(labels)), labels, rotation=300)
    plt.grid(True)
    plt.show()



def plot_bias_and_linear_coefficients(model):
    weights = model.classifier[0].weight.detach().cpu().numpy()
    bias = model.classifier[0].bias.detach().cpu().numpy()
    
    # Calculate linear coefficients
    linear_coeffs = [weights[i].mean() for i in range(8)]
    abs_coeffs = [abs(weights[i]).mean() for i in range(8)]

    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1]})

    # Plot the bias values in the first subplot
    ax1.bar(range(0, 8), bias, color='red', alpha=0.5)
    ax1.set_xticks([])
    ax1.set_ylabel('Bias Value')

    # Plot the linear coefficients in the second subplot
    ax2.bar(range(0, 8), linear_coeffs, color='blue', alpha=0.5)
    ax2.set_xticks(range(0, 8))
    ax2.set_xticklabels(labels, rotation=300)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Mean Weight Value')

    # Adjust layout to make plots touch
    plt.subplots_adjust(hspace=0)

    # Show the plot
    plt.show()