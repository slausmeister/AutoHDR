import numpy as np
import scipy.ndimage
import torch


def smoothing(labels, method='moving_average', window_size=5, sigma=2):
    if method == 'moving_average':
        # Apply moving average smoothing
        if window_size > 0:
            smoothed_labels = np.convolve(labels, np.ones(window_size)/window_size, mode='same')
        else: 
            smoothed_labels = labels
    elif method == 'gaussian':
        # Apply Gaussian smoothing
        smoothed_labels = scipy.ndimage.gaussian_filter1d(labels, sigma=sigma)
    else:
        raise ValueError("Unsupported smoothing method")
    return smoothed_labels


def smooth_labels(label_dict, config={"method":"moving_average", "window_size":3, "sigma":2}):
    values_array = torch.stack(list(label_dict.values())).numpy()
    smoothed_labels = np.zeros_like(values_array)

    for i in range(len(values_array[0])):
        labellist = values_array[:, i]

        smoothed_labels[:,i] = smoothing(labellist, config["method"], config["window_size"], config["sigma"])

    smoothed_label_dict = smoothed_label_dict = {key: torch.tensor(smoothed_labels[i]) for i, key in enumerate(label_dict.keys())}

    return smoothed_label_dict


if __name__ == "__main__":

    # Example labels
    labels = torch.load("data/label_tensors.pt")
    # Convert the dictionary values to a 2D numpy array
    #values_array = np.array(list(labels.values()))

    # Extract the first column
    #label = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1])
    label = np.array([0.23,0.,0.264,0.244])
    smooth = smoothing(label, method = "moving_average", window_size=2, sigma=0.5)

    print("Original Labels:", label)
    print("Smoothed Labels:", smooth)

    # Smooth the labels using moving average
    original_labels = smooth_labels(labels, {"method":"moving_average", "window_size":0, "sigma":0})
    smoothed_labels = smooth_labels(labels, {"method":"gaussian", "window_size":2, "sigma":0.5})
    

    #print("Original Labels:", original_labels)
    #print("Smoothed Labels:", smoothed_labels)

    
    