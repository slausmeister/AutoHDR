import numpy as np
import scipy.ndimage
import torch


def smoothing(labels, method='moving_average', window_size=5, sigma=2):
    """
    Takes a list of labels and gives back a list of smoothed labeles depending on the chosen method.
    window_size is parameter for moving average. sigma is a paramter for gaussian smoothing
    """
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
    """
    Takes a dict of labels as produced by DataSet and gives back the corresponding dict with smoothed label values.
    """
    values_array = torch.stack(list(label_dict.values())).numpy()
    smoothed_labels = np.zeros_like(values_array)

    for i in range(len(values_array[0])):
        labellist = values_array[:, i]

        smoothed_labels[:,i] = smoothing(labellist, config["method"], config["window_size"], config["sigma"])

    smoothed_label_dict = smoothed_label_dict = {key: torch.tensor(smoothed_labels[i]) for i, key in enumerate(label_dict.keys())}

    return smoothed_label_dict

    
    