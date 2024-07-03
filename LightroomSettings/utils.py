import torch
import numpy as np

def get_model_output(model, sample_img_tensor):
    model.eval()
    with torch.no_grad():
        output = model(sample_img_tensor).logits
    print(output)
    return output


def tensor_to_image(tensor):
    # Denormalize the tensor
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clip(0, 1)
    img_np = tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
    img_np = (img_np * 255).astype(np.uint8)  # Convert to 0-255 range
    return img_np


def get_mean_of_set(data_loader):

    average = []
    for img_tensors, img_labels in data_loader:
        img_labels.to("cpu")
        img_labels = img_labels.numpy()
        average.append(np.mean(img_labels, axis=0))

    average = np.mean(average, axis=0)

    return average


def get_std_of_set(data_loader):
    all_labels = []

    for img_tensors, img_labels in data_loader:
        img_labels = img_labels.cpu().numpy()
        all_labels.append(img_labels)

    # Combine all labels into one array
    all_labels_combined = np.concatenate(all_labels, axis=0)

    # Calculate the standard deviation over the entire dataset
    std_dev = np.std(all_labels_combined, axis=0)

    return std_dev