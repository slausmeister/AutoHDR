import re
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import rawpy
import torch
from torchvision import transforms
from torchvision.transforms import v2

import time

from LightroomSettings.helpers import _ema
from LightroomSettings.augment import gridmask_deletion, local_rotation
from LightroomSettings.label_smoothing import smooth_labels


class RawImageDataset(torch.utils.data.Dataset):
    """
    This class forms the foundation of ImageDatasets. For a directory of RAWs and their corresponding XMPs RawImageDataset provides
    the possibility to access them as preprocessed PyTroch-tensors. Note that for every call of __getitem__ a RAW is loaded and preprocessed
    which is computational expensive.
    """

    def __init__(self, directory_path):
        """
        For each data only its path is stored and the actual data is only loaded when it is needed.
        """
        self.directory_path = directory_path
        self.filenames = self.get_unique_filenames(directory_path)

     
    def preprocess_image(self, rgb_array):
        """
        Sample image down to tensor and normalizes it.
        """
        preprocess = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(rgb_array)

        return img_tensor

    
    def get_unique_filenames(self, directory_path):
        unique_filenames = set()
        pattern = re.compile(r"DSC\d{5}\.ARW")
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if pattern.match(file):
                    filename_without_extension = os.path.splitext(file)[0]
                    unique_filenames.add(filename_without_extension)
                    
        return list(unique_filenames)

    
    def __len__(self):
        return len(self.filenames)
        
    
    def remove_margin(self, image_array, margin=12):
        return image_array[margin:-margin, margin:-margin,:]

    
    def __getitem__(self, idx):
        """
        Given the stored path an image and its label is loaded by applying the framwork of the class. Loading an image is expensive.
        """
        file = self.filenames[idx]

        arw_path = os.path.join(self.directory_path, file + ".ARW")
        xmp_path = os.path.join(self.directory_path, file + ".xmp")

        if not os.path.exists(arw_path):
            raise FileNotFoundError(f"Raw file {arw_path} does not exist")

        if not os.path.exists(xmp_path):
            raise FileNotFoundError(f"XMP file {xmp_path} does not exist")

        with rawpy.imread(arw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
            rgb = self.remove_margin(rgb)    

        img_tensor = self.preprocess_image(rgb)
        img_label = torch.tensor(self.get_label(xmp_path))

        return img_tensor, img_label


    # Import Lightroom settings
    @staticmethod
    def get_label(xmp_file_path):
        
        with open(xmp_file_path, 'r') as file:
            xmp_data = file.read()

        # Parse the XMP data
        root = ET.fromstring(xmp_data)

        # Define the namespace map
        ns = {
            'x': 'adobe:ns:meta/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/',
            'tiff': 'http://ns.adobe.com/tiff/1.0/',
            'exif': 'http://ns.adobe.com/exif/1.0/',
            'aux': 'http://ns.adobe.com/exif/1.0/aux/',
            'exifEX': 'http://cipa.jp/exif/1.0/',
            'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
            'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
            'stEvt': 'http://ns.adobe.com/xap/1.0/sType/ResourceEvent#',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'crd': 'http://ns.adobe.com/camera-raw-defaults/1.0/',
            'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/'
        }

        # Extract and cast the required values to floats
        values = [
            5 ** (-1) *float(root.find('.//rdf:Description[@crs:Exposure2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Contrast2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Contrast2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Highlights2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Highlights2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Shadows2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Shadows2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Whites2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Whites2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Blacks2012]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Blacks2012']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Vibrance]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Vibrance']),
            100 ** (-1) *float(root.find('.//rdf:Description[@crs:Saturation]', ns).attrib['{http://ns.adobe.com/camera-raw-settings/1.0/}Saturation'])
        ]

        return values    


class ImageDataset(torch.utils.data.Dataset):
    """
    This class creates a Basic Dataset. For this, it uses the technical groundwork from RawImageDataset and loads each of the images 
    and labels to save them once as a torch.Tensor. Through this class, the tensors can always be accessed directly without having to 
    load the images first. This should significantly speed up the iterative loading process, as everything only needs to be loaded once 
    at the beginning.
    """
    def __init__(self, raw_dataset, reload_data=False):

        self.directory_path = raw_dataset.directory_path
        self.filenames = raw_dataset.filenames 
        tensor_file_path = os.path.join(self.directory_path, "img_tensors.pt")
        label_file_path = os.path.join(self.directory_path, "label_tensors.pt")
        
        if os.path.exists(tensor_file_path) and os.path.exists(label_file_path) and not reload_data:
            #if the data is already loaded skip the loading process
            print("Loading tensors from file")
            self.img = torch.load(tensor_file_path)
            self.label = torch.load(label_file_path)
        else:
            print("Creating tensors from raw data")
            img_tensors = {}
            label_tensors = {}
            time_array = []

            for i in range(len(raw_dataset)):
                t0 = time.time()

                #access the preprocessed data via RawImageDataset.__getitem__
                name = raw_dataset.filenames[i]
                img_tensor, label_tensor = raw_dataset[i]
                img_tensors[name] = img_tensor
                label_tensors[name] = label_tensor

                t1 = time.time()
                time_array.append(t1 - t0)

                #loading bar
                per = int((i / len(raw_dataset)) * 100)
                bar_length = 20
                filled_length = int(bar_length * i // len(raw_dataset))
                bar = '=' * filled_length + '-' * (bar_length - filled_length)

                sys.stdout.write(f'\r[{bar}] {per}%  Time left: {_ema(time_array)*(len(raw_dataset)-i):.2f}s      ')
                sys.stdout.flush()

            print("\n")

            img_path = os.path.join(self.directory_path, "img_tensors.pt")
            label_path = os.path.join(self.directory_path, "label_tensors.pt")
            
            torch.save(img_tensors, img_path)
            torch.save(label_tensors, label_path)

            self.img = img_tensors  # Store the actual tensor data
            self.label = label_tensors  # Store the actual tensor data

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        img = self.img[file]
        label = self.label[file]
        return img, label


class AugmentedDataset(torch.utils.data.Dataset):
    """
    In this class, all the work on the dataset should happen where the source image remains the same - particularly during 
    the processes of data augmentation and label smoothing. The idea is that with this workflow division, the base images need 
    to be loaded only once (when the base set is initially created), as this is likely the biggest bottleneck, and afterwards, 
    work is only done at the tensor level. However, the construction principle is maintained so that the torch DataLoader can 
    be applied to this structure. For this reason, a separate dataset is created for each augmentation operation.
    """
    def __init__(self, dataset, flip=False, flip_direction=None, crop=False, crop_per = 0.95, distortion=False, distortion_scale=0.2, elastic_transform=False, magnitude=50.0, gauss = False, sigma=(0.1,2.0), grid=False, grid_param = {"ratio":0.2,"d_min":2,"d_max":3}, local_rot = False, auto_augment=False, rand_augment = False, aug_mix= False, smooth_config={'method':None}):
        
        if isinstance(dataset, torch.utils.data.Subset):
            self.img = dataset.dataset.img
            self.label = dataset.dataset.label
        else:
            self.img = dataset.img
            self.label = dataset.label

        #initialize all augmention and smoothing parameters    
        self.flip = flip
        self.flip_direction = flip_direction
        self.crop=crop
        self.crop_per = crop_per
        self.dist = distortion
        self.dist_scale = distortion_scale
        self.elast = elastic_transform
        self.magnitude = magnitude
        self.gauss = gauss
        self.sigma = sigma
        self.grid = grid
        self.grid_param = grid_param
        self.locrot = local_rot
        self.auto_aug = auto_augment
        self.rand_aug = rand_augment
        self.aug_mix = aug_mix
        

        if isinstance(dataset, torch.utils.data.Subset):
            self.filenames = [dataset.dataset.filenames[i] for i in dataset.indices]
        else:
            self.filenames = dataset.filenames

        #prepare smoothing methods and set default values
        self.smoothing_config = smooth_config
        self.smoothing = False

        if self.smoothing_config['method'] == None:
            self.smoothing = False
        elif self.smoothing_config['method'] == "moving_average" or self.smoothing_config['method'] == "gaussian":
            self.smoothing = True
            
            if "window_size" not in self.smoothing_config.keys(): 
                self.smoothing_config['window_size'] = 3
                if self.smoothing_config['method'] == "average_smoothing":
                    raise Warning("No window size seleceted for moving average smoothing. Selecting '3' instead!")
            if "sigma" not in self.smoothing_config.keys(): 
                self.smoothing_config['sigma'] = 1
                if self.smoothing_config['method'] == "gaussian":
                    raise Warning("No sigma seleceted for gaussian smoothing. Selecting '1' instead!")
        else:
            raise ValueError(f"Unknown smoothing method: {smooth_config['method']}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        #get basic data from ImageDatatset
        file = self.filenames[idx]
        img = self.img[file]
        labels = self.label
        label = labels[file]

        ###Data Augmentation
        
        #Flipping and Rotating
        if self.flip:
            if self.flip_direction == "horizontal":
                img = transforms.functional.hflip(img) 
            elif self.flip_direction == "rot90":
                img = transforms.functional.rotate(img, 90.0)
            elif self.flip_direction == "rot180":
                img = transforms.functional.rotate(img, 180.0)
            elif self.flip_direction == "rot270":
                img = transforms.functional.rotate(img, 270.0)
            elif self.flip_direction == "rot90h":
                img = transforms.functional.rotate(transforms.functional.hflip(img), 90.0)
            elif self.flip_direction == "rot180h":
                img = transforms.functional.rotate(transforms.functional.hflip(img), 180.0)
            elif self.flip_direction == "rot270h":
                img = transforms.functional.rotate(transforms.functional.hflip(img), 270.0)

        #Cropping
        elif self.crop == True:
            size = (img.shape[1],img.shape[2])
            img = transforms.RandomResizedCrop(size, scale=(self.crop_per,1.0))(img)

        #Random Perspective
        elif self.dist == True:
            img = transforms.RandomPerspective(distortion_scale=self.dist_scale, p=1.0)(img)

        #Blurring/Noising
        elif self.elast ==True:
            img = transforms.ElasticTransform(alpha=self.magnitude)(img)
        elif self.gauss == True:
            img = transforms.GaussianBlur(kernel_size=(9,9), sigma=self.sigma)(img)

        #GridMask deletion
        elif self.grid == True:
            img = gridmask_deletion(img, r=self.grid_param["ratio"], d_min=self.grid_param["d_min"], d_max=self.grid_param["d_max"])

        #local rotation
        elif self.locrot == True:
            img = local_rotation(img)

        #label-smoothing
        if self.smoothing:
            labels= smooth_labels(labels, self.smoothing_config)
            label = labels[file]

        return img, label


def load_data(directory_path, data_params):
    """
    Application of the Dataset architecture. For a given path where RAWs and XMPs are stored and sufficient parameters a dataloader for
    training and validation each is initialized that can be used for training in the usual PyTroch framework.
    ----------------------
    Parameters:
        directory_path (str): path to the directory where RAWs and XMPs are stored. Corresponding RAWs and XMPs has to be named the same way.
        data_params (dict): all parameters considering data preprocessing and loading including augmention and smoothing.

    Return:
        data_loader (torch.utils.data.DataLoader): prepared training data
        val_oader (torch.utils.data.DataLoader): prepared validation data
    """

    #initialize all parameters
    batch_size = data_params['batch_size']
    shuffle = data_params['shuffle_dataset']
    flip_dirs = data_params['flip_directions']
    num_workers = data_params['num_of_dataloader_workers']
    reload_data = data_params['force_preprocessing']
    rand_crop = data_params["random_cropping"]
    crop_scale = data_params["cropping_scale"]
    rand_per = data_params["perspective_transform"]
    distortion_scale = data_params["distortion_scale"]
    elastic_transform = data_params["elastic_transform"]
    gauss = data_params["gauss"]
    sigma = data_params["sigma"]
    grid_mask = data_params["grid_mask"]
    grid_param = data_params["grid_param"]
    local_rot = data_params["local_rotation"]
    validation_split = data_params["validation_split"]  
    smooth_configuration_list = data_params["smooth_config"]
    
    #initialize Datasets and split
    raw_data = RawImageDataset(directory_path)
    tensor_data = ImageDataset(raw_data, reload_data=reload_data)
    
    base_data, val_data = torch.utils.data.random_split(tensor_data, validation_split)

    datalist = []
    #prepare smoothing configurations
    if smooth_configuration_list == [] or smooth_configuration_list == None:
        sml = [{"method":None}]
    else:
        sml = smooth_configuration_list

    #for each augmention/smoothing option an augmented dataset is created
    for smooth_configuration in sml:
        datalist.append(AugmentedDataset(base_data, smooth_config=smooth_configuration))

        for dir in flip_dirs:
            flipped_dataset = AugmentedDataset(base_data, flip=True, flip_direction=dir, smooth_config=smooth_configuration)
            datalist.append(flipped_dataset)

        for i in range(rand_crop):
            cropped_dataset = AugmentedDataset(base_data, crop=True, crop_per=crop_scale, smooth_config=smooth_configuration)
            datalist.append(cropped_dataset)

        for j in range(rand_per):
            distorted_dataset = AugmentedDataset(base_data, distortion = True, distortion_scale = distortion_scale, smooth_config=smooth_configuration)
            datalist.append(distorted_dataset)

        for magnitude in elastic_transform:
            elastic_dataset = AugmentedDataset(base_data, elastic_transform=True, magnitude = magnitude, smooth_config=smooth_configuration)
            datalist.append(elastic_dataset)

        for k in range(gauss):
            gauss_dataset = AugmentedDataset(base_data, gauss = True, sigma = sigma, smooth_config=smooth_configuration)
            datalist.append(gauss_dataset)

        for _ in range(grid_mask):
            grid_dataset = AugmentedDataset(base_data, grid = True, grid_param = grid_param, smooth_config=smooth_configuration)
            datalist.append(grid_dataset)

        for _ in range(local_rot):
            loc_rot_dataset = AugmentedDataset(base_data, local_rot=True, smooth_config=smooth_configuration)
            datalist.append(loc_rot_dataset)

    # Concatenate the original dataset with the augmented datasets
    augmented_dataset = torch.utils.data.ConcatDataset(datalist)
    
    # Init dataloaders
    data_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader, val_loader