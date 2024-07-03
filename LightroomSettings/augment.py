import torch
from torchvision import transforms
import random


def grid_mask(shape, r, d, delta_x, delta_y):
    """
    Creates a grid mask consisting of values zero and one for a given shape and parameters. A grid consits of small mask units, 
    where the parameter r in (0,1) denotes the ratio of the shorter visable edge in a unit ans the unit size d=random(d_{min},d_{max})
    is randomly chosen. Lastly the distances delta_x,delta_y in (0,d-1) between the first intact unit and the boundary of the image 
    are also chosen randomly.
    """
    mask = torch.ones(shape)
    ones_l = round(r*d)
    zeros_l = d-ones_l
    start_x, start_y = delta_x, delta_y

    while start_x<= shape[1]:
        end_x = min(start_x+zeros_l, shape[1])
        
        while  start_y<=shape[2]:
            end_y = min(start_y+zeros_l, shape[2])
            mask[:,start_x:end_x, start_y:end_y] = 0
            start_y = end_y + ones_l
        start_x = end_x + ones_l
        start_y = delta_y    

    return mask


def gridmask_deletion(img, r, d_min, d_max):
    """
    GridMask Data Augmentation as introduced in [1]. Ereasing-based data augmentation method that uses structured dropping regions
    to avoid deleting too excesive deleting of important structures. It is shown to perform better for training of CNNs on Computer Vison tasks compared to
    the Cutout-method or the HideandSeek-method.
    We form the grid by building units consiting of 4 squares (3 with value 1 and one with value 0 which is on the top left).
    The unit length is randomly chosen between d_min and d_max. The distances between the first intact unit on the top left and 
    the boundary of the image are denoted by delta_x in (0,d-1) and delta_y in (0,d-1).

    ---------
    [1] Pengguang Chen, Shu Liu, Hengshuang
        Zhao, and Jiaya Jia. Gridmask data augmentation. arXiv
        preprint arXiv:2001.04086, 2020.
    ---------
    

    Parameters:

        img (tensor of shape (3,H,W) or (1,H,W)) - base image
        r (float between 0 and 1) - ratio between the edges of keeping and deletion region
        d_min (int) - minimum unit length
        d_max (int) - maximum unit length

    Output:
        Tensor of the modified image
    """
    d = random.randint(d_min, d_max)
    delta_x = random.randint(0,d-1)
    delta_y = random.randint(0,d-1)

    M = grid_mask(img.size(), r, d, delta_x, delta_y )
    masked_img = torch.mul(img, M)
    
    return masked_img


def local_rotation(img):
    """
    Implementation of the local rotation method, which divides the image in 4 equal parts to then randomly rotate
    each of them and glue them back together. In [1] it is shown that training a CNN withthis method increases the test 
    accurancy compared to a model trained with pure global rotation. The aim of this model is takle the local bias of CNNs.
    --------
    [1] Youmin Kim, AFM Shahab Uddin, and Sung-Ho Bae. Local augment:
    Utilizing local bias property of convolutional neural networks for data
    augmentation. IEEE Access, 9:15191–15199, 2021.
    --------

    Parameters:
        img (tensor of shape (C,W,H)) - base image

    Output:
        Tensor of the modified image
    """

    h, w = img.size()[1], img.size()[2]

    mid_h = h//2
    mid_w = w//2

    #Divide image in 4 patches

    patch_1 = img[:, :mid_h, :mid_w]
    patch_2 = img[:, :mid_h, mid_w:]
    patch_3 = img[:, mid_h:, :mid_w]
    patch_4 = img[:, mid_h:, mid_w:]

    #Randomly rotate each patch

    rotated_patches = []

    for patch in [patch_1, patch_2, patch_3, patch_4]:
        r = random.randint(0,3)
        patch = transforms.functional.rotate(patch, 90.0*r)
        rotated_patches.append(patch)

    #Glue the Patches toghether

    top_half = torch.cat((rotated_patches[0],rotated_patches[1]), dim=2)
    bottom_half = torch.cat((rotated_patches[2],rotated_patches[3]), dim=2)

    rot_img = torch.cat((top_half,bottom_half),dim=1)
    
    return rot_img