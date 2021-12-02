import numpy
import torch
from torch import nn
from torch.nn.functional import grid_sample
from scipy.spatial import transform
from scipy import interpolate
from matplotlib import pyplot as plt


def grid_sample_values(input, height, width):
    # ================================ Grid Sample Values ============================= #
    # Input:    Torch Tensor [3,H*W]m where the 3 Dimensions mean [x,y,z]               #
    # Height:   Image Height                                                            #
    # Width:    Image Width                                                             #
    # --------------------------------------------------------------------------------- #
    # Output:   tuple(value_ipl, valid_mask)                                            #
    #               value_ipl       -> [H,W]: Interpolated values                       #
    #               valid_mask      -> [H,W]: 1: Point is valid, 0: Point is invalid    #
    # ================================================================================= #
    device = input.device
    ceil = torch.stack([torch.ceil(input[0,:]), torch.ceil(input[1,:]), input[2,:]])
    floor = torch.stack([torch.floor(input[0,:]), torch.floor(input[1,:]), input[2,:]])
    z = input[2,:].clone()

    values_ipl = torch.zeros(height*width, device=device)
    weights_acc = torch.zeros(height*width, device=device)
    # Iterate over all ceil/floor points
    for x_vals in [floor[0], ceil[0]]:
        for y_vals in [floor[1], ceil[1]]:
            # Mask Points that are in the image
            in_bounds_mask = (x_vals < width) & (x_vals >=0) & (y_vals < height) & (y_vals >= 0)

            # Calculate weights, according to their real distance to the floored/ceiled value
            weights = (1 - (input[0]-x_vals).abs()) * (1 - (input[1]-y_vals).abs())

            # Put them into the right grid
            indices = (x_vals + width * y_vals).long()
            values_ipl.put_(indices[in_bounds_mask], (z * weights)[in_bounds_mask], accumulate=True)
            weights_acc.put_(indices[in_bounds_mask], weights[in_bounds_mask], accumulate=True)

    # Mask of valid pixels -> Everywhere where we have an interpolated value
    valid_mask = weights_acc.clone()
    valid_mask[valid_mask > 0] = 1
    valid_mask= valid_mask.bool().reshape([height,width])

    # Divide by weights to get interpolated values
    values_ipl = values_ipl / (weights_acc + 1e-15)
    values_rs = values_ipl.reshape([height,width])

    return values_rs.unsqueeze(0).clone(), valid_mask.unsqueeze(0).clone()

def forward_interpolate_pytorch(flow_in):
    # Same as the numpy implementation, but differentiable :)
    # Flow: [B,2,H,W]
    flow = flow_in.clone()
    if len(flow.shape) < 4:
        flow = flow.unsqueeze(0)

    b, _, h, w = flow.shape
    device = flow.device

    dx ,dy = flow[:,0], flow[:,1]
    y0, x0 = torch.meshgrid(torch.arange(0, h, 1), torch.arange(0, w, 1))
    x0 = torch.stack([x0]*b).to(device)
    y0 = torch.stack([y0]*b).to(device)

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.flatten(start_dim=1)
    y1 = y1.flatten(start_dim=1)
    dx = dx.flatten(start_dim=1)
    dy = dy.flatten(start_dim=1)

    # Interpolate Griddata...
    # Note that a Nearest Neighbor Interpolation would be better. But there does not exist a pytorch fcn yet.
    # See issue: https://github.com/pytorch/pytorch/issues/50339
    flow_new = torch.zeros(flow.shape, device=device)
    for i in range(b):
        flow_new[i,0] = grid_sample_values(torch.stack([x1[i],y1[i],dx[i]]), h, w)[0]
        flow_new[i,1] = grid_sample_values(torch.stack([x1[i],y1[i],dy[i]]), h, w)[0]

    return flow_new

class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size)%self.min_size
            self.pad_width = (self.min_size - width % self.min_size)%self.min_size
        else:
            pad_height = (self.min_size - height % self.min_size)%self.min_size
            pad_width = (self.min_size - width % self.min_size)%self.min_size
            if pad_height != self.pad_height or pad_width != self.pad_width:
                raise
        return nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]
