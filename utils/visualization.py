import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy
import os
import loader.utils as loader
import utils.transformers as transformers
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
from matplotlib.lines import Line2D
from skimage.transform import rotate, warp
from skimage import io
import cv2
import imageio
from torchvision.transforms import CenterCrop

class BaseVisualizer(object):
    def __init__(self, dataloader, save_path, additional_args=None):
        super(BaseVisualizer, self).__init__()
        self.dataloader = dataloader
        self.visu_path = helper.create_save_path(save_path, 'visualizations')
        self.submission_path = os.path.join(save_path, 'submission')
        os.mkdir(self.submission_path)
        self.mode = 'test'
        self.additional_args = additional_args

    def __call__(self, batch, epoch=None):
        for j in batch['loader_idx'].cpu().numpy().astype(int):
            # Get Batch Index
            batch_idx = torch.nonzero(batch['loader_idx'] == j).item()

            # Visualize Ground Truths, but only in first epoch or if we're cropping
            if epoch == 1 or epoch is None or 'crop_window' in batch.keys():
                self.visualize_ground_truths(batch, batch_idx=batch_idx, epoch=epoch,
                                                data_aug='crop_window' in batch.keys())

            # Visualize Estimations
            self.visualize_estimations(batch, batch_idx=batch_idx, epoch=epoch)

    def visualize_ground_truths(self, batch, batch_idx, epoch=None, data_aug=False):
        raise NotImplementedError

    def visualize_estimations(self, batch, batch_idx, epoch=None):
        raise NotImplementedError

    def visualize_image(self, image, true_idx, epoch=None, data_aug=False):
        true_idx = int(true_idx)
        name = TEMPLATES.IMG.format('inference', true_idx)
        save_image(os.path.join(self.visu_path, name), image.detach().cpu())

    def visualize_events(self, image, batch, batch_idx, epoch=None, flip_before_crop=True, crop_window=None):
        raise NotImplementedError

    def visualize_flow_colours(self, flow, true_idx, epoch=None, data_aug=False, is_gt=False, fix_scaling=10,
                               custom_name=None, prefix=None, suffix=None, sub_folder=None):
        true_idx = int(true_idx)

        if custom_name is None:
            name = TEMPLATES.FLOW_TEST.format('inference', true_idx)
        else:
            name = custom_name
        if prefix is not None:
            name = prefix + name
        if suffix is not None:
            split = name.split('.')
            name =  split[0] + suffix + "." +split[1]
        if sub_folder is not None:
            name = os.path.join(sub_folder, name)
        # Visualize
        _, scaling = visualize_optical_flow(flow.detach().cpu().numpy(),
                                            os.path.join(self.visu_path, name),
                                            scaling=fix_scaling)
        return scaling

    def visualize_flow_submission(self, seq_name: str, flow: numpy.ndarray, file_index: int):
        # flow_u(u,v) = ((float)I(u,v,1)-2^15)/128.0;
        # flow_v(u,v) = ((float)I(u,v,2)-2^15)/128.0;
        # valid(u,v)  = (bool)I(u,v,3);
        # [-2**15/128, 2**15/128] = [-256, 256]
        #flow_map_16bit = np.rint(flow_map*128 + 2**15).astype(np.uint16)
        _, h,w = flow.shape
        flow_map = numpy.rint(flow*128 + 2**15)
        flow_map = flow_map.astype(numpy.uint16).transpose(1,2,0)
        flow_map = numpy.concatenate((flow_map, numpy.zeros((h,w,1), dtype=numpy.uint16)), axis=-1)
        parent_path = os.path.join(
            self.submission_path,
            seq_name
        )
        if not os.path.exists(parent_path):
            os.mkdir(parent_path)
        file_name = '{:06d}.png'.format(file_index)

        imageio.imwrite(os.path.join(parent_path, file_name), flow_map, format='PNG-FI')

class FlowVisualizerEvents(BaseVisualizer):
    def __init__(self, dataloader, save_path, clamp_flow=True, additional_args=None):
        super(FlowVisualizerEvents, self).__init__(dataloader, save_path, additional_args=additional_args)
        self.flow_scaling = 0
        self.clamp_flow = clamp_flow

    def visualize_events(self, image, batch, batch_idx, epoch=None, flip_before_crop=True, crop_window=None):
        # Plots Events on top of an Image.
        if image is not None:
            im = image.detach().cpu()
        else:
            im = None

        # Load Raw events
        events = self.dataloader.dataset.get_events(loader_idx=int(batch['loader_idx'][batch_idx].item()))
        name_events = TEMPLATES.EVENTS.format('inference', int(batch['idx'][batch_idx].item()))

        # Event Sequence to Event Image
        events = events_to_event_image(events,
                                       int(batch['param_evc']['height'][batch_idx].item()),
                                       int(batch['param_evc']['width'][batch_idx].item()),
                                       im,
                                       crop_window=crop_window,
                                       rotation_angle=False,
                                       horizontal_flip=False,
                                       flip_before_crop=False)
        # center-crop 256x256
        crop = CenterCrop(256)
        events = crop(events)
        # Save
        save_image(os.path.join(self.visu_path, name_events), events)

    def visualize_ground_truths(self, batch, batch_idx, epoch=None, data_aug=False):
        # Visualize Events
        if 'image_old' in batch.keys():
            image_old = batch['image_old'][batch_idx]
        else:
            image_old = None
        self.visualize_events(image_old, batch, batch_idx, epoch)

        # Visualize Image
        '''
        if 'image_old' in batch.keys():
            self.visualize_image(batch['image_old'][batch_idx], batch['idx'][batch_idx],epoch, data_aug)
        '''
        # Visualize Flow GT
        flow_gt = batch['flow'][batch_idx].clone()
        flow_gt[~batch['gt_valid_mask'][batch_idx].bool()] = 0.0
        self.flow_scaling = self.visualize_flow_colours(flow_gt, batch['idx'][batch_idx], epoch=epoch,
                                                        data_aug=data_aug, is_gt=True, fix_scaling=None, suffix='_gt')

    def visualize_estimations(self, batch, batch_idx, epoch=None):
        # Visualize Flow Estimation
        if self.clamp_flow:
            scaling = self.flow_scaling[1]
        else:
            scaling = None
        self.visualize_flow_colours(batch['flow_est'][batch_idx], batch['idx'][batch_idx], epoch=epoch,
                                    is_gt=False, fix_scaling=scaling)

        # Visualize Masked Flow
        flow_est = batch['flow_est'][batch_idx].clone()
        flow_est[~batch['gt_valid_mask'][batch_idx].bool()] = 0.0
        self.visualize_flow_colours(flow_est, batch['idx'][batch_idx], epoch=epoch,
                                    is_gt=False, fix_scaling=scaling, suffix='_masked')

class DsecFlowVisualizer(BaseVisualizer):
    def __init__(self, dataloader, save_path, additional_args=None):
        super(DsecFlowVisualizer, self).__init__(dataloader, save_path, additional_args=additional_args)
        # Create Visu folders for every sequence
        for name in self.additional_args['name_mapping']:
            os.mkdir(os.path.join(self.visu_path, name))
            os.mkdir(os.path.join(self.submission_path, name))

    def visualize_events(self, image, batch, batch_idx, sequence_name):
        sequence_idx = [i for i, e in enumerate(self.additional_args['name_mapping']) if e == sequence_name][0]
        delta_t_us = self.dataloader.dataset.datasets[sequence_idx].delta_t_us
        loader_instance = self.dataloader.dataset.datasets[sequence_idx]
        h, w = loader_instance.get_image_width_height()
        events = loader_instance.event_slicer.get_events(
            t_start_us=batch['timestamp'][batch_idx].item(),
            t_end_us=batch['timestamp'][batch_idx].item()+delta_t_us
        )
        p = events['p'].astype(numpy.int8)
        t = events['t'].astype(numpy.float64)
        x = events['x']
        y = events['y']
        p = 2*p - 1
        xy_rect = loader_instance.rectify_events(x, y)
        x_rect = numpy.rint(xy_rect[:, 0])
        y_rect = numpy.rint(xy_rect[:, 1])

        events_rectified = numpy.stack([t, x_rect, y_rect, p], axis=-1)
        event_image = events_to_event_image(
            event_sequence=events_rectified,
            height=h,
            width=w
        ).numpy()
        name_events = TEMPLATES.EVENTS.format('inference', int(batch['file_index'][batch_idx].item()))
        out_path = os.path.join(self.visu_path, sequence_name, name_events)
        imageio.imsave(out_path, event_image.transpose(1,2,0))

    def __call__(self, batch, batch_idx, epoch=None):
        for batch_idx in range(len(batch['file_index'])):
            if batch['save_submission'][batch_idx]:
                sequence_name = self.additional_args['name_mapping'][int(batch['name_map'][batch_idx].item())]
                # Save for Benchmark Submission
                self.visualize_flow_submission(
                    seq_name=sequence_name,
                    flow=batch['flow_est'][batch_idx].clone().cpu().numpy(),
                    file_index=int(batch['file_index'][batch_idx].item()),
                )
            if batch['visualize'][batch_idx]:
                sequence_name = self.additional_args['name_mapping'][int(batch['name_map'][batch_idx].item())]
                # Visualize Flow
                self.visualize_flow_colours(
                    batch['flow_est'][batch_idx],
                    batch['file_index'][batch_idx],
                    epoch=epoch,
                    is_gt=False,
                    fix_scaling=None,
                    sub_folder=sequence_name
                )
                # Visualize Events
                self.visualize_events(
                    image=None,
                    batch=batch,
                    batch_idx=batch_idx,
                    sequence_name=sequence_name
                )

def save_tensor(filepath, tensor):
    map = plt.get_cmap('plasma')
    t = tensor[0].numpy() / tensor[0].numpy().max()
    image = map(t) * 255
    io.imsave(filepath, image.astype(numpy.uint8))


def grayscale_to_rgb(tensor, permute=False):
    # Tensor [height, width, 3], or
    # Tensor [height, width, 1], or
    # Tensor [1, height, width], or
    # Tensor [3, height, width]

    # if permute -> Convert to [height, width, 3]
    if permute:
        if tensor.size()[0] < 4:
            tensor = tensor.permute(1, 2, 0)
        if tensor.size()[2] == 1:
            return torch.stack([tensor[:, :, 0]] * 3, dim=2)
        else:
            return tensor
    else:
        if tensor.size()[0] == 1:
            return torch.stack([tensor[0, :, :]] * 3, dim=0)
        else:
            return tensor


def save_image(filepath, tensor):
    # Tensor [height, width, 3], or
    # Tensor [height, width, 1], or
    # Tensor [1, height, width], or
    # Tensor [3, height, width]

    # Convert to [height, width, 3]
    tensor = grayscale_to_rgb(tensor, True).numpy()
    use_pyplot=False
    if use_pyplot:
        fig = plt.figure()
        # Change Dimensions of Tensor
        plot = plt.imshow(tensor.astype(numpy.uint8))
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        fig.savefig(filepath, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        io.imsave(filepath, tensor.astype(numpy.uint8))


def events_to_event_image(event_sequence, height, width, background=None, rotation_angle=None, crop_window=None,
                          horizontal_flip=False, flip_before_crop=True):
    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 1].astype(numpy.int)
    y_negative = event_sequence[~polarity, 2].astype(numpy.int)
    x_positive = event_sequence[polarity, 1].astype(numpy.int)
    y_positive = event_sequence[polarity, 2].astype(numpy.int)

    positive_histogram, _, _ = numpy.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = numpy.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = numpy.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = numpy.transpose(positive_histogram > negative_histogram)
    # Normally, we flip first, before we apply the other data augmentations
    if flip_before_crop:
        if horizontal_flip:
            red = numpy.flip(red, axis=1)
            blue = numpy.flip(blue, axis=1)
        # Rotate, if necessary
        if rotation_angle is not None:
            red = rotate(red, angle=rotation_angle, preserve_range=True).astype(bool)
            blue = rotate(blue, angle=rotation_angle, preserve_range=True).astype(bool)
        # Crop, if necessary
        if crop_window is not None:
            tf = transformers.RandomCropping(crop_height=crop_window['crop_height'],
                                             crop_width=crop_window['crop_width'],
                                             left_right=crop_window['left_right'],
                                             shift=crop_window['shift'])
            red = tf.crop_image(red, None, window=crop_window)
            blue = tf.crop_image(blue, None, window=crop_window)
    else:
        # Rotate, if necessary
        if rotation_angle is not None:
            red = rotate(red, angle=rotation_angle, preserve_range=True).astype(bool)
            blue = rotate(blue, angle=rotation_angle, preserve_range=True).astype(bool)
        # Crop, if necessary
        if crop_window is not None:
            tf = transformers.RandomCropping(crop_height=crop_window['crop_height'],
                                             crop_width=crop_window['crop_width'],
                                             left_right=crop_window['left_right'],
                                             shift=crop_window['shift'])
            red = tf.crop_image(red, None, window=crop_window)
            blue = tf.crop_image(blue, None, window=crop_window)
        if horizontal_flip:
            red = numpy.flip(red, axis=1)
            blue = numpy.flip(blue, axis=1)

    if background is None:
        height, width = red.shape
        background = torch.full((3, height, width), 255).byte()
    if len(background.shape) == 2:
        background = background.unsqueeze(0)
    else:
        if min(background.size()) == 1:
            background = grayscale_to_rgb(background)
        else:
            if not isinstance(background, torch.Tensor):
                background = torch.from_numpy(background)
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(numpy.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(numpy.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(numpy.linspace(minval, maxval, n)))
    return new_cmap


def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points


def visualize_optical_flow(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    flow = flow.transpose(1,2,0)
    flow[numpy.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = numpy.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = numpy.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = numpy.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=numpy.pi*2
    hsv[..., 0] = ang/numpy.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = numpy.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
    plot_with_pyplot = False
    if plot_with_pyplot:
        fig = plt.figure(frameon=False)
        plot = plt.imshow(bgr)
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
    if text is not None:
        plt.text(0, -5, text)

    if savepath is not None:
        if plot_with_pyplot:
            fig.savefig(savepath, bbox_inches='tight', dpi=200)
            plt.close()
        else: #Plot with skimage
            out = bgr*255
            io.imsave(savepath, out.astype('uint8'))
    return bgr, (mag.min(), mag.max())
