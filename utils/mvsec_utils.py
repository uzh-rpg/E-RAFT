import numpy as np
# import cv2
from matplotlib import pyplot as plt
import os
from utils import filename_templates as TEMPLATES

def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    return

def estimate_corresponding_gt_flow(path_flow,
                                   gt_timestamps,
                                   start_time,
                                   end_time):
    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between
    # gt_iter and gt_iter+1.
    # gt_timestamps[gt_iter]    -> Timestamp just before start_time

    gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    # Load Flow just before start_time
    flow_file = os.path.join(path_flow, TEMPLATES.MVSEC_FLOW_GT_FILE.format(gt_iter))
    flow = np.load(flow_file)

    x_flow = flow[0]
    y_flow = flow[1]
    #x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    #y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt > dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt
    else:
        raise Exception
