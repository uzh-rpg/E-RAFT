# =========================== Templates for saving Images ========================= #
GT_FLOW = '{}_{}_flow_gt.png'
FLOW_TEST = '{}_{}_flow.png'
IMG = '{}_{}_image.png'
EVENTS = '{}_{}_events.png'

# ========================= Templates for saving Checkpoints ====================== #
CHECKPOINT = '{:03d}_checkpoint.tar'

# =========================        MVSEC DATALOADING         ====================== #
MVSEC_DATASET_FOLDER = '{}_{}'
MVSEC_TIMESTAMPS_PATH_DEPTH = 'timestamps_depth.txt'
MVSEC_TIMESTAMPS_PATH_FLOW = 'timestamps_flow.txt'
MVSEC_TIMESTAMPS_PATH_IMAGES = 'timestamps_images.txt'
MVSEC_EVENTS_FILE = "davis/{}/events/{:06d}.h5"
MVSEC_FLOW_GT_FILE = "optical_flow/{:06d}.npy"
