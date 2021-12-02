import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from loader.loader_mvsec_flow import *
from loader.loader_dsec import *
from utils.logger import *
import utils.helper_functions as helper
import json
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visu
import argparse
from test import *
import git
import torch.nn
from model import eraft

def initialize_tester(config):
    # Warm Start
    if config['subtype'].lower() == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents

def get_visualizer(args):
    # DSEC dataset
    if args.dataset.lower() == 'dsec':
        return visualization.DsecFlowVisualizer
    # MVSEC dataset
    else:
        return visualization.FlowVisualizerEvents
        
def test(args):
    # Choose correct config file
    if args.dataset.lower()=='dsec':
        if args.type.lower()=='warm_start':
            config_path = 'config/dsec_warm_start.json'
        elif args.type.lower()=='standard':
            config_path = 'config/dsec_standard.json'
        else:
            raise Exception('Please provide a valid argument for --type. [warm_start/standard]')
    elif args.dataset.lower()=='mvsec':
        if args.frequency==20:
            config_path = 'config/mvsec_20.json'
        elif args.frequency==45:
            config_path = 'config/mvsec_45.json'
        else:
            raise Exception('Please provide a valid argument for --frequency. [20/45]')
        if args.type=='standard':
            raise NotImplementedError('Sorry, this is not implemented yet, please choose --type warm_start')
    else:
        raise Exception('Please provide a valid argument for --dataset. [dsec/mvsec]')


    # Load config file
    config = json.load(open(config_path))
    # Create Save Folder
    save_path = helper.create_save_path(config['save_dir'].lower(), config['name'].lower())
    print('Storing output in folder {}'.format(save_path))
    # Copy config file to save dir
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    # Logger
    logger = Logger(save_path)
    logger.initialize_file("test")

    # Instantiate Dataset
    # Case: DSEC Dataset
    additional_loader_returns = None
    if args.dataset.lower() == 'dsec':
        # Dsec Dataloading
        loader = DatasetProvider(
            dataset_path=Path(args.path),
            representation_type=RepresentationType.VOXEL,
            delta_t_ms=100,
            config=config,
            type=config['subtype'].lower(),
            visualize=args.visualize)
        loader.summary(logger)
        test_set = loader.get_test_dataset()
        additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}
    
    # Case: MVSEC Dataset
    else:
        if config['subtype'].lower() == 'standard':
            test_set = MvsecFlow(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
            )
        elif config['subtype'].lower() == 'warm_start':
            test_set = MvsecFlowRecurrent(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
            )
        else:
            raise NotImplementedError 
        test_set.summary(logger)

    # Instantiate Dataloader
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # Load Model
    model = eraft.ERAFT(
        config=config, 
        n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']
    )
    # Load Checkpoint
    checkpoint = torch.load(config['test']['checkpoint'])
    model.load_state_dict(checkpoint['model'])

    # Get Visualizer
    visualizer = get_visualizer(args)

    # Initialize Tester
    test = initialize_tester(config)

    test = test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=logger,
        save_path=save_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )

    test.summary()
    test._test()

if __name__ == '__main__':
    config_path = "config/config_test.json"
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Dataset path", required=True)
    parser.add_argument('-d', '--dataset', default="dsec", type=str, help="Which dataset to use: ([dsec]/mvsec)")
    parser.add_argument('-f', '--frequency', default=20, type=int, help="Evaluation frequency of MVSEC dataset ([20]/45) Hz")
    parser.add_argument('-t', '--type', default='warm_start', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('-v', '--visualize', action='store_true', help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='How many sub-processes to use for data loading')
    args = parser.parse_args()

    # Run Test Script
    test(args)
