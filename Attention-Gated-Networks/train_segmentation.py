import os
from torch.utils.data import DataLoader
from tqdm import tqdm



import numpy as np
from dataio.LiTS import LiTSDataset
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Split
    data_path = json_opts.data_path.LiTS
    print('data_path ', data_path)
    SEED = 0 
    np.random.seed(SEED)
    image_dir = os.listdir(os.path.join(data_path,'Training Batch 1')) + os.listdir(os.path.join(data_path,'Training Batch 2'))
    all_indexes = [ int(file_name[7:-4]) for file_name in image_dir if 'volume' in file_name]
    split = np.random.permutation(all_indexes)
    n_train, n_val, n_test = int(0.8 * len(split)), int(0.1 * len(split)), int(0.1 * len(split))
    
    train = split[: n_train]
    val = split[n_train : n_train+n_val]
    test = split[n_train + n_val :]

    # Setup Data Loader
    train_dataset = LiTSDataset(data_path, train, augment=True)
    val_dataset = LiTSDataset(data_path, val)
    test_dataset = LiTSDataset(data_path, test)
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=val_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()
    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                visuals = model.get_current_visuals()
                #visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        #for split in ['train', 'validation', 'test']:
            #visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            #visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
