import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import json


import numpy as np
from dataio.LiTS import LiTSDataset
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from torch.utils.tensorboard import SummaryWriter

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
    train_dataset = LiTSDataset(data_path, train, augment=True, no_tumor=True)
    val_dataset = LiTSDataset(data_path, val, no_tumor=True)
    test_dataset = LiTSDataset(data_path, test, no_tumor=True)
    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=val_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=8, batch_size=train_opts.batchSize, shuffle=False)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()
    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()
    
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    
    time = datetime.datetime.today()
    log_id = '{}_{}h{}min'.format(time.date(), time.hour, time.minute)
    log_path = os.path.join(args.logdir, log_id)
    i = 0
    created = False
    while not created:
        try:
            os.mkdir(log_path)
            created = True
        except:
            i += 1
            log_id ='{}_{}h{}min_{}'.format(time.date(), time.hour, time.minute, i)
            log_path = os.path.join(args.logdir,log_id)
            
    with open(os.path.join(log_path,'config.json'), 'w') as file:
        json.dump(json_opts, file)
    
    writer = SummaryWriter(log_path)

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        avg_errors = {}
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')
            
            for loss in errors.keys():
                writer.add_scalar('train_batch_errors_{}'.format(loss), errors[loss], epoch)

            for loss in errors.keys():
                if (loss not in avg_errors.keys()) or (avg_errors[loss] == 0): 
                    avg_errors[loss] = errors[loss]
                avg_errors[loss] = 0.9 * avg_errors[loss] + 0.1 * errors[loss]
                
        for loss in avg_errors.keys():
            writer.add_scalar('train_epoch_errors_{}'.format(loss), avg_errors[loss], epoch)
        print('Training_{} Average Errors : '.format(epoch), avg_errors)

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            avg_errors = {}
            avg_stats = {}
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)
                
                for loss in errors.keys():
                    writer.add_scalar('{}_batch_errors_{}'.format(split, loss), errors[loss], epoch)
                for seg_stat in stats.keys():
                    writer.add_scalar('{}_batch_stats_{}'.format(split, seg_stat), stats[seg_stat], epoch)
                
                for loss in errors.keys():
                    if (loss not in avg_errors.keys()) or (avg_errors[loss] == 0): 
                        avg_errors[loss] = errors[loss]
                    avg_errors[loss] = 0.9 * avg_errors[loss] + 0.1 * errors[loss]
                
                for seg_stat in stats.keys():
                    if (seg_stat not in avg_stats.keys()) or (avg_stats[seg_stat] == 0): 
                        avg_stats[seg_stat] = stats[seg_stat]
                    avg_stats[seg_stat] = 0.9 * avg_stats[seg_stat] + 0.1 * stats[seg_stat]

                # Visualise predictions
                #visuals = model.get_current_visuals()
                #visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
            for loss in avg_errors.keys():
                writer.add_scalar('{}_epoch_errors_{}'.format(split, loss), avg_errors[loss], epoch)
            for seg_stat in avg_stats.keys():
                writer.add_scalar('{}_epoch_stats_{}'.format(split, seg_stat), avg_stats[seg_stat], epoch)
            
            print('{}_{} Average Errors : '.format(split, epoch), avg_errors)
            print('{}_{} Average Stats : '.format(split, epoch), avg_stats)
            print('-'*150)

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

    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('--logdir', required=True, 
                        type=str,help = 'path to the directory containing all run folders')
    args = parser.parse_args()

    train(args)
