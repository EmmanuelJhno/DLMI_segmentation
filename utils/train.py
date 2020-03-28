import os
import sys
from copy import deepcopy
import json  
import argparse
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data_loader.LiTS import LiTSDataset
from models import get_model
# from utils.utilities import ObjFromDict

class ObjFromDict:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ObjFromDict(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, ObjFromDict(b) if isinstance(b, dict) else b)
                
                
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous()
    tflat = target.contiguous()
    intersection = (iflat * tflat).sum(dim=(2,3,4))

    A_sum = torch.sum(tflat * iflat, dim=(2,3,4))
    B_sum = torch.sum(tflat * tflat , dim=(2,3,4))
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )                
                
                
def train_one_epoch(config, model, optimizer, data_loader, device, epoch, writer, freq_print=10000):
    model.train()
    avg_loss = 0
    dice_epoch = 0
    
        
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        #### Check dimension of the loss sould be bsx1 for later averaging
        batch_loss = dice_loss(output, target)
        loss = torch.mean(batch_loss[:,1])
        loss.backward()
        optimizer.step()
    
            
        if avg_loss == 0 : 
            avg_loss = loss.item()
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        if batch_idx % freq_print == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tavg Loss: {:.6f}\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), avg_loss, loss.item()))



        writer.add_scalar('train_batch_loss', avg_loss, batch_idx +len(data_loader) * epoch)

        dice_epoch += 1 - loss 
            
    dice_epoch = dice_epoch/len(data_loader)
    print('epoch : {0} train_loss : {1} | train_dice : {2}'.format(epoch, avg_loss, dice_epoch))

    writer.add_scalar('train_epoch_loss', avg_loss, epoch)
    writer.add_scalar('train_epoch_dice', dice_epoch, epoch)
    
    return writer
    
            
def evaluate(config, model, data_loader, device, epoch, writer):
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        validation_dice = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Compute the scores
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = dice_loss(output, target)
            batch_dice_loss = torch.mean(batch_loss[:,1])
            validation_loss += batch_dice_loss  
            validation_dice += 1 - batch_dice_loss
            
        val_loss = validation_loss/len(data_loader)
        validation_dice = validation_dice/len(data_loader)
        writer.add_scalar('val_epoch_loss', val_loss, epoch)
        writer.add_scalar('val_epoch_dice', validation_dice, epoch)
        eval_score = {}
        eval_score['val_loss'], eval_score['validation_dice'] = val_loss, validation_dice
        #print('epoch : {} val_loss : {} , top1_acc {},  top3_acc {}'.format(epoch, val_loss, top1_acc, top3_acc))
        print('epoch : {0} val_loss : {1} | dice {2}'.format(epoch, val_loss, validation_dice))
    return writer, eval_score
            
            
def main(raw_args=None):
    """
    Main function to run the code. Can either take the raw_args in argument or get the arguments from the config_file.
    """
    
    #-----------------------------------------------------------------------------------------
    ### First, set the parameters of the function, including the config file, log directory and the seed.
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config_file', required=True, 
                        type=str,help = 'path to the config file for the training')
    parser.add_argument('--logdir', required=True, 
                        type=str,help = 'path to the directory containing all run folders')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader threads')
    parser.add_argument('--seed', type=int, default=np.random.randint(2**32 - 1),
                        help='the seed for reproducing experiments')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to debug or not')
    args = parser.parse_args(raw_args)
    
    print("SEED used is ", args.seed) # Allows to quickly know if you set the wrong seed
    torch.manual_seed(args.seed) # the seed for every torch calculus
    np.random.seed(args.seed) # the seed for every numpy calculus
    #-----------------------------------------------------------------------------------------
    ### Prepare the log by adding the config with runtime and seed
    with open(args.config_file) as json_file:
        config = json.load(json_file)
       
    try :
        print( 'loss type is', config['training']['loss']['type'])
    except:
        if not 'loss' in config['training'].keys() or not 'type' in config['training']['loss'].keys():
            print('loss type is defaulted to classification_only')
            config['training']['loss'] = {'type':'classification_only'}
    config['runtime']= {}
    config['runtime']['num_workers'] = args.num_workers
    config['dataset']['num_workers'] = args.num_workers
    config['runtime']['SEED'] = args.seed
    
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
        
    time = datetime.datetime.today()
    log_id = '{}_{}h{}min'.format(time.date(), time.hour, time.minute)
    log_path = os.path.join(args.logdir,log_id)
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
        json.dump(config, file)
        

    #-----------------------------------------------------------------------------------------
    ### Get the parameters according to the configuration
    config = ObjFromDict(config)
    
    model = get_model(config.model)
    
    data_path = config.dataset.root
    print('data_path ', data_path)
    
    # fix the seed for the split
    split_seed = 0 
    np.random.seed(split_seed)
    
    image_dir = os.listdir(os.path.join(data_path,'Training Batch 1')) + os.listdir(os.path.join(data_path,'Training Batch 2'))
    all_indexes = [ int(file_name[7:-4]) for file_name in image_dir if 'volume' in file_name]
    split = np.random.permutation(all_indexes)
    n_train, n_val, n_test = int(0.8 * len(split)), int(0.1 * len(split)), int(0.1 * len(split))
    
    train = split[: n_train]
    val = split[n_train : n_train+n_val]
    test = split[n_train + n_val :]

    with open(os.path.join(log_path,' splits.json'), 'w+') as file:
        json.dump({
            "train": train.tolist(),
            "val": val.tolist(),
            "test": test.tolist()
        }, file)
    
    # reset the previous seed
    torch.manual_seed(args.seed) # the seed for every torch calculus
    np.random.seed(args.seed)
    
    # Setup Data Loader
    if args.debug: 
        train_dataset = LiTSDataset(data_path, train[:1], augment=False, no_tumor=True)
        val_dataset = LiTSDataset(data_path, train[:1], no_tumor=True)
        test_dataset = LiTSDataset(data_path, train[:1], no_tumor=True)
    else :
        train_dataset = LiTSDataset(data_path, train, augment=True, no_tumor=True)
        val_dataset = LiTSDataset(data_path, val, no_tumor=True)
        test_dataset = LiTSDataset(data_path, test, no_tumor=True)
    
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config.dataset.num_workers,
                                  batch_size=config.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=config.dataset.num_workers,
                                batch_size=config.training.batch_size, shuffle=False)
    test_dataloader  = DataLoader(dataset=test_dataset,  num_workers=config.dataset.num_workers,
                                  batch_size=config.training.batch_size, shuffle=False)
    # Compute on gpu or cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer and learning rate
    optimizer = torch.optim.Adam(params, lr=config.optimizer.learning_rate, 
                                 weight_decay=config.optimizer.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.optimizer.lr_scheduler.step_size,
                                                   gamma=config.optimizer.lr_scheduler.gamma)
    # tensorboard logs
    writer = SummaryWriter(log_path)
    best_scores = {}
    best_scores['validation_dice'] = -1
    #-----------------------------------------------------------------------------------------
    ### Now, we can run the training
    for epoch in range(config.training.epochs):
        writer = train_one_epoch(config, model, optimizer, train_dataloader, device, epoch, writer, freq_print=10000)
        writer, eval_score = evaluate(config, model, val_dataloader, device, epoch, writer)
        lr_scheduler.step()

        if eval_score['validation_dice'] > best_scores['validation_dice']: 
            torch.save(model.state_dict(), os.path.join(log_path,'best_{}.pth'.format('validation_dice')))
            best_scores['validation_dice'] = eval_score['validation_dice']
        elif epoch % 3 == 0 or epoch == config.training.epochs - 1:
            torch.save(model.state_dict(), os.path.join(log_path, 'epoch_{}.pth'.format(epoch)))
         
    writer.close()
    
    return best_scores
        
        
if __name__ == '__main__': 
    main()
