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
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data_loader import setup_dataset
from models import get_model
from utils.utilities import ObjFromDict

class ObjFromDict:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ObjFromDict(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, ObjFromDict(b) if isinstance(b, dict) else b)
                
                
def train_one_epoch(config, model, optimizer, data_loader, device, epoch, writer, freq_print=10000, metrics=['fieldguide_score','accuracy']):
    model.train()
    avg_loss = 0
    correct = correct_top3 = 0
    top1_epoch_acc = top3_epoch_acc = 0
    
    if config.training.loss.type == 'geometric':
        lambda_ = config.training.loss.coef
        target_embedding = torch.zeros((config.model.num_classes, model.classifier.in_features)).to(device)
        
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), (target['label']).to(device)
        optimizer.zero_grad()
        
        if config.training.loss.type == 'geometric': 
            output, embedding = model(data)
            if epoch > 0 : 
                mask = torch.ones(config.training.batch_size,model.classifier.out_features).to(device).long().scatter_(1, target.view(config.training.batch_size,1), torch.tensor(-1).float().to(device)).detach()
                representation_loss = torch.mean(torch.sum(mask * torch.sum((embedding.view(config.training.batch_size,1,model.classifier.in_features) - target_embedding.view(1,model.classifier.out_features, model.classifier.in_features))**2, dim = -1), dim = -1))
                representation_loss = representation_loss / (model.classifier.out_features * model.classifier.in_features *config.training.batch_size)
                loss = (1 - lambda_) * F.cross_entropy(output, target) + lambda_ * representation_loss
            else:
                loss = F.cross_entropy(output, target)
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        
        if config.training.loss.type == 'geometric_loss':
            target_embedding[target] = (1-momentum) * target_embedding[target] + momentum * torch.squeeze(embedding)
            target_embedding = target_embedding.detach()
            
        if avg_loss == 0 : 
            avg_loss = loss.item()
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        if batch_idx % freq_print == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tavg Loss: {:.6f}\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), avg_loss, loss.item()))
            if epoch> 0 : print(representation_loss)
        for metric in metrics:
            if metric=='fieldguide_score': 
                prob_pred_top3, class_pred_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
                for idx in range(len(target)):
                    if target[idx] in class_pred_top3[idx]:
                        correct_top3 += 1
            else :
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        top1_batch_acc = correct/len(data)
        top3_batch_acc = correct_top3/len(data)

        writer.add_scalar('train_batch_loss', avg_loss, epoch)
        writer.add_scalar('train_batch_acc_top1', top1_batch_acc, epoch)
        writer.add_scalar('train_batch_acc_top3', top3_batch_acc, epoch)

        top1_epoch_acc += top1_batch_acc
        top3_epoch_acc += top3_batch_acc
            
            
    writer.add_scalar('train_epoch_loss', avg_loss, epoch)
    writer.add_scalar('train_epoch_acc_top1', top1_epoch_acc/(data_loader.batch_size * len(data_loader.dataset)), epoch)
    writer.add_scalar('train_epoch_acc_top3', top3_epoch_acc/(data_loader.batch_size * len(data_loader)), epoch)
    
    return writer
    
            
def evaluate(config, model, data_loader, device, epoch, writer, loss_function=F.cross_entropy, metrics=['fieldguide_score','accuracy']):
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        correct = 0
        correct_top3 = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx==0:
                # Record some of the images
                grid = torchvision.utils.make_grid(data.cpu())
                writer.add_image('images', grid, 0)
            # Compute the scores
            data, target = data.to(device), (target['label']).to(device)
            if config.training.loss.type == 'geometric':
                output,_ = model(data)
            else:
                output = model(data)
            validation_loss += loss_function(output, target).item()  # sum up batch loss
            for metric in metrics:
                if metric=='fieldguide_score': 
                    prob_pred_top3, class_pred_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
                    for idx in range(len(target)):
                        if target[idx] in class_pred_top3[idx]:
                            correct_top3 += 1
                else :
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss = validation_loss/len(data_loader)
        top1_acc = correct/len(data_loader.dataset)
        top3_acc = correct_top3/len(data_loader.dataset)
        writer.add_scalar('val_epoch_loss', val_loss, epoch)
        writer.add_scalar('val_epoch_acc_top1', top1_acc, epoch)
        writer.add_scalar('val_epoch_acc_top3', top3_acc, epoch)
        eval_score = {}
        eval_score['fieldguide_score'], eval_score['accuracy'] = top3_acc, top1_acc
        #print('epoch : {} val_loss : {} , top1_acc {},  top3_acc {}'.format(epoch, val_loss, top1_acc, top3_acc))
        print('epoch : {0} val_loss : {1} | top1_correct {2}/{3} -> top1_acc {4} | top3_correct {5}/{3} -> top3_acc {6}'.format(epoch, val_loss, correct, len(data_loader.dataset), top1_acc, correct_top3, top3_acc))
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
    
    model = get_model(config.model, config.training.loss.type)
    train_dataloader, val_dataloader = setup_dataset(config, debug=args.debug)
    # Compute on gpu or cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    ### Set the variables
    # loss function : Doesn't depend on the model right ?
    loss_function = F.cross_entropy
    # trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer and learning rate
    optimizer = torch.optim.Adam(params, lr=10**(-config.optimizer.learning_rate), 
                                 weight_decay=10**(-config.optimizer.weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.optimizer.lr_scheduler.step_size,
                                                   gamma=config.optimizer.lr_scheduler.gamma)
    # tensorboard logs
    writer = SummaryWriter(log_path)
    best_scores = {}
    metrics = ['fieldguide_score','accuracy']
    for metric in metrics : 
        best_scores[metric] = -1
    #-----------------------------------------------------------------------------------------
    ### Now, we can run the training
    for epoch in range(config.training.epochs):
        writer = train_one_epoch(config, model, optimizer, train_dataloader, device, epoch, writer, freq_print=10000, metrics=metrics)
        writer, eval_score = evaluate(config, model, val_dataloader, device, epoch, writer, metrics=metrics)
        lr_scheduler.step()
        for metric in metrics: 
            if eval_score[metric] > best_scores[metric]: 
                torch.save(model.state_dict(), os.path.join(log_path,'best_{}.pth'.format(metric)))
                best_scores[metric] = eval_score[metric]
         
    writer.close()
    
    return best_scores
        
        
if __name__ == '__main__': 
    main()