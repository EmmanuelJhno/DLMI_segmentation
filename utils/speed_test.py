import time 
import os
import sys
import json
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.train import ObjFromDict


input = torch.zeros(2,1,64,64,64)
n_inference = 5




config_file = 'configs/attention_gated_unet/bs_2.json'
with open(config_file) as json_file:
    config = json.load(json_file)
    
config = ObjFromDict(config)
model = get_model(config.model) 
print(15*'-'+'time performances for attention_gated_unet' + 15*'-')
start = time.time()
for i in range(n_inference):
    intermediate = time.time()
    with torch.no_grad():
        model(input)
    print('time_for_1 :', time.time()- intermediate)
print('mean time for {} inference'.format(n_inference), (time.time()-start)/n_inference)
print('model trainable parameters', sum(p.numel() for p in model.parameters()))

config_file = 'configs/unet/bs_2.json'
with open(config_file) as json_file:
    config = json.load(json_file)
    
config = ObjFromDict(config)
model = get_model(config.model) 
print(15*'-'+'time performances for unet' + 15*'-')
start = time.time()
for i in range(n_inference):
    intermediate = time.time()
    with torch.no_grad():
        model(input)
    print('time_for_1 :', time.time()- intermediate)
    
print('mean time for {} inference'.format(n_inference), (time.time()-start)/n_inference)
print('model trainable parameters', sum(p.numel() for p in model.parameters()))


config_file = 'configs/deeplab_resnet/bs_2.json'
with open(config_file) as json_file:
    config = json.load(json_file)
    
config = ObjFromDict(config)
model = get_model(config.model) 
print(15*'-'+'time performances for deeplab_resnet' + 15*'-')
start = time.time()
for i in range(n_inference):
    intermediate = time.time()
    with torch.no_grad():
        model(input)
    print('time_for_1 :', time.time()- intermediate)
    
print('mean time for {} inference'.format(n_inference), (time.time()-start)/n_inference)
print('model trainable parameters', sum(p.numel() for p in model.parameters()))


config_file = 'configs/deeplab_mobilenet/bs_2.json'
with open(config_file) as json_file:
    config = json.load(json_file)
    
config = ObjFromDict(config)
model = get_model(config.model) 
print(15*'-'+'time performances for deeplab_mobilenet' + 15*'-')
start = time.time()
for i in range(n_inference):
    intermediate = time.time()
    with torch.no_grad():
        model(input)
    print('time_for_1 :', time.time()- intermediate)
    
print('mean time for {} inference'.format(n_inference), (time.time()-start)/n_inference)
print('model trainable parameters', sum(p.numel() for p in model.parameters()))
