# DLMI_segmentation
This is the repository of the Efficient 3D liver segmementation of the "CentraleSupelec - Spring 2020 MVA-DLMI: Deep Learning in Medical Imaging" course. 
In this project we tested a 3D version of Deeplab with a mobile net ad a resnet backbone and compare it to a U-net and a attention gated Unet

The group of student was composed of three students : 
- [Emmanuel Jehanno](https://github.com/EmmanuelJhno)
- [Simon Brandeis](https://github.com/SBrandeis)
- [Brice Rauby](https://github.com/bricerauby)

The report is available on demand.

## Prerequisites
Python3

## Installation 
  1. Download the data from the [LiTS challenge](https://competitions.codalab.org/competitions/17094) (train batch 1 and train batch 2)
  2. Clone this repository
  3. Run ```pip install -r requirements.txt```
  
## Getting Started
All the commands are to be executed from the main directory of this repository.
### Training 
- From the `configs` directory, chose the json file you want to train or create a new one and adapt the datapath to the folder containing ```Training Batch 1``` and ```Training Batch 2``` downloaded. 
- To launch a training run ```python utils/train.py --config_file=<path to config.json> --logdir=<oath to the directory containing all the log dirs>```

### Evaluation 
To evaluate a run use : 
```python utils/error_analysis.py –run_dir=<path to the log dir>``` (the log dir is the one named after the date and the time of the training)

To run a the test time evalutation, run: 
```python utils/speed_test.py```
