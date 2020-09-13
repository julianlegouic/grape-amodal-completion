"""
    Author: Julian LE GOUIC
    Python version: 3.6.7

    IMP Lab - Osaka Prefecture University.
"""

import argparse
import datetime
import functools
import glob
import os

import torch
from torch.utils.data import DataLoader

from unet.synGrapeDataset import data_split
from unet.synGrapeDataset import SynGrapeDataset

from unet.model import Unet as Model

from unet.utils import my_collate, number_min_epoch

# Allows switching to gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', dest='batch_size', type=int,
                    default=4, help='Specify the batch size')
parser.add_argument('--workers', '-w', dest='num_workers', type=int,
                    default=4, help='num_workers for DataLoader function')
parser.add_argument('--learning_rate', '-lr', dest='learning_rate', type=float,
                    default=0.001, help='Learning rate for the network')
parser.add_argument('--epoch', '-e', dest='nb_epoch', type=number_min_epoch,
                    default=50, help='Set the number of epochs (> 5)')
parser.add_argument('--threshold', '-thresh', dest='threshold', type=float,
                    default=0.5, help='Threshold for evaluation.')
parser.add_argument('--augmentation', '-aug', dest='data_aug',
                    action='store_true', default=False, required=False,
                    help='Do data augmentation.')

# Training or evaluation mode
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--training', '-train', dest='training', action='store_true',
                  help='Set training mode')
mode.add_argument('--evaluate', '-eval', dest='evaluate', action='store_true',
                  help='Set evaluation mode')

# Activate debugging mode or not
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--debug', '-d', dest='debug', action='store_true',
                   help="Set the experiment's name to 'debug' for debugging")
group.add_argument('--experiment', '-exp', dest='exp', type=str,
                   help='Extension for the name of the experiment')
args = parser.parse_args()

# Loading dataset
data_file = './data/synthetic_grape/fname_dataset.json'
train_df, val_df, test_df = data_split(data_file)

classes = ('grape')

transform = True

##############################################################################
# TRAINING
##############################################################################
if args.training:
    # Create a folder name for each experiment
    date = str(datetime.datetime.now()).split(' ')[0].split('-')
    year, month, day = date[0][2:], date[1], date[2]
    if args.debug:
        exp_name = 'debug'
    else:
        exp_name = Model.__name__+'-'+month+day+year + \
            '-lr-'+str(args.learning_rate) + \
            '-b'+str(args.batch_size)+'_'+args.exp

    experiment_name = os.path.join('unet', 'output', exp_name)

    # Creating folders for the experience
    if os.path.exists(experiment_name) and (exp_name != 'debug'):
        print('The experiment name you chose already exists.',
              'Overwrite its content? (y/[n])')
        choice = None
        while choice != 'y' and choice != 'n' and choice != '':
            choice = input()
        if choice == 'y':
            # Clear the logs files
            log_folder = os.path.join(experiment_name, 'train', 'logs')
            files = glob.glob(f'{log_folder}/*')
            for f in files:
                os.remove(f)
        else:
            print('Process aborted.')
            exit(-1)
    else:
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
            os.makedirs(os.path.join(experiment_name, 'eval'))
            os.makedirs(os.path.join(experiment_name, 'train', 'logs'))
            os.makedirs(os.path.join(experiment_name, 'train', 'models'))

    trainset = SynGrapeDataset(data=train_df, transform=transform,
                               data_aug=args.data_aug)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=my_collate,
                              num_workers=args.num_workers)

    valset = SynGrapeDataset(data=val_df, transform=transform)
    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=my_collate,
                            num_workers=args.num_workers)

    data_loader = [train_loader, val_loader]

    # Create the network and put it on GPU
    model = Model(nb_classes=1, experiment=experiment_name, device=device)
    model.to(device)

    # Train the network
    model.train_model(data_loader, args.nb_epoch, args.learning_rate)

##############################################################################
# EVALUATION
##############################################################################
if args.evaluate:
    # Used when debugging because args.exp is not defined
    if args.debug:
        args.exp = os.path.join('unet', 'output', 'debug')

    testset = SynGrapeDataset(data=test_df.copy(), transform=transform)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=my_collate,
                             num_workers=args.num_workers)

    # Create a model with same parameters as the first one
    model_load = Model(nb_classes=1, experiment=args.exp, device=device)

    # Absolute path to the best model file
    best_model = os.path.abspath(os.path.join(
        args.exp, 'train', 'models', 'best_model.pth'))

    # Load the best model and put it on GPU
    model_load.load_state_dict(
        torch.load(
            best_model,
            map_location=device
        )
    )
    model_load.to(device)

    # Evaluate the network
    model_load.predict(test_loader, args.threshold)
