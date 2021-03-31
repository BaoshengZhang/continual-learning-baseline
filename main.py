#!/usr/bin/env python3
import numpy as np
import os
from scipy.stats import entropy
import torch
from torch import optim
from torch.utils.data import ConcatDataset
from torch.nn import functional as F

# -custom-written libraries
import utils.options as options
import utils.utils as utils
import utils.define_models as define
from data.load import get_multitask_experiment


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"  # use gpu 0,1,2,3,4
## Function for specifying input-options and organizing / checking them

def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options

    # Define input options
    parser = options.define_args(filename="main", description='Compare continual learning approaches.')
    parser = options.add_general_options(parser,)
    parser = options.add_task_options(parser,)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args,)
    return args


## Function for running one continual learning experiment
def run(args):

    # Use cuda
    device = torch.device("cuda")

    # Set random seeds
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,
        normalize=True if utils.checkattr(args, "normalize") else False,
        augment=True if utils.checkattr(args, "augment") else False,
        exception=True if args.seed<10 else False, only_test=False
    )
    print(" --> scenario is {}, it has {} tasks, class per task is {}.".format(args.scenario,
                                                                                args.tasks,
                                                                                classes_per_task))
    exit()

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- MAIN MODEL -----#
    #----------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if verbose and (utils.checkattr(args, "pre_convE") or utils.checkattr(args, "pre_convD")) and \
            (hasattr(args, "depth") and args.depth>0):
        print("\nDefining the model...")
    if utils.checkattr(args, 'feedback'):
        model = define.define_autoencoder(args=args, config=config, device=device)
    else:
        model = define.define_classifier(args=args, config=config, device=device)

    # Initialize / use pre-trained / freeze model-parameters
    # - initialize (pre-trained) parameters
    model = define.init_params(model, args)
    # - freeze weights of conv-layers?
    if utils.checkattr(args, "freeze_convE"):
        for param in model.convE.parameters():
            param.requires_grad = False
    if utils.checkattr(args, 'feedback') and utils.checkattr(args, "freeze_convD"):
        for param in model.convD.parameters():
            param.requires_grad = False

    # Define optimizer (only optimize parameters that "requires_grad")
    model.optim_list = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    ]
    model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

if __name__ == '__main__':
    args = handle_inputs()
    run(args)