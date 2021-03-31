import argparse
from utils.utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, **kwargs):
    parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='./store/models', dest='m_dir', help="default: %(default)s")
    parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir',help="default: %(default)s")
    return parser


def add_task_options(parser, **kwargs):
    # expirimental task parameters
    task_params = parser.add_argument_group('Task Parameters')
    MNIST_tasks = ['splitMNIST', 'permMNIST']
    image_tasks = ['CIFAR10', 'CIFAR100']
    task_choices = MNIST_tasks+image_tasks
    task_default = 'CIFAR100'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=task_choices)
    task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'class'])

    task_params.add_argument('--augment', action='store_true',
                             help="augment training data (random crop & horizontal flip)")
    task_params.add_argument('--no-norm', action='store_false', dest='normalize',
                             help="don't normalize images (only for CIFAR)")
    # 'task':   each task has own output-units, always only those units are considered
    # 'class':  each task has own output-units, all units of tasks seen so far are considered
    task_params.add_argument('--tasks', type=int, help='number of tasks')
    return parser


def add_model_options(parser, generative=False, **kwargs):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    # -conv-layers
    model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
    model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
    model.add_argument('--depth', type=int, default=None,
                        help="# of convolutional layers (0 = only fc-layers)")
    model.add_argument('--reducing-layers', type=int, dest='rl',help="# of layers with stride (=image-size halved)")
    model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
    model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
    model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
    model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")

    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, default=None, metavar="N",
                       help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    if generative:
        model.add_argument('--z-dim', type=int, default=100,help='size of latent representation (if feedback, def=100)')
    return parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args, **kwargs):

    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.augment = args.augment if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.tasks= (5 if args.experiment=='splitMNIST' else (10 if args.experiment=="CIFAR100" else 10)
                 ) if args.tasks is None else args.tasks

    return args