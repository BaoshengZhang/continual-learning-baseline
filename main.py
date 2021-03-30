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
from eval import evaluate
from eval import callbacks as cb
import eval.precision_recall as pr
import eval.fid as fid
from train.train import train_cl
from utils.param_stamp import get_param_stamp
from models.cl.continual_learner import ContinualLearner


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"  # use gpu 0,1,2,3,4
## Function for specifying input-options and organizing / checking them

def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'none'}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare & combine continual learning approaches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
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
        verbose=verbose, exception=True if args.seed<10 else False, only_test=(not args.train)
    )


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


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------------------------#
    #----- CL-STRATEGY: REGULARIZATION / ALLOCATION -----#
    #----------------------------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'ewc'):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.online = utils.checkattr(args, 'online')
        if model.online:
            model.gamma = args.gamma

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'xdg') and args.xdg_prop>0:
        model.define_XdGmask(gating_prop=args.xdg_prop, n_tasks=args.tasks)

    # add REGULARIZATION methods here and in ContinualLearner
    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay') and not args.replay=="none":
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = (hasattr(args, 'replay') and args.replay=="generative" and not utils.checkattr(args, 'feedback'))
    if train_gen:
        # Specify architecture
        generator = define.define_autoencoder(args, config, device, generator=True)

        # Initialize parameters
        generator = define.init_params(generator, args)
        # -freeze weights of conv-layers?
        if utils.checkattr(args, "freeze_convE"):
            for param in generator.convE.parameters():
                param.requires_grad = False
        if utils.checkattr(args, "freeze_convD"):
            for param in generator.convD.parameters():
                param.requires_grad = False

        # Set optimizer(s)
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()),
             'lr': args.lr_gen if hasattr(args, 'lr_gen') else args.lr},
        ]
        generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
    else:
        generator = None

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        train_cl(
            model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else "none",
            scenario=args.scenario, classes_per_task=classes_per_task, iters=args.iters,
            batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
            generator=generator, gen_iters=g_iters, gen_loss_cbs=generator_loss_cbs,
            feedback=utils.checkattr(args, 'feedback'), sample_cbs=sample_cbs, eval_cbs=eval_cbs,
            loss_cbs=generator_loss_cbs if utils.checkattr(args, 'feedback') else solver_loss_cbs,
            args=args, reinit=utils.checkattr(args, 'reinit'), only_last=utils.checkattr(args, 'only_last')
        )
        # Save evaluation metrics measured throughout training
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(precision_dict, file_name)
        # Save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
            if generator is not None:
                save_name = "gM-{}".format(param_stamp) if (
                    not hasattr(args, 'full_stag') or args.full_stag == "none"
                ) else "{}-{}".format(generator.name, args.full_stag)
                utils.save_checkpoint(generator, args.m_dir, name=save_name, verbose=verbose)

    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of the previously trained models...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose,
                              add_si_buffers=(isinstance(model, ContinualLearner) and utils.checkattr(args, 'si')))
        if generator is not None:
            load_name = "gM-{}".format(param_stamp) if (
                not hasattr(args, 'full_ltag') or args.full_ltag == "none"
            ) else "{}-{}".format(generator.name, args.full_ltag)
            utils.load_checkpoint(generator, args.m_dir, name=load_name, verbose=verbose)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- EVALUATION of CLASSIFIER-----#
    #-----------------------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if args.scenario=="task" else None
    ) for i in range(args.tasks)]
    average_precs = sum(precs)/args.tasks
    # -print on screen
    if verbose:
        print("\n Accuracy of final model on test-set:")
        for i in range(args.tasks):
            print(" - {} {}: {:.4f}".format("For classes from task" if args.scenario=="class" else "Task",
                                            i + 1, precs[i]))
        print('=> Average accuracy over all {} {}: {:.4f}\n'.format(
            args.tasks*classes_per_task if args.scenario=="class" else args.tasks,
            "classes" if args.scenario=="class" else "tasks", average_precs
        ))
    # -write out to text file
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs))
    output_file.close()

if __name__ == '__main__':
    args = handle_inputs()
    run(args)