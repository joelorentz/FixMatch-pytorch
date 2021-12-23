from __future__ import division, print_function

import os
import sys
import argparse
import json, hjson
from itertools import repeat

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, LambdaLR

import wandb

root,_ = os.path.split(sys.path[0])
sys.path.append(root)
sys.path.append(os.path.join(root, "classification"))
print(root)
from pytorch_utils.utils import *
from pytorch_utils.models import initialize_model
from pytorch_utils.trainer import Trainer
from pytorch_utils.dataloader import *
from pytorch_utils.plot import *
from dataset.cebi import get_cebi, show_unalbeled_examples, CEBI_MEAN, CEBI_STD



parser = argparse.ArgumentParser()
parser.add_argument(
    'config',
    type=str,
    help='Path to experiment configuration json file')
parser.add_argument(
    '--start_from',
    default=None,
    help='start from old snapshot found at indicated path')
parser.add_argument(
    '--show',
    default=False,
    action="store_true",
    help='Show examples of augmented samples')
parser.add_argument(
    '--show_only',
    default=False,
    action="store_true",
    help='Show examples of augmented samples, then stop computation')
parser.add_argument(
        '--find_lr', default=False, action="store_true", help='Plot best learning rate for setup then leave')
parser.add_argument('--seed', default=0, type=int,
                        help="random seed")

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():

    args = parser.parse_args()
    print("Python Version:", sys.version)
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # set seed
    if args.seed is not None:
        set_seed(args)

    # load config
    with open(args.config,"r") as f:
        config = hjson.load(f)

     # init wandb only if model will be trained
    if not (args.show_only or args.find_lr):
        wandb.init(
            entity="lorentzj",
            project=config["exp_name"],
            dir=config["res_dir"],
            config=config,
        )
        wandb.watch_called = False
        wandb.run.save()
        run_id = wandb.run.name
        print("\n Starting Run %s \n"%run_id)
    

        snap_dir = os.path.join(config["res_dir"], 'snapshots', config["exp_name"], str(run_id))
        config_dir = os.path.join(config["res_dir"], 'config', config["exp_name"])
        
        if not os.path.exists(snap_dir):
                os.makedirs(snap_dir)
    
    labeled_dataset, unlabeled_dataset, test_dataset = get_cebi()

    train_loader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=8,
        num_workers=6,
        drop_last=True,
        pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=8,
        num_workers=6)

    dataloaders_dict = {
        "train":train_loader,
        "val":test_loader
    }
    idx_to_class=[0,1]
    # # Initialize the Dataloaders
    # dataloaders_dict, idx_to_class = get_loader_dict(
    #     config["csv_paths"],
    #     config["transform_args"],
    #     config["oversample"],
    #     config["mini_batch_size"],
    #     config["workers"],
    # )
    # print(idx_to_class)
    # print Dataset size
    print("Train set size: %s / Val set size: %s" %tuple([len(dataloaders_dict[x].dataset) for x in ['train','val']]))

    # Display examples of augmented  images
    if args.show or args.show_only:
        show_examples_multiview(dataloaders_dict['train'],4, config["transform_args"]["train"]['mean'], config["transform_args"]["train"]['std'])
        if args.show_only:
            return

    # # Initialize the model for this run
    model = initialize_model(
        config["model_name"], config["classes"], config["params_to_train"], use_pretrained=config["pretrained"])
    

    # Print the model we just instantiated
    # print(model)

    # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(
        "GPU: %s \n Properties: %s" %
        (torch.cuda.get_device_name(),
         torch.cuda.get_device_properties("cuda:0")))

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run
    params_to_update = []
    print("The following parameters will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)    
            print(name)
    

    # Create optimizer
    if config["optimizer"] == 'sgd':
        optimizer = optim.SGD(
            params_to_update,
            lr=config["lr"],
            momentum=0.9,
            weight_decay=1e-4)
    elif config["optimizer"] == 'adam':
        optimizer = optim.Adam(
            params_to_update,
            lr=config["lr"])
    else:
        raise UnknownOptimizerException("Chosse adam or sgd as optimizer")


    # enable weight decay at milestones
    # if config["optimizer"] == 'sgd':
    #     scheduler = MultiStepLR(
    #         optimizer,
    #         milestones=config["milestones"],
    #         gamma=0.1)
    # else:
    #     scheduler = None


    # scheduler = CosineAnnealingWarmRestarts(optimizer,95,2)
    scheduler = CosineAnnealingWarmRestarts(optimizer,2,2)

    # Setup the loss fxn
    if config["loss"] == 'cross':
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == 'focal':
        if "focal_alpha" in config.keys():
            criterion = FocalLoss(gamma=config["focal_gamma"],alpha=config["focal_alpha"])
        else:
            criterion = FocalLoss(gamma=config["focal_gamma"])

    
    # find best learning rate for setup, then leave script
    if args.find_lr:
        lr_finder = LrFinder(model, optimizer, dataloaders_dict["train"],criterion)
        lrs, losses = lr_finder.find_lr()
        lr_finder.plot(lrs,losses)

        return
    
    # WandB - watch model
    wandb.watch(model,log="all")

    if "batch_size" not in config.keys():
        config["batch_size"] = None
    # Train and evaluate
    trainer = Trainer(
        model,
        dataloaders_dict,
        criterion,
        scheduler,
        optimizer,
        device,
        idx_to_class,
        config["mini_batch_size"],
        batch_size=config["batch_size"],
        use_amp=False,
        is_inception=(config["model_name"] == "inception"),
        snap_dir=snap_dir,
        model_name=config["model_name"],
        exp_name=config["exp_name"],
        mean= config["transform_args"]["train"]['mean'],
        std = config["transform_args"]["train"]['std'],
        )
        

    model, hist = trainer.train(
        config["epochs"], config["start_epoch"], snap_interval=config["snap_interval"], resume_path=args.start_from)

    # test model
    if "num_mistakes" in config.keys():
        trainer.test_model(num_mistakes=config["num_mistakes"])
    
    print("\n Finished with %s Run %s \n" % (config["exp_name"],run_id))


if __name__ == '__main__':
    main()
