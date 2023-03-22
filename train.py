#!/usr/bin/env python

# Used tutorials:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/asabuncuoglu13/custom-vision-pytorch-mobile/blob/main/torch_transfer_learning_mobilenet3.ipynb
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from comet_ml import Experiment
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import warnings # Ignore warnings
from utilities import *
import time
import copy
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import datetime
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

# Set project name
P_NAME = "pytorch_classifier"

# Set root path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Set device for pytorch
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
device = torch.device(dev_str)
print("Used device for training: ",device)

#----------------------------#
#   Parse arguments method   #
#----------------------------#
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data', type=str, default=ROOT / 'data/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--enable_comet', action='store_true', help='enable comet.ml')
    parser.add_argument('--save_dir', type=str, default='weights', help='directory to save weights')

    return parser.parse_known_args()[0] if known else parser.parse_args()


#%%----------------------#
#   Train model method   #
#------------------------#
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, experiment = None):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if device == "cuda":
        inputs, labels = inputs.cuda(), labels.cuda()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Training")
            else:
                model.eval()   # Set model to evaluate mode
                print("Validating")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for d in tqdm(dataloaders[phase]):

                the_input = d['image'].to(device)
                the_label = d['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(the_input)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, the_label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * the_input.size(0)
                running_corrects += torch.sum(preds == the_label.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train' and (experiment != None):
                experiment.log_metric("train_accuracy", epoch_acc, epoch=epoch)
                experiment.log_metric("train_loss", epoch_loss, epoch=epoch)
            elif phase == 'val' and (experiment != None):
                experiment.log_metric("val_accuracy", epoch_acc, epoch=epoch)
                experiment.log_metric("val_loss", epoch_loss, epoch=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#-----------------#
#   Main method   #
#-----------------#
def main(opt):

    print("Train args: ", vars(opt))
    epochs, batch_size, learn_rate, data, img_size, enable_comet, save_dir = opt.epochs, opt.batch_size, opt.learn_rate, opt.data, opt.img_size, opt.enable_comet, opt.save_dir

    # Read data.yaml file to get classes, classes count, train and val paths
    with open(data, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Get data paths from data.yaml file
    train_path, val_path = get_data_paths(data)    

    print("Training data path: ", train_path)
    print("Validation data path: ", val_path)

    # Define hyperparameters for cometML logging
    hyper_params = {
        "input_size": img_size,
        "num_classes": data['nc'],
        "batch_size": batch_size,
        "num_epochs": epochs,
        "learning_rate": learn_rate
    }

    # Create cometML experiment and start logging
    if enable_comet:
        # Add your CometML info to track your training online
        experiment = Experiment(
            api_key = os.environ['COMET_API_KEY'],
            project_name = P_NAME,
            workspace="",
        )
        experiment.log_parameters(hyper_params)

    # Create datasets
    ds_train = Dataset_classifier(data_path=train_path, rescale=img_size, transform=True)
    ds_valid = Dataset_classifier(data_path=val_path, rescale=img_size, transform=True)

    print("Train dataset size: ", ds_train.__len__())
    print("Validation dataset size: ", ds_valid.__len__())

    # Create dataloaders from datasets
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)

    dataloaders   = {'train': train_loader,  'val': valid_loader }
    dataset_sizes = {'train': len(ds_train), 'val': len(ds_valid)}

    # Loading the model (LeNet)
    model = LeNet()
    print("Number of model parameters: ", count_parameters(model))

    # Move model to device
    model = model.to(device)
    # Set loss function
    criterion = nn.CrossEntropyLoss()
    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= round(batch_size/10) if (round(batch_size/10) >= 1) else 1, gamma=0.5)

    # Start training
    if enable_comet:
        with experiment.train():
            model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs, experiment=experiment)
    else:
        model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)


    # Save model weights
    folder_name = P_NAME + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(os.getcwd(), save_dir, folder_name)
    save_weights(model, save_dir)
    
    # Visualize model predictions and save the figure
    fig_path = visualize_model(model,valid_loader, data['names'], save_dir, num_images=12) 
    
    # Log the figure to cometML and end the experiment
    if enable_comet:
        experiment.log_image(fig_path)
        experiment.end()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)