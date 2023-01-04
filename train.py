#!/usr/bin/env python

# Used tutorials:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/asabuncuoglu13/custom-vision-pytorch-mobile/blob/main/torch_transfer_learning_mobilenet3.ipynb

#%%-------------------------------------#
#   Import libraries and set settings   #
#---------------------------------------#
from __future__ import print_function, division
import torch
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights
import warnings # Ignore warnings
from comet_ml import Experiment
from utilities import *
import time
import datetime
import copy
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
device = torch.device(dev_str)
print("Used device for training: ",device)

#%%-----------------------------------------------#
#   Define data directories and hyperparameters   #
#-------------------------------------------------#
# Dirs
TRAIN_DIR = "sets/train_dn/labels.csv"
IM_TRAIN_DIR = "sets/train_dn/images/"
VALID_DIR = "sets/valid_dn/labels.csv"
IM_VALID_DIR = "sets/valid_dn/images/"
MODELPATH = 'weights/'
LABELMAP = ("day", "night")

# Hyperparameters
ENABLECOMET = False
BATCHSIZE = 8
EPOCHS = 2
IMSIZE = 640 # rescales to that value
NUMCLASSES = len(LABELMAP)
LEARNRATE = 0.001
SAVEWEIGHTS = True


#%%----------------------#
#   Train model method   #
#------------------------#
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    print("#--------------------#")
    print("#   Start training   #")
    print("#--------------------#")
    print("Training parameters: ")
    print(" ")
    for key, value in hyper_params.items():
        print(key, ' : ', value)

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
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for d in dataloaders[phase]:

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

            if phase == 'train' and ENABLECOMET:
                experiment.log_metric("train_accuracy", epoch_acc, epoch=epoch)
                experiment.log_metric("train_loss", epoch_loss, epoch=epoch)
            elif phase == 'val' and ENABLECOMET:
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


#%%--------------------------------------------------#
#   Setup comet connection and log hyperparameters   #
#----------------------------------------------------#
hyper_params = {
        # "sequence_length": 28,
        "input_size": IMSIZE,
        # "hidden_size": 128,
        # "num_layers": 2,
        "num_classes": NUMCLASSES,
        "batch_size": BATCHSIZE,
        "num_epochs": EPOCHS,
        "learning_rate": LEARNRATE
}
if ENABLECOMET:
    # Add your CometML info to track your training online
    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
    )
    experiment.log_parameters(hyper_params)


#%%-----------------------------------------------------#
#   Loading the data and put them into the dataloader   #
#-------------------------------------------------------#
ds_train = Dataset_dn(csv_file=TRAIN_DIR, root_dir=IM_TRAIN_DIR, rescale=IMSIZE, transform=True)
ds_valid = Dataset_dn(csv_file=VALID_DIR, root_dir=IM_VALID_DIR, rescale=IMSIZE, transform=True)

train_loader = DataLoader(dataset=ds_train, batch_size=BATCHSIZE, shuffle=True)
valid_loader = DataLoader(dataset=ds_valid, batch_size=BATCHSIZE, shuffle=True)

dataloaders   = {'train': train_loader,  'val': valid_loader }
dataset_sizes = {'train': len(ds_train), 'val': len(ds_valid)}


#%%--------------------------------------------#
#   Load model and define tools for training   #
#----------------------------------------------#

# Loading the model
# model = mobilenet_v3_large()
# model = torchvision.models.mobilenet_v3_small(weights=True, width_mult=1.0,  reduced_tail=False, dilated=False)
model = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights, width_mult=1.0,  reduced_tail=False, dilated=False)

# Set training tools
for param in model.parameters():
    param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.classifier[0].in_features
# Here the size of each output sample is set to 2. Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.classifier = nn.Linear(num_ftrs, len(LABELMAP))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that only parameters of final layer are being optimized as opposed to before.
optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNRATE, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= round(BATCHSIZE/10) if (round(BATCHSIZE/10) >= 1) else 1, gamma=0.5)


#%%-------------------#
#   Train the model   #
#---------------------#
if ENABLECOMET:
    with experiment.train():
        model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)
    experiment.end()
else:
    model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)


#%%---------------------------------------------------#
#   Visualize some output on the validation dataset   #
#-----------------------------------------------------#
visualize_model(model,valid_loader, LABELMAP)


#%%-------------------------#
#   Save trained wheights   #
#---------------------------#
if SAVEWEIGHTS:
    mydir = os.path.join(os.getcwd(), MODELPATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    savedir = os.path.join(mydir, "weights.pth")
    torch.save(model.state_dict(), os.path.join(MODELPATH, mydir, "weights.pth"))
    print("Saved weights to: ", savedir)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), "wmslatest.pth"))
# %%
