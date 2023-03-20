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
import argparse
import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # <-- this is the root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
P_NAME = "pytorch_classifier"

dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
device = torch.device(dev_str)
print("Used device for training: ",device)

#%%-----------------------------------------------#
#   Define data directories and hyperparameters   #
#-------------------------------------------------#


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


#%%-------------------#
#   Parse arguments   #
#---------------------#
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data', type=str, default=ROOT / 'data/fsoco.yaml', help='dataset.yaml path')
    parser.add_argument('--save_weights', type=bool, default=True, help='save weights')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--enable_comet', type=bool, default=False, help='enable comet.ml')
    parser.add_argument('--device', default='', help='cuda, cpu or mps for Aplle devices')
    parser.add_argument('--save_dir', type=str, default='weights', help='directory to save weights')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    print("Train args: ", vars(opt))

    epochs, batch_size, learn_rate, data, save_weights, img_size, enable_comet, device, save_dir = opt.epochs, opt.batch_size, opt.learn_rate, opt.data, opt.save_weights, opt.img_size, opt.enable_comet, opt.device, opt.save_dir

    if device == '':
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        dev_str = "cpu" if torch.has_mps else dev_str
        device = torch.device(dev_str)
    else:
        device = torch.device(device)
    print("Used device for training: ",device)

######################################################################################

    with open(data, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for k in 'train', 'val', 'names':
        assert k in data, f" data.yaml '{k}:' field missing"
    if isinstance(data['names'], (list, tuple)):  # old array format
        data['names'] = dict(enumerate(data['names']))  # convert to dict
    assert all(isinstance(k, int) for k in data['names'].keys()), 'data.yaml names keys must be integers, i.e. 2: car'
    data['nc'] = len(data['names'])

    path = Path(data.get('path'))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    train_path, val_path = data['train'], data['val']

    print("train_path: ", train_path)
    print("val_path: ", val_path)

######################################################################################

    hyper_params = {
        # "sequence_length": 28,
        "input_size": img_size,
        # "hidden_size": 128,
        # "num_layers": 2,
        "num_classes": data['nc'],
        "batch_size": batch_size,
        "num_epochs": epochs,
        "learning_rate": learn_rate
    }
    if enable_comet:
        # Add your CometML info to track your training online
        experiment = Experiment(
            api_key = os.environ['COMET_API_KEY'],
            project_name = P_NAME,
            workspace="",
        )
        experiment.log_parameters(hyper_params)

######################################################################################

    # TODO: Change Dataset class to work with the json format
    ds_train = Dataset_classifier(data_path=train_path, rescale=img_size, transform=True)
    ds_valid = Dataset_classifier(data_path=val_path, rescale=img_size, transform=True)

    print("Train dataset size: ", ds_train.__len__())
    print("Validation dataset size: ", ds_valid.__len__())

    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)

    dataloaders   = {'train': train_loader,  'val': valid_loader }
    dataset_sizes = {'train': len(ds_train), 'val': len(ds_valid)}

    # Loading the model
    # model = mobilenet_v3_large()
    # model = torchvision.models.mobilenet_v3_small(weights=True, width_mult=1.0,  reduced_tail=False, dilated=False)
    # model = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights, width_mult=1.0,  reduced_tail=False, dilated=False)

    model = Net()
    print("Number of model parameters: ", count_parameters(model))

    # Set training tools
    # for param in model.parameters():
    #     param.requires_grad = False
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model.classifier[0].in_features
    # # Here the size of each output sample is set to 2. Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.classifier = nn.Linear(num_ftrs, data['nc'])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before.
    # optimizer = optim.SGD(model.classifier.parameters(), lr=learn_rate, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= round(batch_size/10) if (round(batch_size/10) >= 1) else 1, gamma=0.5)

    if enable_comet:
        with experiment.train():
            model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs, experiment=experiment)
        experiment.end()
    else:
        model = train_model(model,dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)

    ###############################################

    fig_save_path = save_dir + "/output.png"

    ###############################################

    # if save_weights:
    folder_name = P_NAME + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mydir = os.path.join(os.getcwd(), save_dir, folder_name)
    os.makedirs(mydir)
    savedir = os.path.join(mydir, "weights.pth")
    torch.save(model.state_dict(), os.path.join(save_dir, mydir, "weights.pth"))
    print("Saved weights to: ", savedir)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), "wmslatest.pth"))

    fig_save_path = os.path.join(mydir, "output.png")
    visualize_model(model,valid_loader, data['names'], fig_save_path, experiment, num_images=12) 
    if enable_comet:
        experiment.log_image(fig_save_path)
        # print("Logged image to CometML: ", fig_save_path)
        experiment.end()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)