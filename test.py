#!/usr/bin/env python

#%%-------------------------------------#
#   Import libraries and set settings   #
#---------------------------------------#

from utilities import *
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v3_large
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
print("Used device for testing: ",device)

TRAIN_DIR = "sets/train_dn/labels.csv"
IM_TRAIN_DIR = "sets/train_dn/images/"
VALID_DIR = "sets/valid_dn/labels.csv"
IM_VALID_DIR = "sets/valid_dn/images/"
TEST_DIR  = "sets/test_dn/labels.csv" 
IM_TEST_DIR  = "sets/test_dn/images/" 
WEIGHTSPATH = './weights/latest.pth'
LABELMAP = ("day", "night")

IMSIZE = 128
BATCHSIZE = 8


#%%------------------------------------#
#   Load test data and trained model   #
#--------------------------------------#

ds_test  = Dataset_dn(csv_file=TEST_DIR, root_dir=IM_TEST_DIR,   rescale=IMSIZE, transform=True)
test_loader  = DataLoader(dataset=ds_test , batch_size=BATCHSIZE, shuffle=True)

model = mobilenet_v3_large(width_mult=1.0,  reduced_tail=False, dilated=False)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, len(LABELMAP))
model.load_state_dict(torch.load(WEIGHTSPATH, map_location=torch.device(device)), strict=False)
model = model.to(device)


#%%-------------------------------#
#   Evaluate model on test data   #
#---------------------------------#
test_model(model=model, dataloader=test_loader)
# %%
