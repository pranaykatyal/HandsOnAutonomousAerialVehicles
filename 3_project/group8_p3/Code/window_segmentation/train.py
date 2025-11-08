# IMPORTS----------------------------------------------------------------------------
# STANDARD
import sys
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Subset

import wandb

# CUSTOM
from network import *
from utils import *
from dataloader import *
import pdb
import utils

# Load the parameters
from Params import *

#  TRAIN ----------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epochstep):
    
    # dp('train started')
    model.train()
    epochloss = 0
    for batchcount, (rgb, label) in enumerate(dataloader):
        dp(' batch', batchcount)
        
        rgb = rgb.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        pred = model(rgb)
        print(f'logits info: min{pred.min()}, max:{pred.max()}')
        loss = loss_fn(pred, label)        
        loss.backward()
        optimizer.step()
        
        epochloss += loss.item()

        wandb.log({
            "epochstep": epochstep,
            "batch/loss/train": loss.item(),
                })
            
        if batchcount == 0: # only for the first batch every epoch
            wandb_images = []
            for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                combined_image_np = CombineImages(pred_single, label_single, rgb_single)

                # Create wandb.Image object and append to the list
                wandb_images.append(wandb.Image(combined_image_np))

            wandb.log(
            {
                "images/train": wandb_images,
            })
                    
    if shouldLog():
        wandb.log({
            "epoch/loss/train": epochloss,
                    })
    
# Define the val function
def val(dataloader, model, loss_fn, epochstep):
    model.eval()
    
    epochloss = 0
    with torch.no_grad():
        for batchcount, (rgb, label) in enumerate(dataloader):
            dp(' batch', batchcount)
            
            rgb = rgb.to(device)
            label = label.to(device)
            
            pred = model(rgb)
            # print(pred)
            # print(pred.shape)
            loss = loss_fn(pred, label)       

            epochloss += loss.item()
        
            wandb.log({
                "batch/loss/": loss.item(),
                    })
            
            if batchcount == 0: # only for the first batch every epoch
                wandb_images = []
                for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                    combined_image_np = CombineImages(pred_single, label_single, rgb_single)
                    wandb_images.append(wandb.Image(combined_image_np))

                wandb.log(
                {
                    "images/val": wandb_images,
                })
            
    wandb.log({
        "epoch/loss/val": epochloss,
                })
                
base = "parameter"
n = 0
while True:
    _model_folder_name = f"{base}{n}"
    TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, _model_folder_name)
    if not os.path.exists(TRAINED_MDL_PATH):
        os.makedirs(TRAINED_MDL_PATH)
        break
    n += 1


if not os.path.exists(JOB_FOLDER):
    os.mkdir(JOB_FOLDER)


# DATASET ---------------------------------------------------------------------------
datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# Define the dataset size
dataset = WindowDataset(DS_PATH)
print("testing dataset:")

print("no modification")
# Split the dataset into train and validation
dataset_size = len(dataset)

train_size = int(0.75 * dataset_size)
test_size = dataset_size - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
trainLoader = torch.utils.data.DataLoader(trainset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
valLoader = torch.utils.data.DataLoader(valset, BATCH_SIZE, True, num_workers=NUM_WORKERS)

# Network and optimzer --------------------------------------------------------------
model = Network(3, 1)
# model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True))
model = model.to(device)

# LOSS FUNCTION AND OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def shouldLog(batchcount=None):
    if batchcount==None:
        return LOG_WANDB=='true'
    else:
        return batchcount%LOG_BATCH_INTERVAL == 0 and LOG_WANDB=='true'

# INIT LOGGER
wandb.init(
    project=MODEL_NAME,
    name=str(JOB_ID),
    
    # track hyperparameters and run metadata
    config={
    "JOB_ID":JOB_ID,
    "learning_rate": LR,
    "batchsize": BATCH_SIZE,
    "dataset": DS_PATH,
    }
)

# STORE ORIGINAL PARAMTERS
trainedMdlPath = TRAINED_MDL_PATH + f"/test.pth"
torch.save(model.state_dict(), trainedMdlPath)

# SCRIPT ---------------------------------------------------------------------------------
epochs = 100

lossFn = nn.BCEWithLogitsLoss()  #nn.CrossEntropyLoss(), but that did not seem to work much; nn.BCEWithLogitsLoss() is the one that worked best

for eIndex in range(epochs):
    dp(f"Epoch {eIndex+1}\n")

    print(" training:")
    train(trainLoader, model, lossFn, optimizer, eIndex)
    print(" validation:")
    val(valLoader, model, lossFn, eIndex)

    trainedMdlPath = TRAINED_MDL_PATH + f"/{eIndex}.pth"
    torch.save(model.state_dict(), trainedMdlPath)