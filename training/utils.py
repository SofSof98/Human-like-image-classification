import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
#import torchvision.transforms as T
from torchvision import models, transforms

import time
from tqdm import tqdm
import copy
import cv2
from sklearn import preprocessing
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision.models import resnet50, ResNet50_Weights,vit_b_16, ViT_B_16_Weights
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None, explain = False):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.explain = explain
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.explain:
          #print(self.img_labels.iloc[idx, 0])
          label = [label,  self.img_labels.iloc[idx, 0]]
        return image, label

class CustomImageDataset_imagenette(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None, explain = False):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.explain = explain
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        #print(image.shape)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.explain:
          #print(self.img_labels.iloc[idx, 0])
          label = [label,  self.img_labels.iloc[idx, 0]]
        return image, label


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, path,num_epochs=25, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
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
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if is_inception:
                        
                        if phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    #torch.save(model.state_dict(best_model_wts), path)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), path)
    return model


def load_model(model, ft=False):

  model_name = 'model_ft.pt' if ft else 'model_tl.pt'

  if model == 'resnet':
    model_path = os.path.join('Models/resnet', model_name)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,3)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  elif model == 'vit':
    model_path = os.path.join('Models/vit', model_name)
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_ftrs,3)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  elif model == 'inception':
    model_path = os.path.join('Models/inception', model_name)
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # the auxilary net, only primary net used at test time
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = torch.nn.Linear(num_ftrs, 3)
    # the primary net
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,3)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  else:

    raise ValueError('Model not available')

  return model
