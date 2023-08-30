import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
#import torchvision.transforms as T
from torchvision import models, transforms
import matplotlib.pyplot as plt
import shutil
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


from utils import CustomImageDataset_imagenette, train_model

print('Using device:', device)

model_path = 'resnet/model.pt'

dire = 'imagenette2'
train_labels = pd.read_csv('imagenette2/train_labels.csv')
#train_labels.set_index(train_labels.columns[0])

test_labels = pd.read_csv('imagenette2/test_labels.csv')
#test_labels.set_index(test_labels.columns[0])

train_images_dir = os.path.join(dire, 'train')
test_images_dir = os.path.join(dire, 'val')

le = preprocessing.LabelEncoder()
le.fit(train_labels.iloc[:,1])
print(le.classes_)
train_labels['labels_enc'] = le.transform(train_labels.iloc[:,1])

le = preprocessing.LabelEncoder()
le.fit(test_labels.iloc[:,1])
test_labels['labels_enc'] = le.transform(test_labels.iloc[:,1])
exit
# Resnet
model_ft = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
for param in model_ft.parameters():
    #param.requires_grad = False
    param.requires_grad = True

num_ftrs = model_ft.fc.in_features
print(num_ftrs)
print(model_ft)
model_ft.fc = torch.nn.Linear(num_ftrs,10)

model_ft = model_ft.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(232),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

training_data = CustomImageDataset_imagenette(train_labels, train_images_dir, transform)
test_data =  CustomImageDataset_imagenette(test_labels, test_images_dir, transform)
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

dataloaders = {'train': train_dataloader, 'val': test_dataloader}
dataset_sizes = {'train': len(training_data), 'val': len(test_data)}

model_ft = train_model(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler,model_path, 
                       num_epochs=20)
