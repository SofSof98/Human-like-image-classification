import os
import torch
from torchvision.models import resnet50, ResNet50_Weights,vit_b_16, ViT_B_16_Weights, inception_v3,Inception_V3_Weights, mobilenet_v2, MobileNet_V2_Weights, vgg16, VGG16_Weights
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import models, transforms
import numpy as np
from matplotlib import image

def load_model(model,path= 'Models', ft=False):

  model_name = 'model.pt'
  
  if model == 'resnet':
    model_path = os.path.join(path+'/resnet', model_name)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,10)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
  
  elif model == 'vit':
    model_path = os.path.join(path+'/vit', model_name)
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_ftrs,10)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  elif model == 'inception':
    model_path = os.path.join(path+'/inception', model_name)
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # the auxilary net, only primary net used at test time
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = torch.nn.Linear(num_ftrs, 10)  
    # the primary net
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,10)   
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  elif model == 'mobilenet':
    model_path = os.path.join(path+'/mobilenet', model_name)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs,10)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))

  elif model == 'vgg':
    model_path = os.path.join(path+'/vgg', model_name)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs,10)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
    
  else:

    raise ValueError('Model not available')
  
  return model

def load_txt(filename):
    content = {}
    with open(filename)as f:
        for line in f:
            content[line.strip().split()[1]] = line.strip().split()[0]
    return content
    
def flatten_comprehension(matrix):
   return list(dict.fromkeys([item for row in matrix for item in row]))
   
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

        cl = self.img_labels.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, cl, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.explain:
          #print(self.img_labels.iloc[idx, 0])
          label = [label,  os.path.join(cl,self.img_labels.iloc[idx, 0])]
        return image, label
    

# get transform

def get_transform(model):
  
  if model == 'resnet':

    transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(232),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
   
  elif model == 'vit':

    transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(242),
                         transforms.CenterCrop(224), 
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
  elif model == 'inception':

    transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(342),
                         transforms.CenterCrop(299),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  elif model == 'mobilenet':

    transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(232),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


  elif model == 'vgg':

    transform =  transforms.Compose([ transforms.ToPILImage(),
                        transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  else:

    raise ValueError('Model not available')
  
  return transform

# get target layer for each model

def get_target_layer(model,resnet, vit, inception, mobilenet, vgg):

  if model == 'resnet':
    target_layer = resnet.layer4[2].bn3

  elif model == 'vit':
    target_layer =  vit.encoder.layers.encoder_layer_11.ln_1
    # target_layer =  vit.encoder.layers.encoder_layer_11.ln_2
    # target_layer =  vit.encoder.ln
    

  elif  model == 'inception':
    target_layer = inception.Mixed_7c.branch_pool.bn


  elif model == 'mobilenet':
    target_layer = mobilenet.features[18][1]
    
  elif model == 'vgg':
    target_layer = vgg.features[28]
    
  else:
    raise ValueError('Model not available')

  return target_layer
# torch.Size([1, 197, 768])
# In ViT the output of the layers are typically
# the first element represents the class token,
# and the rest represent the 14x14 patches in the image

# Since the final classification is done on the class token computed in the last
# attention block, the output will not be affected by the 14x14 channels in the 
# last layer. The gradient of the output with respect to them, will be 0!
# We should chose any layer before the final attention block

# get model 

def get_model(model, resnet, vit, inception, mobilenet, vgg):


  if model == 'resnet':
    model = resnet

  elif model == 'vit':
    model = vit

  elif  model == 'inception':
    model = inception

  elif model == 'mobilenet':
    model = mobilenet
    
  elif model == 'vgg':
    model = vgg
    
  else:
    raise ValueError('Model not available')

  return model

def get_rgb_png(image): # for png images
  size = image.size
  rgbs = list()
  rgb_im = image.convert('RGB')
  for i in range(0,size[0]):
    for j in range(0, size[1]):
      coordinate = i,j
      rgbs.append(rgb_im.getpixel(coordinate))
  return rgbs


def euclidean(A,B):
  d = list()
  for i in range(len(A)):
    d.append(np.sqrt((A[i][0] - B[i][0])**2 + (A[i][1] - B[i][1])**2+ (A[i][2] - B[i][2])**2))
  return sum(d)/len(A)
  
def IoU_coeff(y_pred, y_true, flatten=False):
    smooth = 0.000001

    if flatten:
      y_pred = y_pred.flatten()
      y_true = y_true.flatten()
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
                
    return  IoU
    
def Intersection(y_pred, y_true, flatten=False):

    if flatten:
      y_pred = y_pred.flatten()
      y_true = y_true.flatten()
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (y_pred * y_true).sum()
    total = y_true.sum()
        
    intersection = (intersection)/(total)
                
    return  intersection


def not_intersection(y_pred, y_true, flatten=False):

    if flatten:
      y_pred = y_pred.flatten()
      y_true = y_true.flatten()
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (y_pred * y_true).sum()
    not_inter = y_pred.sum() - intersection
    total = y_pred.shape[0]*y_pred.shape[1]
        
    intersection = (not_inter)/(total)
                
    return  intersection

def F_score(y_pred, y_true, beta, flatten=False):

    if flatten:
      y_pred = y_pred.flatten()
      y_true = y_true.flatten()

    y_false = np.zeros_like(y_pred)
    y_false[y_true==0] = 1
    y_pred_neg = np.zeros_like(y_pred)
    y_pred_neg[y_pred==0] = 1

    # True Positive
    TP = (y_pred * y_true).sum()
    # False Negative
    FN = (y_pred_neg*y_true).sum()
    # False Positive
    FP = (y_pred*y_pred_neg).sum()

    score = ((1+beta**2)*TP)/((1+beta**2)*TP+(beta**2)*FN+FP)

    return  score

def Tversky_coeff(y_pred, y_true, beta, flatten=False):
    alpha = 1 - beta
    smooth = 0.000001
    if flatten:
      y_pred = y_pred.flatten()
      y_true = y_true.flatten()

    #True Positives, False Positives & False Negatives
    TP = (y_pred * y_true).sum()
    FP = ((1-y_true) * y_pred).sum()
    FN = (y_true * (1-y_pred)).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return Tversky

def compute_metrics(segmented, grad_img, threshold, IoU_list, Precision_list, f2score_list, f1score_list, f0score_list, T1score_list, T2score_list, T3score_list):
        # Get segmentation map
        A = image.imread(segmented)
        if len(A.shape) > 2:
          A = A[:, :, 0]
        A_t = np.zeros_like(A)
        A_t[A >= 0.5] = 1 # Maybe there are some grey pixel
        # Get grayscale heatmap
        B = image.imread(grad_img)/255.
        B_t = np.zeros_like(B)
        B_t[B >= threshold] = 1.0
        iou = IoU_coeff(B_t, A_t)
        i_only = not_intersection(B_t,A_t)
        f2score = F_score(B_t,A_t,2)
        f1score = F_score(B_t,A_t,1)
        f0score = F_score(B_t,A_t,0.5)
        T1 = Tversky_coeff(B_t, A_t, 0.2)
        T2 = Tversky_coeff(B_t, A_t, 0.5)
        T3 = Tversky_coeff(B_t, A_t, 0.8)
        IoU_list.append([grad_img,iou])
        Precision_list.append([grad_img,i_only])
        f2score_list.append([grad_img,f2score])
        f1score_list.append([grad_img,f1score])
        f0score_list.append([grad_img,f0score])
        T1score_list.append([grad_img,T1])
        T2score_list.append([grad_img,T2])
        T3score_list.append([grad_img,T3])

def average_classes(IoU_list, Precision_list, F2_list, F1_list, F0_list, T1_list, T2_list, T3_list,
                    IoU_avg, Precision_avg, F2_avg, F1_avg, F0_avg, T1_avg, T2_avg, T3_avg):
    for t in IoU_list:
      values = [x[1] for x in IoU_list[t]]
      IoU_avg.append(np.mean(values))
      values = [x[1] for x in Precision_list[t]]
      Precision_avg.append(np.mean(values))

      values = [x[1] for x in F2_list[t]]
      F2_avg.append(np.mean(values))
      values = [x[1] for x in F1_list[t]]
      F1_avg.append(np.mean(values))
      values = [x[1] for x in F0_list[t]]
      F0_avg.append(np.mean(values))

      values = [x[1] for x in T1_list[t]]
      T1_avg.append(np.mean(values))
      values = [x[1] for x in T2_list[t]]
      T2_avg.append(np.mean(values))
      values = [x[1] for x in T3_list[t]]
      T3_avg.append(np.mean(values))
