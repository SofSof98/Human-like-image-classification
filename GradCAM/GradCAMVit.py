import torch
import cv2
import os
import numpy as np

# GradCAMViT
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def GradCAMViT(model, img_path,output_dir):
    img_path = os.path.join('Images', img_path)
    target_layers = [model.encoder.layers.encoder_layer_11.ln_1]
    
    cam = GradCAM(model=model,target_layers=target_layers, reshape_transform=reshape_transform_vit)
    
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None
    
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)
    
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    img = cv2.imread(img_path)
    res = cv2.resize(grayscale_cam, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = res.squeeze()
    heatmap = np.uint8(255 * heatmap)
    cv2.imwrite(output_dir, heatmap)
    