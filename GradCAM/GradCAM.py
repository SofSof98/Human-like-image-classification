import torch
import cv2
import os
import numpy as np

# GradCAM

class Model_gradCAM2(object):
  def __init__(self, model, target_layer, is_resnet= False, is_vit = False):
    super(Model_gradCAM2, self).__init__()

    self.model = model
    self.target_layer = target_layer
    self.is_resnet = is_resnet
    self.is_vit = is_vit
    
    # relu 
    self.relu = torch.nn.ReLU(inplace=False)
    #placeholder for activations and gradients
    self.gradients = None
    self.activations = None
    self.model.eval()
    self.layer_activations = self.target_layer.register_forward_hook(self.layer_activations_hook)
    self.layer_gradients = self.target_layer.register_forward_hook(self.layer_gradients_hook)

  def layer_activations_hook(self, module, input, output):
    if self.is_vit:
      output = self.reshape_transform(output)
    self.activations = output
    
  def _store_grad(self, grad):
    
    if self.is_vit:
  
      grad = self.reshape_transform(grad)
    #self.gradients = [grad.cpu().detach()] #+ self.gradients
    self.gradients = grad #+ self.gradients

  def layer_gradients_hook(self, module, input, output):


  # Gradients are computed in reverse order
    #print(output.requires_grad)
    #print(hasattr(output, "requires_grad"))
    #self.g = output.register_hook(self._store_grad)
    self.g = output.register_hook(self._store_grad)

  #def layer_gradients_hook(self, module, grad_input, grad_output):
    #self.gradients = grad_output[0]

  def reshape_transform(self, tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

  def getGradCAM(self, pred, label):

    self.model.zero_grad()
    
    pred[:,label.item()].backward(retain_graph=True)

    activations = self.activations
    gradients = self.gradients
    if self.is_resnet:
      activations = self.relu(activations)

    #print(activations.shape)
    #print(gradients.shape)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    activations = activations.detach() * pooled_gradients

    cam = torch.sum(activations, dim=1, keepdim=True)
    cam = self.relu(cam)
    # min-max normalization
    #cam -= torch.min(cam) # min is zero
    cam /= torch.max(cam)

    return cam

  
  def forward(self, x, label):

    """
        Args:
            x: input image. shape =>(1, 3, H, W)
            label: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """
  

    pred = self.model(x)
    pred_label = torch.argmax(pred)
    # print(pred_label)
    # print(label)
    if pred_label == label:

      cam = self.getGradCAM(pred, label)
      return cam.cpu()
    else: 
      print('Wrong class predicted')

  def remove(self):
    self.layer_activations.remove()
    self.layer_gradients.remove()


  def __call__(self, x, label):
    return self.forward(x, label)

# get full resolution CAM

def get_full_cam(img_name, cam, dire):

  imge = cv2.imread(os.path.join('Images', img_name))
  heatmap = torch.nn.functional.interpolate(cam, size=(imge.shape[0], imge.shape[1]), mode='bilinear', align_corners=False) 
  heatmap = heatmap.squeeze()
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  #print(heatmap.shape)
  #superimposed_img = heatmap * 0.6 + imge
  #cv2.imwrite('map1.jpg', superimposed_img)
  cv2.imwrite(os.path.join(dire, img_name), heatmap)
 
def get_full_cam2(img_name, cam, dire):

  imge = cv2.imread(os.path.join('Images', img_name))
  heatmap = torch.nn.functional.interpolate(cam, size=(imge.shape[0], imge.shape[1]), mode='bilinear', align_corners=False) 
  heatmap = heatmap.squeeze()
  heatmap = np.uint8(255 * heatmap)
  #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  #print(heatmap.shape)
  #superimposed_img = heatmap * 0.6 + imge
  #cv2.imwrite('map1.jpg', superimposed_img)
  cv2.imwrite(os.path.join(dire, img_name), heatmap)