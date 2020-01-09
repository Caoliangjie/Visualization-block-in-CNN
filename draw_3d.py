import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
grad_list = []
def draw_features(x,savename):
    x = x.cpu().detach().numpy()
    if not os.path.exists(savename):
       os.mkdir(savename)
    #if not os.path.exists(savename+'sum'):
     #  os.mkdir(savename+'sum')
    for j in range(x.shape[2]):
      for i in range(x.shape[1]):
        #feature_map_combination = []
        #if not os.path.exists(savename+'tongdao{}'.format(i)):
         #    os.mkdir(savename+'tongdao{}'.format(i))
        img = x[0, i, j, :, :]
        img = cv2.resize(img,(224,224))
        #plt.imshow(img,cmap='jet')
        plt.imsave(savename+'tongdao{}_clip{}.jpg'.format(i,j), img, dpi=600,cmap='jet')#规定像素
        #feature_map_combination.append(img)
      #feature_map_sum = sum(ele for ele in feature_map_combination)
      #plt.imsave(savename+'sum/clip{}.jpg'.format(j), feature_map_sum, dpi=600,cmap='jet')
    plt.close()
def draw_features_grad(x,savename,grad):##打出特征  
    #image = np.ones([224,224,3])*255
    x = x.cpu().detach().numpy()
    grad = grad.cpu().detach().numpy()
    v = grad
    if not os.path.exists(savename):
        os.mkdir(savename)
    for j in range(x.shape[2]):
        grads = v[:,:,j,:,:]
        #print('val',grads.shape)
        img = x[0, : , j, :, :]##这里写j才是真正的能看到时序上的一些信息。
        weights = np.mean(grads, axis = (2, 3))[0,:]##平均池化
        #print('weights',weights.shape)
        cam = np.zeros(img[0, :, :].shape, dtype = np.float32)
        #print('cam',cam.shape)
        #print('img',img.shape)
        for i, w in enumerate(weights):
            if w<0:##已经把负的点约去了
               w=0
            cam += w * img[i, :, :]
        cam = np.maximum(cam, 0)##relu这里就是relu
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        #print('camshape',cam.shape)
        #show_cam_on_image(image,cam)
        plt.imsave(savename+'clip{}_tongdao_zong.jpg'.format(j), cam, dpi=600,cmap='jet')#规定像素
    plt.close()
def print_grad(grad):
    grad_list.append(grad)
def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
class VisBlock(nn.Module):   
    def __init__(self):
        super(VisBlock, self).__init__()        
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,x,savename):
        #x.register_hook(print_grad)
        #grad = x
        return draw_features(x,savename)
class VisBlock_cam(nn.Module):   
    def __init__(self):
        super(VisBlock_cam, self).__init__()
    def forward(self,x,savename):
        x.register_hook(print_grad)
        grad = x
        return draw_features_grad(x,savename,grad)
##usage
##from .draw import VisBlock
##in init          self.vis = VisBlock() or self.vis = VisBlock_cam()
##in forward       self.vis(x,"{}/conv1/".format(savepath))#savename can be changed
##attention: VisBlock_grad() only can be used in train mode
