import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def draw_features(x,savename):##(1,512,7,7)
    x = x.cpu().detach().numpy()
    if not os.path.exists(savename):
       os.mkdir(savename)
    for j in range(x.shape[1]):
      #for i in range(x.shape[1]):
        img = x[0,j, :, :]
        img = cv2.resize(img,(224,224))
        plt.imshow(img,cmap='jet')
        plt.imsave(savename+'tongdao{}.jpg'.format(j), img, dpi=600,cmap='jet')#规定像素
    plt.close()
class VisBlock(nn.Module):   
    def __init__(self):
        super(VisBlock, self).__init__()        
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,x,savename):
        return draw_features(x,savename)
##usage
##from .draw import VisBlock
##in init          self.vis = VisBlock()
##in forward       self.vis(x,"{}/conv1/".format(savepath))#savename can be changed
