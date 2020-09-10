import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['figure.figsize']=[10,4]
def moving_average(interval, window_size): 
    window = np.ones(int(window_size)) / float(window_size) 
    return np.convolve(interval, window, 'same')##卷积操作转为平滑操作
def draw_figs_3d(x,savename):
    if not os.path.exists(savename):
       os.mkdir(savename)
    x = x.cpu().detach().numpy()
    z1 = np.zeros((x.shape[3],x.shape[4]))
    for i in range(x.shape[1]):
      for j in range(x.shape[3]):
          #x[0,i,:,j,:][0] = moving_average(interval=x[0,i,:,j,:][0], window_size=10)
          for k in range(x.shape[4]):
               z1[j,k] = x[0,i,:,j,k]
      x1 = np.arange(0,x.shape[3],1)
      y1 = np.arange(0,x.shape[4],1)
      fig=plt.figure()
      ax=Axes3D(fig)
      x1,y1=np.meshgrid(x1,y1)
      #colors = plt.cm.jet(np.linspace(0,1,5))
      ax.plot_surface(x1,y1,z1,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
      ax.set_zlim(0,1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.savefig(savename+'channel_{}_.jpg'.format(i),dpi=600) 
      plt.close()
      z1 = np.zeros((x.shape[3],x.shape[4]))
class PrintBlock3D(nn.Module):   
    def __init__(self):
        super(PrintBlock3D, self).__init__()        
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,x,savename):
        #x.register_hook(print_grad)
        #grad = x
        return draw_figs_3d(x,savename)
def draw_figs(savename,b,a,idx):
  if not os.path.exists(savename):
    os.mkdir(savename)
  for k in range(len(a)): 
    fig = plt.figure()
    #ax = fig.add_axes([0.1, 0.1, 0.95, 0.95])
    n = np.arange(0,len(a[k]),1)
    colors = plt.cm.jet(np.linspace(0,1,len(a[k])))
    #a[k] = moving_average(interval = a[k], window_size = 10)
    #b[k] = moving_average(interval = b[k], window_size = 10)
    plt.xlim(0,len(a[k]))#+len(b[k])+3)
    plt.ylim(-0.3,1)
    #ax = plt.axis
    #ax.get_proj = lambda: np.dot(ax.get_proj(ax), np.diag([0.5, 1]))
    #my_x_ticks = np.arange(0, len(a[k])+1, 1)
    #my_y_ticks = np.arange(-0.5, 1, 0.5)
    #plt.xticks([])
    #plt.yticks([])
    plt.figure(figsize=(9, 4))
    plt.scatter(n,b[k],color = colors,marker = '.',linewidth=1.5)
    plt.plot(n,b[k],color = 'c',label='Before',linewidth=1.5)
    plt.scatter(n,a[k],color = colors,marker = '.',linewidth=1.5)
    plt.plot(n,a[k],color = 'm',label='After',linewidth=1.5)
    #plt.plot(n,a[k]-b[k],color = 'y',label='Difference',linewidth=1.5)
    plt.legend()
    plt.xlabel('channel(n)')
    plt.ylabel('weights')
    #plt.rcParams['figure.figsize']=[20,4]
    plt.savefig(savename+'{}_{}.jpg'.format(idx,k),dpi=600)  
    plt.close()
class PrintBlock(nn.Module):   
    def __init__(self):
        super(PrintBlock, self).__init__()        
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,savename,b,a,idx):
        #x.register_hook(print_grad)
        #grad = x
        return draw_figs(savename,b,a,idx)
def draw_figs_list_c(x,layer_id,idx):
    b = []
    x = x.cpu().detach().numpy()
    for j in range(x.shape[2]):
      #for i in range(x.shape[1]):
         img = list(x[0, :, j, :, :])
         for i in range(len(img)):
             img[i] = float(img[i])
         b.append(img)
    return np.array(b)
class GetBlock_c(nn.Module):
    def __init__(self):
        super(GetBlock_c, self).__init__()
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,x,layer_id,idx):
        #x.register_hook(print_grad)
        #grad = x
        return draw_figs_list_c(x,layer_id,idx)
def draw_figs_list_s(x,layer_id,idx):
    b = []
    x = x.cpu().detach().numpy()
    for j in range(x.shape[1]):
      #for i in range(x.shape[1]):
         img = list(x[0, j, :, :, :])
         for i in range(len(img)):
             img[i] = float(img[i])
         b.append(img)
    return np.array(b)
class GetBlock_s(nn.Module):
    def __init__(self):
        super(GetBlock_s, self).__init__()
        #self.x = x.cpu().detach().numpy()
        #self.savename = savename
    def forward(self,x,layer_id,idx):
        #x.register_hook(print_grad)
        #grad = x
        return draw_figs_list_s(x,layer_id,idx)
