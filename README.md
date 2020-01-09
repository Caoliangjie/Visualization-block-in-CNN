Visualization-block-in-CNN
==========================

This is a simple work for building a block by using PyTorch, which is used for
Visualization features.

## How to use it
For example:
···
import the blocks you want to use: from draw2d import VisBlock
insert into init:        self.vis = VisBlock() or self.vis = VisBlock_cam()
in forward process:      self.vis(x,"{}/conv1/".format(savepath))#savename can be changed
attention: VisBlock_grad() only can be used in train mode or if you have save gradients in your model in test mode. 
···
I also add some description in the code. When we use draw2d.py, we can insert this block to 2D CNN architecture such as image classification work.
I add gradcam method as another way to visualize the features that can get a clear sight of the results no using for choosing a lot of pictures.
