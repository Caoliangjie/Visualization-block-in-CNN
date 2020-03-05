import matplotlib.pyplot as plt
import json
import re
import random
from pylab import *
labels = json.load(open('something_v1_01.json','r'))
baseline_save = open('baseline_f.txt','r')
ground_truth = open('grundtruth_f.txt','r')
channel_save = open('mgma_f.txt','r')
#sorted_baseline = open('sorted_baseline.txt','w')
diff_dict = open('improve_ctsr.txt','w')
#sorted_sr=open('sorted_ctsr.txt','w')
import numpy as np
count = 0
count_channel = 0
label_dict = {}
channel_label_dict = {}
ground_truth_dict = {}
label_idx = []
baseline_result = []
channel_result = []
ground_truth_result = []
name_list = []
for k in labels['labels']:
     label_dict.update({k:0})
     label_idx.append(k)
     channel_label_dict.update({k:0})
     ground_truth_dict.update({k:0})
#print(label_idx)
line = baseline_save.readline()
line1 = channel_save.readline()
line2 = ground_truth.readline()
while line:
    l = re.sub('[1234567890\n]', '', line)[:-1]
    label_dict.update({l:float(line.strip().split(' ')[-1])})
    line = baseline_save.readline()
#label_dict = sorted(label_dict.items(),key=lambda item:item[1], reverse=True)
#for k in label_dict:
#    sorted_baseline.write(str(k)+'\n')
print(label_dict)
while line1:
    l = re.sub('[1234567890\n]', '', line1)[:-1]
    channel_label_dict.update({l:float(line1.strip().split(' ')[-1])})
    line1 = channel_save.readline()
while line2:
    l = re.sub('[1234567890\n]', '', line2)[:-1]
    ground_truth_dict.update({l:float(line2.strip().split(' ')[-1])})
    line2 = ground_truth.readline()
#channel_label_dict = sorted(channel_label_dict.items(),key=lambda item:item[1], reverse=True)
#for k in channel_label_dict:
#    sorted_sr.write(str(k)+'\n')
improve = {}
for k,v in channel_label_dict.items():
    if v-label_dict[k]>=0:
      v = float('%.2f'% ((v-label_dict[k])/ground_truth_dict[k]*100))
      improve.update({k:v})
improve = sorted(improve.items(),key=lambda item:item[1], reverse=True)
print(improve)
for i in improve:
    diff_dict.write(str(i)+'\n')
diff = []
for i,(k,v) in enumerate(improve):
    if i<20:
        print(v)
        name_list.append(k.replace('omething','th.'))
        baseline_result.append(label_dict[k])
        channel_result.append(channel_label_dict[k])
        diff.append(v)
print(name_list)
#channel_label_dict=sorted(channel_label_dict.items(),key=lambda item:item[1], reverse=True)
'''while line2:
    ground_truth_result.append(int(line2.strip().split(' ')[-1]))
    line2 = ground_truth.readline()'''
x =list(range(len(baseline_result)))
ax = plt.gca()
total_width, n = 0.8,2
width = total_width / n
for i in range(len(x)):
    x[i] = x[i] + width
print(x)
plt.figure(figsize=(15*1.2, 9*1.2))
plt.xlim(0, 20)
plt.ylim(0, 25.0001)
my_x_ticks = np.arange(0,20,1)
#plt.xticks(my_x_ticks,fontsize=8, rotation=0)
#plt.xticks(x+width/2, label=name_list)
#plt.xticks(x+width/2,name_list)
my_y_ticks = np.arange(0,25.0001,5)
plt.yticks(my_y_ticks,fontsize=18)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 25,
}
#plt.xlabel('category(n)',font2)
plt.ylabel('Boost',font1,rotation=90, horizontalalignment="right")
tick_params(top='off',bottom='off',left='on',right='off')
plt.bar(x,diff,width = 2*width,label = 'difference',fc = '#f7a35c')
for a,b in zip(x,diff):
    plt.text(a+0.05,float(b)+0.05,'%.2f'% float(b),ha = 'center',va = 'bottom',fontsize=18)
plt.text(-0.6,24.8,'(%)',horizontalalignment="right",fontsize=18)
red = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.xticks(my_x_ticks+width,red,ha='center',fontsize=18)
plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.05)
plt.savefig('boost_analysis.png',dpi=600)
for k in name_list:
    print(k)
plt.show()
