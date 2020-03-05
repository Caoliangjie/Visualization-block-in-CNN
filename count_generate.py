#import matplotlib.pyplot as plt
import json
baseline = json.load(open('results_val_baseline.json','r'))
channel = json.load(open('results_val_mgma.json','r'))
labels = json.load(open('something_v1_01.json','r'))
baseline_save = open('baseline_f.txt','w')
ground_truth = open('grundtruth_f.txt','w')
channel_save = open('mgma_f.txt','w')
#import numpy as np
count = 0
count_channel = 0
gd_count = 0
label_dict = {}
channel_label_dict = {}
ground_truth_dict = {}
label_idx = []
baseline_result = []
channel_result = []
ground_truth_result = []
for k in labels['labels']:
     label_dict.update({k:0})
     label_idx.append(k)
#print(label_dict)
#print(label_idx)
#for k in labels['labels']:
 #   channel_label_dict.update({k:0})
    #label_idx.append(k)
#for k in labels['labels']:
 #   ground_truth_dict.update({k:0})
#print(label_idx)
ground_truth_dict = {}
channel_label_dict = {}#label_dict
for k in labels['labels']:
    channel_label_dict.update({k:0})
for k in labels['labels']:
    ground_truth_dict.update({k:0})
#print(channel_label_dict)
for k1,v1 in baseline.items():
    #for k,v in labels['database'].items():
        #if k1==k:
            #print('1',labels['database'][k1])
            #print('2',label_idx[int(v1['target_label'])])
            index = v1['pred_outputs'].index(max(v1['pred_outputs']))
            if label_idx[int(v1['target_label'])]==label_idx[index]:#v1['pred_outputs']:#labels['database'][k1]['annotations']['label']:
                k = labels['database'][k1]['annotations']['label']#v['annotations']['label']
                label_dict.update({k:label_dict[k]+1})
                #print({k:label_dict[k]+1})
                count += 1
for k,v in label_dict.items():
    baseline_save.write(k+' '+str(v)+'\n')
for k1,v1 in channel.items():
    #for k,v in labels['database'].items():
        #print(k,k1)
        #if k1==k:
            #gr_key = v['annotations']['label']
            #ground_truth_dict.update({gr_key:ground_truth_dict[gr_key]+1})
            index = v1['pred_outputs'].index(max(v1['pred_outputs']))
            if label_idx[int(v1['target_label'])]==label_idx[index]:#labels['database'][k1]['annotations']['label']:
                k = labels['database'][k1]['annotations']['label']#v['annotations']['label']
                channel_label_dict.update({k:channel_label_dict[k]+1})
                #print({k:label_dict[k]+1})
                count_channel += 1
for k,v in channel_label_dict.items():
    channel_save.write(k+' '+str(v)+'\n')
for k1,v1 in channel.items():
  #for k,v in labels['database'].items():
        #if len(v)>1:
            #print('v = ',v)
            gr_key = labels['database'][k1]['annotations']['label']
            ground_truth_dict.update({gr_key:ground_truth_dict[gr_key]+1})
            gd_count+=1
for k,v in ground_truth_dict.items():
    ground_truth.write(k+' '+str(v)+'\n')
print('total baseline = ',count)
print('total channel = ',count_channel)
print('total gd = ',gd_count)
print(len(baseline))