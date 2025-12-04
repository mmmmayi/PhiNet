import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire, torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.utils import parse_config_or_kwargs
from wespeaker.models.weights import get_weight
from wespeaker.utils.checkpoint import load_checkpoint

def split_sort_score(score_dir):
    with open(score_dir,'r') as file:
        lines = file.readlines()
    target_rows, non_target_rows = [],[]
    for line in lines:
        utt1,utt2,score,label = line.strip().split(' ')
        if label=='target':
            target_rows.append([utt1,utt2,float(score),label])
        else:
            non_target_rows.append([utt1,utt2,float(score),label])
    target_rows = sorted(target_rows,key=lambda x:x[2],reverse=True)
    non_target_rows = sorted(non_target_rows,key=lambda x:x[2],reverse=True)
    return target_rows,non_target_rows

def print_similarity(model, dur,
              eval_scp_path='',
              score_dir='',
              trial=()):
    store_path = os.path.join(score_dir,os.path.basename(trial[0]) +str(dur)+'.pho_score')
    weight_path = kaldiio.load_scp(os.path.join(score_dir,os.path.basename(trial[0]) +str(dur)+ '.weights.scp'))
    similarity_path = kaldiio.load_scp(os.path.join(score_dir,os.path.basename(trial[0])  +str(dur)+ '.similarity.scp'))
    target_rows, non_target_rows = split_sort_score(store_path)
    
    target_rows = target_rows[5000:5010]
    non_target_rows = non_target_rows[5000:5010]
    for line in target_rows:
        utt1 = line[0]
        utt2 = line[1]
        score = line[2]
        same_similarity = similarity_path[utt1+utt2+'same']
        same_weight = weight_path[utt1+utt2+'same']
        same_score = torch.from_numpy(same_weight*same_similarity)
        same_index = np.argsort(same_score.numpy())
        if 'separate' in  score_dir:
            cross_similarity = similarity_path[utt1+utt2+'cross']
            cross_weight = weight_path[utt1+utt2+'cross']
            cross_score = F.tanh(torch.from_numpy(cross_weight*cross_similarity))

            cross_index = np.argsort(cross_score.numpy())
        analysis_file = os.path.join(score_dir, 'Target_'+utt1.strip('.wav').replace('/','_')+'_'+utt2.strip('.wav').replace('/','_'))
        with open(analysis_file, 'w') as w_f:
            w_f.write('score: {}\n'.format(score))
            w_f.write('same phoneme score:'+'\n')
            w_f.write(str(same_score[same_index].tolist())+'\n')
            
            w_f.write('same phoneme similarity:'+'\n')
            w_f.write(str(same_similarity[same_index].tolist())+'\n')
            
            w_f.write('same phoneme weight:'+'\n')
            w_f.write(str(same_weight[same_index].tolist())+'\n')
            w_f.write('phoneme index:'+'\n')
            w_f.write(str(same_index)+'\n')
            
            if 'separate' in  score_dir:
                w_f.write('cross phoneme score:'+'\n')
                w_f.write(str(cross_score[cross_index].tolist())+'\n')
                w_f.write('cross phoneme similarity:'+'\n')
                w_f.write(str(cross_similarity[cross_index].tolist())+'\n')
                w_f.write('cross phoneme weight:'+'\n')
                w_f.write(str(cross_weight[cross_index].tolist())+'\n')        


def plot(paras,score_dir):
    line_idx = 0
    dict_raw = dict()

    with open('/home/svu/e0643891/test/wespeaker/examples/voxceleb/v2/pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1
    paras = (paras-torch.amin(paras))/(torch.amax(paras)-torch.amin(paras))
    sorted_tensor, sorted_indices = torch.sort(paras,descending=True)
    sorted_names = [list(dict_raw.keys())[i] for i in sorted_indices]

    plt.figure(figsize=(18, 4))
    bars = plt.bar(sorted_names, sorted_tensor.numpy())
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom',fontsize=10)
    
#    plt.xticks(rotation=90)  
    plt.xlabel('Phone')
    plt.ylabel('Weight')
    plt.xlim(-0.5, len(sorted_names) - 0.5)
    #plt.grid(True)
    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(score_dir, 'phone_weight'))

def main(exp_dir,
         dur,
         eval_scp_path,
         model_path,
         configs,
         *trials):

    configs = parse_config_or_kwargs(configs)
    
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    '''
    weight = get_weight(configs['weight_args'])
    model.add_module("weight", weight)
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()

    store_score_dir = os.path.join(exp_dir, 'scores')
    print_similarity(model,dur,eval_scp_path, store_score_dir,trials)
    '''
    store_score_dir = os.path.join(exp_dir, 'scores')
    para = torch.load(os.path.join(exp_dir,'models','weight.pt'))
    plot(para.squeeze(),store_score_dir)
    
    
if __name__ == "__main__":
    fire.Fire(main)