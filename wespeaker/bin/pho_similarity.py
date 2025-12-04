import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire, torch
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
    
def trials_cosine_score(eval_scp_path='',
                        score_dir='',
                        trials=(),
                        num=10):
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb_dict[utt] = np.transpose(emb)
    '''
    for trial in trials:
        store_path = os.path.join(score_dir,
                                  os.path.basename(trial) + '.score')
        target_rows, non_target_rows = split_sort_score(store_path)
        true_target = target_rows[:num]
        false_target = target_rows[-num:]
        true_nontarget = non_target_rows[-num:]
        false_nontarget = non_target_rows[:num]
        
        for i in true_target:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            
            #print(cosine_similarity)
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'true_target')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'true_target',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
        for i in false_target:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'false_target')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'false_target',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
        for i in true_nontarget:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'true_nontarget')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'true_nontarget',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
        for i in false_nontarget:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'false_nontarget')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'false_nontarget',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
    '''
    for trial in trials:
        store_path = os.path.join(score_dir, os.path.basename(trial) + '.pho_score')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            lines = trial_r.readlines()
            for line in tqdm(lines,desc='scoring trial {}'.format(os.path.basename(trial))):
                segs = line.strip().split()
                emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                cosine_similarity = np.matmul(emb1,emb2.T)
                #cosine_score = cosine_similarity(emb1,emb2)
                #print(cosine_score.shape)
                
                #idx = np.count_nonzero(cosine_score)
                #print(idx)
                #quit()
                idx_diag = len(np.where(np.diagonal(cosine_similarity)!=0)[0])
                score = np.sum(np.diagonal(cosine_similarity))/idx_diag
                #idx_full = len(np.where(cosine_similarity!=0)[0])
                #score = np.sum(cosine_similarity)/idx_full
                #score = (score_full-score_diag)/(idx_full-idx_diag)
             

                w_f.write('{} {} {:.5f} {}\n'.format(segs[0], segs[1], score, segs[2]))
                
def pho_score(dur,model,
              eval_scp_path='',
              score_dir='',
              trials=(),
              dele=None):
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb_dict[utt] = emb
    for trial in trials:
        store_path = os.path.join(score_dir, os.path.basename(trial) + str(dur)+'.pho_score')
        weight_ark = os.path.join(score_dir, os.path.basename(trial) + str(dur)+'.weights.ark')
        weight_scp = os.path.join(score_dir, os.path.basename(trial) + str(dur)+'.weights.scp')
        similarity_ark = os.path.join(score_dir, os.path.basename(trial) + str(dur)+'.similarity.ark')
        similarity_scp = os.path.join(score_dir, os.path.basename(trial) + str(dur)+'.similarity.scp')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            with kaldiio.WriteHelper('ark,scp:' + weight_ark + "," +weight_scp) as weight_writer, kaldiio.WriteHelper('ark,scp:' + similarity_ark + "," +similarity_scp) as similarity_writer:
                lines = trial_r.readlines()
                for line in tqdm(lines,desc='scoring trial {}'.format(os.path.basename(trial))):
                    segs = line.strip().split()
                    emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                    score,weights,similarity=model.weight(torch.from_numpy(emb1).unsqueeze(0).cuda(),torch.from_numpy(emb2).unsqueeze(0).cuda(),True,dele)
                    score=score.squeeze()
                    w_f.write('{} {} {:.5f} {}\n'.format(segs[0], segs[1], score, segs[2]))
                    key = segs[0]+segs[1]
                    '''
                    if len(weights)==2:
                        weight_writer(key+'same',weights[0])
                        weight_writer(key+'cross',weights[1]) 
                        similarity_writer(key+'same',similarity[0]) 
                        similarity_writer(key+'cross',similarity[1])
                    else:
                        weight_writer(key+'same',weights)
                        similarity_writer(key+'same',similarity) 
                    '''
def main(dur,
         exp_dir,
         eval_scp_path,
         model_path,
         configs,
         dele,
         *trials):

    configs = parse_config_or_kwargs(configs)
    model = get_speaker_model(configs['model'])(**configs['model_args'])

    weight = get_weight(configs['weight_args'])
    model.add_module("weight", weight)
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()

    store_score_dir = os.path.join(exp_dir, 'scores')
    #trials_cosine_score(eval_scp_path, store_score_dir, trials)
    pho_score(dur,model,eval_scp_path, store_score_dir, trials, dele)
    
if __name__ == "__main__":
    fire.Fire(main)

