import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import librosa.display
import torch,torchaudio,os
from torch.nn.functional import conv1d, pad

def load_file(size,pho_path,dict):
    weight = _phn2vec(size[1], pho_path, dict)
    
    label = torch.from_numpy(weight).clone().unsqueeze(0).unsqueeze(0)

    return label 
def _phn2vec(wav_length, phn_path, phn_dict):
    #phone_time_seq = np.zeros(wav_length)
    if not os.path.exists(phn_path):
        print(phn_path)
        return np.ones(wav_length)
    phone_seq = []
    phone_weight = np.zeros(wav_length)
    with open(str(phn_path)) as f:
        for line in f:
            sample_start, sample_end, label_char = line.split(' ')
            sample_start = int(float(sample_start)*16000)
            sample_end = int(float(sample_end)*16000)
            if sample_start>wav_length:
                break
            sample_end = wav_length if sample_end>wav_length else sample_end
            label_char = label_char.strip()
            if label_char=='[UNK]':
                label_char = '[SIL]'

            phone_weight[sample_start: sample_end] = phn_dict[label_char]
            
    return phone_weight
def extract_phoneme(wav_path):
    line_idx = 0
    dict_raw = dict()
    with open('pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1
    pho_path=wav_path.replace('.wav','.pre.pho')
    spec,change,value=data(wav_path,pho_path,dict_raw)

    fig,ax=plt.subplots(nrows=1,ncols=1)
    librosa.display.specshow(spec.cpu().squeeze().numpy(),sr=16000)
    for i in range(len(change)):
        if i==0:
            location=change[i].item()/2
        else:
            location=(change[i].item()-change[i-1].item())/2+change[i-1].item()

        ax.axvline(x=change[i].item(),color='r',linestyle='--')
        #ax[0].text(location,ax[0].get_ylim()[1]*1.01,str(int(value[i].item())),fontsize=7,horizontalalignment='center')

    plt.tight_layout()
    save_path = '/hpctmp/ma_yi/dataset_vox1/voxceleb1/pic'
    pairs = wav_path.split('/')
    if not os.path.exists(os.path.join(save_path,pairs[-3],pairs[-2])):
        os.makedirs(os.path.join(save_path,pairs[-3],pairs[-2]))
    plt.savefig(os.path.join(save_path,pairs[-3],pairs[-2],pairs[-1].replace('.wav','.png')))
    plt.close()
    start = 0
    for i in range(len(change)):
        phoneme = value[i]
        end = change[i]
        fig,ax=plt.subplots(nrows=1,ncols=1)
        if end==start+1 or end==start:
            continue

        librosa.display.specshow(spec[:,start:end].cpu().squeeze().numpy(),sr=16000, cmap='coolwarm')
        name = pairs[-1].strip('.wav')+'_'+str(phoneme.item())+'_'+str(end.item())+'.png'
        plt.savefig(os.path.join(save_path,pairs[-3],pairs[-2],name))
        plt.close()
        start = end



def data(path,pho_path,dict_raw):
    waveform, sample_rate = torchaudio.load(path)
    
    pho = load_file(waveform.shape,pho_path,dict_raw)
    win_eye = torch.eye(400).unsqueeze(1)
    pho = conv1d(pho.float(), win_eye, stride=160).squeeze()
    pho, _ = torch.mode(pho, dim=0)
    change = torch.diff(pho)
    change = (change!=0).nonzero(as_tuple=True)[0]
    value=pho[change-1]

    Spec = torchaudio.transforms.Spectrogram(n_fft=400, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2.0, center=False)
    Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,400//2+1)
    spec = Mel_scale((Spec(waveform)+1e-8)+1e-8).log().squeeze()
    return spec,change,value

if __name__ == "__main__":
    file='Target_id10309_XMLMvfrgdzY_00003_id10309_e-IdJ8a4gy4_00001'.split('_')
    enrol=os.path.join(file[-3],file[-2],file[-1]+'.wav')
    test = os.path.join(file[-6],file[-5],file[-4]+'.wav')

    wav_path = '/hpctmp/ma_yi/dataset_vox1/voxceleb1/test/wav/'+enrol
    extract_phoneme(wav_path)
    wav_path = '/hpctmp/ma_yi/dataset_vox1/voxceleb1/test/wav/'+test
    extract_phoneme(wav_path)