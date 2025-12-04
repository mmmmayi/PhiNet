import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import fire, torch
from sklearn.metrics.pairwise import cosine_similarity


def load_metrics(score_dir, trials):

    metrics_path = os.path.join(score_dir, f"{trials}_metrics_results.txt")
    try:
        with open(metrics_path, 'r') as f:
            evaluate = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # 获取基准值
    base_value = float(evaluate[0].split("\t")[1])
    
    # 读取最后 40 个指标
    metrics = [float(line.split("\t")[1]) for line in evaluate[-40:]]

    trait_metrics_path = os.path.join(score_dir, f"{trials}trait_metrics_results.txt")
    try:
        with open(trait_metrics_path, 'r') as f:
            trait_evaluate = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics file not found: {trait_metrics_path}")
    trait_metrics = [float(line.split("\t")[1]) for line in trait_evaluate[-40:]]
    return metrics, trait_metrics, base_value

def compute_score(score_dir,trials):

    for index, trial in enumerate(trials):
        metrics, trait_metrics, base = load_metrics(score_dir, trial)
        print(metrics)
        print(trait_metrics)
        score = 0
        for i in range(40):
            score+=abs(metrics[i]-trait_metrics[i])
        print("score for trial {}: {:.3f}".format(trial,score/40))

def plot(paras,score_dir,trials,color_spec, color_trait):
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
    #metrics = []
    #evaluate = open(os.path.join(score_dir,trials+'_metrics_results.txt'),'r').readlines()
    #base = evaluate[0].split("\t")
    #evaluate = evaluate[-40:]
    #base = float(base[1])
    #for i in evaluate:
        #i = i.split("\t")
        
        #assert int(i[0])<41 and int(i[0])>-1
        #metrics.append(float(i[1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1,1]})

    # 绘制第一个子图（柱状图）
    #bars = ax1.bar(sorted_names, sorted_tensor.numpy())
    #for bar in bars:
        #yval = bar.get_height()
        #ax1.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)

    #ax1.set_ylabel('Weight')
    #ax1.tick_params(axis='y')
    #ax1.set_title('Phoneme Weights')
    axs=[ax1,ax2,ax3]
    labels_spec=['spec-vox1-O','spec-sitw-dev','spec-sitw-eval']
    labels_trait=['trait-vox1-O','trait-sitw-dev','trait-sitw-eval']
    labels_performance=['perfomance-vox1-O','perfomance-sitw-dev','perfomance-sitw-eval']
    for index, trial in enumerate(trials):
        metrics, trait_metrics, base = load_metrics(score_dir, trial)
        name = trial.strip('_cleaned,kaldi')
        #axs[index].scatter(sorted_names, metrics, color=color[index], label=f'Spec:{name}', alpha=0.7)
        axs[index].scatter(sorted_names, metrics, color=color_spec[index], label=labels_spec[index], alpha=0.7)
        axs[index].scatter(sorted_names, trait_metrics, color=color_trait[index],  label=labels_trait[index], alpha=0.7)
        axs[index].axhline(y=base, color=color_spec[index], label=labels_performance[index], linestyle='--', linewidth=1.5)
    #ax2.scatter(sorted_names, metrics, color='r', marker='o')
    #ax1.set_ylim(5.5, 6.3)
    #ax2.set_ylim(5.7, 6.0)
    #ax3.set_ylim(6.8, 7.3)
    ax1.set_ylim(3.9, 4.6)
    ax2.set_ylim(3.7, 4.3)
    ax3.set_ylim(4.2, 4.8)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False, bottom=False)
    #ax2.xaxis.tick_bottom()  # 显示下轴的x刻度
    # 在断点位置绘制斜线
    d = .015  # 断点斜线的长度
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # 让斜线应用到下半轴
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)



    ax2.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax2.tick_params(labelbottom=False, bottom=False)
    ax3.xaxis.tick_bottom()  # 显示下轴的x刻度
    # 在断点位置绘制斜线
    d = .015  # 断点斜线的长度
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax3.transAxes)  # 让斜线应用到下半轴
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)




    ax1.set_ylabel('EER')
    ax2.set_ylabel('EER')
    ax3.set_ylabel('EER')
    ax1.tick_params(axis='y')
    ax3.set_xlabel('Phone')
    #ax2.set_title('Evaluation')
    ax1.legend()
    ax2.legend()  # 显示图例
    ax3.legend()


    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(score_dir, 'phone_weight_evaluated'))

def main(exp_dir,trials):
    trials=trials.split()
    color_spec=['#FF7F0E','#2CA02C','#1F77B4']
    color_trait=['#FFCEA3','#ABE8AB','#A8D2F0']
    store_score_dir = os.path.join(exp_dir, 'scores')
    para = torch.load(os.path.join(exp_dir,'models','weight.pt'))
    #plot(para.squeeze(),store_score_dir,trials,color_spec,color_trait)

    compute_score(store_score_dir,trials)

    
if __name__ == "__main__":
    fire.Fire(main)
