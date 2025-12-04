import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
'''
def visual(path):
    #pos = nn.Softplus()
    weight = torch.load(path)
    weight = (weight-torch.amin(weight))/(torch.amax(weight)-torch.amin(weight))
    weight = weight+weight.T-torch.diag(weight.diag())
    sns.heatmap(weight,annot=False)
    plt.savefig(path.replace('.pt','.png'))
    plt.close()
'''
def plot_multiple(paras_list, score_dir, labels, colors):
    paras_list = [(paras - torch.amin(paras)) / (torch.amax(paras) - torch.amin(paras)) for paras in paras_list]

    line_idx = 0
    dict_raw = dict()

    # Load phoneme list
    with open('/home/svu/e0643891/test/wespeaker/examples/voxceleb/v2/pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1

    # Sort based on the first parameter set
    sorted_tensor, sorted_indices = torch.sort(paras_list[2],descending=True)
    sorted_names = [list(dict_raw.keys())[i] for i in sorted_indices]

    # Reorder all parameter sets according to the sorted indices
    sorted_paras_list = [paras[sorted_indices] for paras in paras_list]

    # Plotting
    bar_width = 0.15  # Width of each bar
    num_sets = len(paras_list)
    index = torch.arange(len(sorted_names))  # The x locations for the groups

    plt.figure(figsize=(18, 6))
    for i, (paras, label,color) in enumerate(zip(sorted_paras_list, labels, colors)):
        # Offset each set's bars to avoid overlap
        offset = bar_width * i
        bars = plt.bar(index + offset, paras.numpy(), bar_width, label=label,  color=color)

        # Annotate bars
        #for bar in bars:
            #yval = bar.get_height()
            #plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)

    # Set up the chart
    plt.xticks(index + bar_width * (num_sets - 1) / 2, sorted_names)
    plt.xlabel('Phone')
    plt.ylabel('Weight')
    plt.legend()
    plt.xlim(-0.5, len(sorted_names) - 0.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    plt.savefig(os.path.join(score_dir, 'multiple_phone_weight.png'))



if __name__ == "__main__":
    #path = '/home/mayi/Downloads/test/wespeaker/examples/voxceleb/v2/exp/cstrPho0_HardDiff0_selfcstr0_veri0.5_allpho_tri_norm/weight.pt'
    #visual(path)
    paras_list = [torch.load('/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_1s/models/weight.pt').squeeze(),
                  torch.load('/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2/models/weight.pt').squeeze(),
                  
                  torch.load('/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_3s/models/weight.pt').squeeze(),

                  torch.load('/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_4s/models/weight.pt').squeeze(),
                  torch.load('/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_5s/models/weight.pt').squeeze(),]
    score_dir = '/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_1s'
    labels = ['1s','2s','3s','4s','5s']
    plot_multiple(paras_list,score_dir,labels,['#4C91E6','#F28E2B','#76B7B2','#E15759','#B07AA1'])
    #plot_multiple(paras_list,score_dir,labels,['#326EA1','#4F91C3','#77AAC8','#9BC3DD','#CADAE9'])

