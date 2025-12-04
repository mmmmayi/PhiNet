import os,random,sys
utt_spk = open('/data_a11/mayi/project/wespeaker/examples/voxceleb/v2/data/vox2_dev/utt2spk_test','r')
wav = open('/data_a11/mayi/project/SIP/lst/utt_len','r')
#train = open('/data_a11/mayi/project/wespeaker/examples/voxceleb/v2/data/vox2_dev/utt2spk_train','w')
#test = open('/data_a11/mayi/project/wespeaker/examples/voxceleb/v2/data/vox2_dev/utt2spk_test','w') 
train = open('/data_a11/mayi/project/SIP/lst/utt_len_train','w')
test = open('/data_a11/mayi/project/SIP/lst/utt_len_test','w')
lines = utt_spk.readlines()
test_spk = []
for line in lines:
    test_spk.append(line.split(' ')[0])
test_spk = list(dict.fromkeys(test_spk))

lines = wav.readlines()
for line in lines:
    if line.split(' ')[0] in test_spk:
        test.write(line)
    else:
        train.write(line)
'''
utts = {}

for line in lines:
    spk = line.strip().split(' ')[1]
    if spk not in utts:
        utts[spk] = [line.strip().split(' ')[0]]
    else:
        utts[spk].append(line.strip().split(' ')[0])

for spk, utt in utts.items():
    
    num = int(len(utts[spk])*0.1)
    shuffled = random.shuffle(utts[spk])
    for i in range(len(utts[spk])):
        if i<num:
            test.write(utts[spk][i]+' '+spk+'\n')
        else:
            train.write(utts[spk][i]+' '+spk+'\n')
sys.stdout.flush()
''' 

    
