#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

. ./path.sh || exit 1

stage=1
stop_stage=3
data=data
data_type="raw"  # shard/raw

config=conf/ecapa_tdnn.yaml

#exp_dir=/hpctmp/e0643891/exp/cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_3s
exp_dir=exp/vox2en_cstrPho0.001_HardDiff0.0015_selfcstr0_veri0.5_same11_c2_3s
gpus="[0]"
num_avg=10
checkpoint=

trials="vox1_O_cleaned.kaldi"
top_n=300


. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 10 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk  \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  #python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  #python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2/${data_type}.list \
      --train_label ${data}/vox2/utt2spk \
      --pho_path /data_a11/mayi/dataset_vox1/voxceleb1/wav \
      --reverb_data /data_a11/mayi/dataset/RIRS_NOISES/file_list \
      --noise_data /data_a11/mayi/dataset/musan/file_list \
      ${checkpoint:+--checkpoint $checkpoint}
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    avg_model=$exp_dir/models/model_0.pt
    dur=0
    trials="sitw_dev.kaldi sitw_eval.kaldi" 
    #trials="sitw_dev.kaldi sitw_dev.kaldi.overlap_2_5pho sitw_dev.kaldi.overlap_2_8pho sitw_dev.kaldi.overlap_2_10pho sitw_dev.kaldi.overlap_2_20pho"
    #trials="sitw_eval.kaldi sitw_eval.kaldi.overlap_2_5pho sitw_eval.kaldi.overlap_2_8pho sitw_eval.kaldi.overlap_2_10pho sitw_eval.kaldi.overlap_2_20pho"
    sub="sitw"
    echo "Extract embeddings for sitw ..."
    
    local/extract_sitw.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}

    echo "Score ..."
    
    local/pho_score.sh \
        --dur $dur \
        --sub "sitw"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/sitw/trials"
    
    local/score.sh \
        --dur $dur\
        --sub "sitw" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  
  trials="vox1_O_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  
  #python wespeaker/bin/average_model.py \
    #--dst_model $avg_model \
    #--src_path $exp_dir/models \
    #--num ${num_avg}
  
  for dur in 0 ; do
    
    local/extract_voxO.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
    
    
    local/pho_score.sh \
        --dur $dur \
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials"
    
    local/score.sh \
        --dur $dur\
        --sub "vox1_O" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"
  done
    '''
    dur=0
    local/understand.sh \
        --dur ${dur} \
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials"
   '''
fi

#if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  #xp_dir="exp/2pooling+exp/2pooling_constrain+exp/2pooling_constrain_diff+exp/2pooling_constrain_cosOneSide+exp/ecapa_tstp"
  #python wespeaker/bin/tsne.py --exp_dir ${exp_dir} 
#fi 

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

  trials="vox1_O_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true
  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)
  
  for idx in $indices; do
    echo $idx
    #if [ "$idx" -eq 16 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
    local/extract_voxO_del_pho.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} --dele $idx
    
    local/pho_score.sh \
        --dur $dur \
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials"
    
    local/score.sh \
        --dur $dur\
        --sub "vox1_O" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"  \
        --idx $idx
    fi
  done
 
  
fi

echo 'start stage 7'
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  trials="vox1_O_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true

  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)
  
  local/extract_voxO.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
  
  for idx in $indices; do
    echo $idx
    
    #if [ "$idx" -eq 32 ]; then
        #start_processing=true
        #continue  
    #fi
    
    if [ "$start_processing" = true ]; then
    local/pho_score.sh \
        --dur $dur \
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --dele $idx
    
    local/score.sh \
        --dur $dur\
        --sub "vox1_O" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"  \
        --idx $idx \
        --type "trait"
    fi
  done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  trials="sitw_dev.kaldi sitw_eval.kaldi" 
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=false
  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)

  for idx in $indices; do
    echo $idx
    if [ "$idx" -eq 16 ]; then
        start_processing=true
        continue  
    fi
    if [ "$start_processing" = true ]; then
    local/extract_sitw.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} --dele $idx
    
    local/pho_score.sh \
        --dur $dur \
        --sub "sitw"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/sitw/trials"
    
    local/score.sh \
        --dur $dur\
        --sub "sitw" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"  \
        --idx $idx
    fi
  done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  trials="sitw_dev.kaldi sitw_eval.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true

  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)
  
  local/extract_sitw.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
  
  for idx in $indices; do
    echo $idx
    #if [ "$idx" -eq 11 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
    local/pho_score.sh \
        --dur $dur \
        --sub "sitw"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/sitw/trials" \
        --dele $idx
    
    local/score.sh \
        --dur $dur\
        --sub "sitw" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"  \
        --idx $idx \
        --type "trait"
    fi
  done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  trials="vox1_O_cleaned.kaldi sitw_dev.kaldi sitw_eval.kaldi"
  #trials="vox1_O_cleaned.kaldi"
  local/plot.sh \
      --exp_dir $exp_dir \
      --trials "$trials"
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  
  #trials="vox1_O_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  #trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi" 
  trials="vox1_H_cleaned.kaldi"
  for dur in 0 ; do
    
    local/extract_vox.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
    
  
    local/pho_score.sh \
        --dur $dur \
        --sub "vox1"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials"
    
    local/score.sh \
        --dur $dur\
        --sub "vox1" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials"
  done
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  avg_model=$exp_dir/models/model_0.pt
  dur=0
  trials="clean.kaldi" 
  sub="librispeech"
  echo "Extract embeddings for librispeech ..."
  
  local/extract_librispeech.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  echo "Score ..."
  
  local/pho_score.sh \
        --dur $dur \
        --sub "librispeech"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/librispeech/trials"
  
    local/score.sh \
        --dur $dur\
        --sub "librispeech" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --idx $idx
  
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  trials="clean.kaldi" 
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true
  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)

  for idx in $indices; do
    #echo $idx
    #if [ "$idx" -eq 16 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
    local/extract_librispeech.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} --dele $idx
    
    local/pho_score.sh \
        --dur $dur \
        --sub "librispeech"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/librispeech/trials"    

    local/score.sh \
        --dur $dur\
        --sub "librispeech" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --idx $idx
    fi
  done
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  trials="clean.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true

  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)
  
  local/extract_librispeech.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  for idx in $indices; do
    echo $idx
    #if [ "$idx" -eq 11 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
    local/pho_score.sh \
        --dur $dur \
        --sub "librispeech"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --trials_dir "data/librispeech/trials" \
        --dele $idx
    
    local/score.sh \
        --dur $dur\
        --sub "librispeech" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --idx $idx \
        --type "trait"
    fi
  done
fi
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  trials="vox1_H_cleaned.kaldi" 
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true
  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)

  for idx in $indices; do
    #echo $idx
    #if [ "$idx" -eq 16 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
    local/extract_vox.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data} --dele $idx 

    local/pho_score.sh \
        --dur $dur \
        --sub "vox1"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials"

    local/score.sh \
        --dur $dur\
        --sub "vox1" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --idx $idx

    fi
  done
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
  trials="vox1_H_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  tensor_file=$exp_dir/models/weight.pt
  start_processing=true

  dur=0
  indices=$(python - <<EOF
import torch
tensor = torch.load("$tensor_file").squeeze()
sorted_indices = tensor.argsort(descending=True)
print(' '.join(map(str, sorted_indices.tolist())))
EOF
)
  
  local/extract_vox.sh \
      --dur $dur \
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  for idx in $indices; do
    echo $idx
    #if [ "$idx" -eq 11 ]; then
        #start_processing=true
        #continue  
    #fi
    if [ "$start_processing" = true ]; then
   
    local/pho_score.sh \
        --dur $dur \
        --sub "vox1"\
        --data ${data} \
        --model_path $avg_model \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --dele $idx
    
    local/score.sh \
        --dur $dur\
        --sub "vox1" \
        --stage 2 --stop-stage 2 \
        --data ${data} \
        --suffix "pho_score" \
        --exp_dir $exp_dir \
        --trials "$trials" \
        --idx $idx \
        --type "trait"


    fi
  done
fi

