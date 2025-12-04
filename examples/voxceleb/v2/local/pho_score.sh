dur=
exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data
model_path=
trials_dir=${data}/vox1/trials
sub="vox1_O"
dele=None

. tools/parse_options.sh
. path.sh
mkdir -p ${exp_dir}/scores
for x in $trials; do
    python wespeaker/bin/pho_similarity.py \
      --dur ${dur} \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/${sub}/phovector_${dur}s.scp \
      --model_path $model_path \
      --configs ${exp_dir}/config.yaml \
      --dele ${dele} \
      ${trials_dir}/${x}
    
done

