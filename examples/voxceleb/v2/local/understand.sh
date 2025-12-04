exp_dir=
dur=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data
model_path=
trials_dir=${data}/vox1/trials

. tools/parse_options.sh
. path.sh
mkdir -p ${exp_dir}/scores
for x in $trials; do

    python wespeaker/bin/analysis.py \
      --exp_dir ${exp_dir} \
      --dur ${dur} \
      --eval_scp_path ${exp_dir}/scores/vox1_O/vox1_O_cleaned.kaldi.weights.scp \
      --model_path $model_path \
      --configs ${exp_dir}/config.yaml \
      ${trials_dir}/${x}
done

