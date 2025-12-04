exp_dir=
trials="vox1_O_cleaned.kaldi sitw_dev.kaldi sitw_eval.kaldi"
trials_dir=${data}/vox1/trials

. tools/parse_options.sh
. path.sh
mkdir -p ${exp_dir}/scores
python wespeaker/bin/plot_evaluate.py \
      --exp_dir ${exp_dir} \
      --trials "${trials}"

