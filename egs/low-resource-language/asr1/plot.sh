#!/bin/bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


backend=pytorch
seed=1
debugmode=1
dumpdir=dump
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option

tag="slavic_irm_base_ly6" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_irm.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
babel_recog=""
gp_recog="Croatian" # Czech Bulgarian Polish
# Generate configs with local/prepare_experiment_configs.py
resume=

# feature configuration
do_delta=false

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

langs_config=
recog_function="recog_ctconly"

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set=train

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

expdir=exp/${expname}
mkdir -p ${expdir}

echo "stage 5: Plotting"
nj=1

extra_opts=""
if ${use_lm}; then
extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
fi


# concatenate all recognition json files
recog_jsons=""
plot_dir=plot_$(basename ${decode_config%.*})
mkdir -p ${expdir}/${plot_dir}

for rtask in ${recog_set}; do
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    recog_jsons="${recog_jsons} ${feat_recog_dir}/data.json"
done
echo "recog jsons: ${recog_jsons}"

concatjson.py ${recog_jsons} > ${expdir}/${plot_dir}/data.merged.json

# evaluate on all snapshot
ngpu=1
pids=() # initialize pids

# for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\."); do
# (
#     echo "Evaluating $recog_model"
#     mkdir -p ${expdir}/${plot_dir}/${recog_model}
#     ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/log/decode.JOB.log \
#         asr_recog.py \
#         --config ${decode_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --recog-json ${expdir}/${plot_dir}/data.merged.json \
#         --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
#         --model ${expdir}/results/${recog_model}  \
#         --recog-function ${recog_function} \
#         ${extra_opts}
#     score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
# ) &
# pids+=($!) # store background pids
# wait $!

# done

# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished Computing CER"

local/plot_test.py -I ${expdir}/${plot_dir} -O "${expdir}/results/cer_test.png"

echo "Finished Plotting"
