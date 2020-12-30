#!/bin/bash

# PIDP=1086104
# echo "Waiting for previous task $PIDP to be done..."
# while ps -p $PIDP > /dev/null;
# do sleep 5;
# done;
# echo "Previous task $PIDP done"
# sleep 100;

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
# recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# tag="l14_transformer_dist" # tag for managing experiments.
# train_config=conf/train_transformer_dist.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer_dist.yaml
# equal_accuracy_ratio=mix
# babel_langs="103 101 107 206 404 203 505  402 307"
# babel_recog="103 101 107 206 404 203 505 402 307"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# ita=0.0
# ctc_ita=0.1
# att_ita=0.0
# # recog_function="recog_transformer"
# recog_function="recog"
# # Generate configs with local/prepare_experiment_configs.py
# langs_config=
# resume=
# recog_model=model.acc.best

# tag="l14_transformer_sim" # tag for managing experiments.
# train_config=conf/train_transformer_sim.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer.yaml
# equal_accuracy_ratio=mix
# babel_langs="103 101 107 206 404 203 505 402 307"
# babel_recog="103 101 107 206 404 203 505 402 307"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# ita=0.0
# ctc_ita=0.0
# att_ita=0.0
# # recog_function="recog_transformer"
# recog_function="recog"
# # Generate configs with local/prepare_experiment_configs.py
# langs_config=
# resume=
# recog_model=model.acc.best

# tag="l13_transformer_sim_0.5" # tag for managing experiments.
# train_config=conf/train_transformer_sim_lsm0.5.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer.yaml
# equal_accuracy_ratio=mix
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# ita=0.0
# ctc_ita=0.0
# att_ita=0.0
# # Generate configs with local/prepare_experiment_configs.py
# resume=
# langs_config=

tag="l13_transformer_ctc_cross_base" # tag for managing experiments.
train_config=conf/train_transformer_ctc.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
equal_accuracy_ratio=mix
babel_langs="307"
babel_recog="307"
gp_langs="Czech"
gp_recog="Czech"
ita=0.0
ctc_ita=0.0
att_ita=0.0
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
recog_model=model.loss.best
recog_function="recog"

# tag="l13_transformer_ctc_signature_cross" # tag for managing experiments.
# train_config=conf/train_transformer_signature.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer_ctc.yaml
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# # Generate configs with local/prepare_experiment_configs.py
# resume=
# langs_config=
# recog_model=snapshot.ep.27
# recog_function="recog_transformer"


# tag="l13_transformer_cross" # tag for managing experiments.
# train_config=conf/train_transformer.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer.yaml
# equal_accuracy_ratio=mix
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# ita=0.0
# ctc_ita=0.0
# att_ita=0.0
# # Generate configs with local/prepare_experiment_configs.py
# resume=
# langs_config=
# recog_model=model.acc.best

# tag="l13_transformer" # tag for managing experiments.
# train_config=conf/train_transformer.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer.yaml
# equal_accuracy_ratio=mix
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# ita=0.0
# ctc_ita=0.0
# att_ita=0.0
# # Generate configs with local/prepare_experiment_configs.py
# resume=
# langs_config=



# tag="l14_transformer_dist_cross" # tag for managing experiments.
# train_config=conf/train_transformer_dist.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer_dist.yaml
# equal_accuracy_ratio=mix
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# # babel_langs=""
# # babel_recog=""
# # gp_langs="Mandarin"
# # gp_recog="Mandarin"
# ita=0.0
# ctc_ita=0.1
# att_ita=0.0
# # recog_function="recog_transformer"
# recog_function="recog"
# # Generate configs with local/prepare_experiment_configs.py
# langs_config=
# resume=
# recog_model=model.acc.best

# Generate configs with local/prepare_experiment_configs.py
langs_config=
retain_langs="307,103,101,402,107,206"

mboshi_train=false
mboshi_recog=false
gp_romanized=false
ipa_transcript=true

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=train
train_dev=dev_cross

# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=1

    extra_opts=""
    if ${use_lm}; then
      extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if ${use_lm}; then
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --recog-function ${recog_function} \
            ${extra_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
       
    ) &
    pids+=($!) # store background pids
    wait $!
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

