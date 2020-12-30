#!/bin/bash

# PIDP=21240
# echo "Waiting for previous task 21240 to be done..."
# while ps -p $PIDP > /dev/null;
# do sleep 5;
# done;
# echo "Previous task 21240 done"
# sleep 100;

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
# resume=exp/train_pytorch_dptr_add/results/snapshot.ep.4        # Resume the training from snapshot


# feature configuration
do_delta=false



# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model= # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp settings

tag="l13_transformer_ctc_cross_base" # tag for managing experiments.
train_config=conf/train_transformer_ctconly.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_transformer.yaml
babel_langs="307"
babel_recog="307"
gp_langs="Czech"
gp_recog="Czech"
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=

# tag="l13_transformer_ctc_signature_cross" # tag for managing experiments.
# train_config=conf/train_transformer_signature.yaml
# lm_config=conf/lm.yaml
# decode_config=conf/decode_transformer.yaml
# babel_langs="307 103 101 402 107 206 404 203 505"
# babel_recog="307 103 101 402 107 206 404 203 505"
# gp_langs="Czech Mandarin Spanish Thai"
# gp_recog="Czech Mandarin Spanish Thai"
# # Generate configs with local/prepare_experiment_configs.py
# resume=
# langs_config=

# BABEL TRAIN:
# Amharic - 307
# Bengali - 103
# Cantonese - 101
# Javanese - 402
# Vietnamese - 107
# Zulu - 206
# Dutch (CGN) Fake Babel Code - 505
# BABEL TEST:
# Georgian - 404
# Lao - 203

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
train_dev=dev

# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
# lmexpname=train_rnnlm_${backend}_${lmtag}
# lmexpdir=exp/${lmexpname}
# lm_train_set=data/local/train.txt
# lm_valid_set=data/local/dev.txt

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

# dtype float 16 to facilitate training speed
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    echo "saving in ${expdir}"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-dtype float32 \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --sortagrad 0
fi
