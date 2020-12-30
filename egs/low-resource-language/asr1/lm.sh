#!/bin/bash

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=2         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_li10.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_li10.yaml

# rnnlm related
use_lm=true
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.

# Generate configs with local/prepare_experiment_configs.py
langs_config=

if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
else
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
  babel_langs="307 103 101 402 107 206 404 203 505"
  babel_recog="307 103 101 402 107 206 404 203 505"
  # gp_langs="Arabic Czech French Korean Mandarin Spanish Thai"
  # gp_recog="${gp_langs}"
  gp_langs="Czech Mandarin Spanish Thai"
  gp_recog="${gp_langs}"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
  ipa_transcript=true
fi

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
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt

recog_set=""
# for l in ${babel_recog} ${gp_recog}; do
#   recog_set="eval_${l} ${recog_set}"
# done
recog_set=${recog_set%% }



feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt


if ${use_lm}; then
  lm_train_set=data/local/train.txt
  lm_valid_set=data/local/dev.txt

  # Make train and valid
  text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_set}/text | shuf | head -5000) \
                > ${lm_train_set}

  text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_dev}/text | shuf | head -500) \
                > ${lm_valid_set}

  ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
          lm_train.py \
          --config ${lm_config} \
          --ngpu ${ngpu} \
          --backend ${backend} \
          --verbose 1 \
          --outdir ${lmexpdir} \
          --tensorboard-dir tensorboard/${lmexpname} \
          --train-label ${lm_train_set} \
          --valid-label ${lm_valid_set} \
          --resume ${lm_resume} \
          --dict ${dict} \
          --train-dtype O1
fi

