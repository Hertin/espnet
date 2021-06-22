#!/bin/bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


declare -A lang2label

lang2label["eval_101"]="101"
lang2label["eval_203"]="203"
lang2label["eval_103"]="103"
lang2label["eval_107"]="107"
lang2label["eval_206"]="206"
lang2label["eval_307"]="307"
lang2label["eval_402"]="402"
lang2label["eval_404"]="404"
lang2label["eval_505"]="N"
lang2label["eval_Spanish"]=SP
lang2label["eval_Polish"]=PL
lang2label["eval_Croatian"]=CR
lang2label["eval_Czech"]=CZ
lang2label[eval_French]=FR
lang2label[eval_Mandarin]=CH
lang2label[eval_Thai]=TH
lang2label[eval_Bulgarian]=BG
lang2label[eval_German]=GE
lang2label[eval_Turkish]=TU
lang2label[eval_Portuguese]=PO

babel_recog="101 203"
gp_recog="Spanish Polish Croatian"
ngpu=1
lm_config=conf/lm.yaml
backend=pytorch
lm_resume=
dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=""
recog_set=""
dev_set=""


for l in ${babel_recog}; do
train_set="${train_set} data/${l}/data/train_${l}"
recog_set="${recog_set} eval_${l}"
dev_set="${dev_set} data/${l}/data/dev_${l}"
done
for l in ${gp_recog}; do
train_set="${train_set} data/GlobalPhone/gp_${l}_train"
recog_set="${recog_set} eval_${l}"
dev_set="${dev_set} data/GlobalPhone/gp_${l}_dev"
done

recog_set=($(echo "$recog_set" | tr ' ' '\n')) # convert string to array
train_set=($(echo "$train_set" | tr ' ' '\n'))
dev_set=($(echo "$dev_set" | tr ' ' '\n'))
echo $train_set

for i in "${!train_set[@]}"; do
    trset="${train_set[i]}"
    rtask="${recog_set[i]}"
    dtask="${dev_set[i]}"
    # echo $i ${recog_set[i]} ${trset}
    lang="${lang2label[${rtask}]}"
    lang_model_path=exp/lang_model/"${lang}".arpa
    lmexpdir=exp/rnnlm/${lang}
    lmexpname=rnnlm${lang}
    mkdir -p exp/lang_model
    mkdir -p ${lmexpdir}
    lm_train_set=${lmexpdir}/train.txt
    lm_valid_set=${lmexpdir}/dev.txt
    echo "Converting lang: ${lang}: ${trset}/text > ${lang_model_path} rtask: ${recog_set[i]} dtask: ${dtask}"

    text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- "${dtask}/text" | head -1000) \
                > ${lm_train_set}

    text2token.py --nchar 1 \
                --space "<space>" \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- "${dtask}/text" | head -100) \
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
          --dict ${dict}
done
