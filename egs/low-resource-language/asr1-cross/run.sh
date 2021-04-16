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
stage=3        # start from 0 if you need to start from data preparation
stop_stage=3
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

exp=wav2vecfexlembglottoonly
plot_merged=false # plot cer with merged data in recog_set
# feature configuration
do_delta=false

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

irm_penalty_multiplier=0
irm_model_regularization=0
irm_phone_aware=false
dro_hard_choice=false
dro_model_regularization=0
lang_label=false
wav2vec_feature=false

babel_recog=""
gp_recog="Croatian Czech Bulgarian Polish"
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
recog_model=snapshot.ep.30
recog_function="recog_v2"

recog_size=200


# Generate configs with local/prepare_experiment_configs.py
langs_config=

mboshi_train=false
mboshi_recog=false
gp_romanized=false
ipa_transcript=true
mask_phoneme=false
lang2ph=""
	
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo "exp $exp"


# exp settings
if [ $exp == multi ]; then
echo Multi
# multilingual phoneme recognition without language label
tag="multi_transformer_ctc_apex" # tag for managing experiments.
train_config=conf/train_transformer_ctconly.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
elif [ $exp == crosslang ]; then
echo CrossLang
# multilingual phoneme recognition without language label
tag="crosslang_transformer_irm_lambda0.00001" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_irm.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
recog_function="recog_ctconly"
recog_model=snapshot.ep.29
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="DefaultSeg"
elif [ $exp == crossseg ]; then
echo CrossSeg
# multilingual phoneme recognition without language label
tag="slavic_ermseg" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_seg_6.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="DefaultSeg"
elif [ $exp == multlang ]; then
echo Multlang
# multilingual phoneme recognition with language label
tag="multilang_transformer_ctc" # tag for managing experiments.
experiment="Multilingual_LangAware"
train_config=conf/train_transformer_ctconly_multlang.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Multilingual_LangAware"
elif [ $exp == irmseg ]; then
echo IRMSEG
tag="slavic_irmseg_pen10"  # tag for managing experiments.
#tag="slavic_irm_base_droppenalty"
#tag="slavic_irm_base_droppenalty_droppenaltyctc"
experiment="IRM"
train_config=conf/train_transformer_ctconly_irmseg_6.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
irm_penalty_multiplier=10
#irm_penalty_multiplier=0
# irm_penalty_multiplier=0.1
irm_model_regularization=0

elif [ $exp == irmphaware ]; then
echo IRMPHAWARE
tag="slavic_irm_base" # tag for managing experiments.
#tag="slavic_irm_base_droppenalty"
#tag="slavic_irm_base_droppenalty_droppenaltyctc"
experiment="IRM"
train_config=conf/train_transformer_ctconly_irm.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
irm_penalty_multiplier=0.001
#irm_penalty_multiplier=0
# irm_penalty_multiplier=0.1
irm_model_regularization=0
irm_phone_aware=true

elif [ $exp == irmatt ]; then
echo IRMATT
tag="slavic_irm_att0.7_pen1" # tag for managing experiments.
#tag="slavic_irm_base_droppenalty"
#tag="slavic_irm_base_droppenalty_droppenaltyctc"
experiment="IRM"
train_config=conf/train_transformer_att0.7_irm.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
irm_penalty_multiplier=1
#irm_penalty_multiplier=0
# irm_penalty_multiplier=0.1
irm_model_regularization=0
irm_phone_aware=false

elif [ $exp == dro ]; then
echo DRO
tag="slavic_dro_hard" # tag for managing experiments.
experiment="DRO"
train_config=conf/train_transformer_ctconly_dro.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
dro_hard_choice=true

elif [ $exp == droseg ]; then
echo DROSEG
#tag="slavic_droseg_drop0.5" # tag for managing experiments.
tag="slavic_droseg_wreg1"
#tag="slavic_irm_base_droppenalty"
#tag="slavic_irm_base_droppenalty_droppenaltyctc"
experiment="DRO"
train_config=conf/train_transformer_ctconly_droseg_6.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
dro_hard_choice=true
dro_model_regularization=1

elif [ $exp == drowreg ]; then
echo DROWREG
tag="slavic_dro_hard_wreg0" # tag for managing experiments.
experiment="DRO"
train_config=conf/train_transformer_ctconly_dro.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
resume=
langs_config=
dro_hard_choice=true
dro_model_regularization=0

elif [ $exp == testcnn ]; then
echo TESTCNN
# multilingual phoneme recognition without language label
tag="testcnn" # tag for managing experiments.
train_config=conf/train_cnn_align.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
elif [ $exp == wav2vecenc ]; then
echo W2VENC
# multilingual phoneme recognition without language label
tag="wav2vecenc" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2venc.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
elif [ $exp == wav2vecfex ]; then
echo W2VFEX
# multilingual phoneme recognition without language label
tag="wav2vecfex" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
#resume=exp/train_pytorch_wav2vecfex/results/snapshot.ep.11
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlgcn ]; then
echo W2VFEXLGCN
# multilingual phoneme recognition without language label
tag="wav2vecfexlgcn" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lgcn.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlemb ]; then
echo W2VFEXLEMB
# multilingual phoneme recognition without language label
tag="wav2vecfexlemb" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lemb.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlemblinear ]; then
echo W2VFEXLEMBLINEAR
# multilingual phoneme recognition without language label
tag="wav2vecfexlemblinear" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lemb_linear.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlembadv ]; then
echo W2VFEXLEMBADV
# multilingual phoneme recognition without language label
tag="wav2vecfexlembadv" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lemb_adversarial.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Adversarial"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlembphonly ]; then
echo W2VFEXLEMBPHONLY
# multilingual phoneme recognition without language label
tag="wav2vecfexlembphonly" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lembphonly.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true
elif [ $exp == wav2vecfexlembglottoonly ]; then
echo W2VFEXLEMBGLOTTOONLY
ngpu=1
# multilingual phoneme recognition without language label
tag="wav2vecfexlembglottoonly" # tag for managing experiments.
train_config=conf/train_transformer_ctconly_w2vfex_lemb_glottoonly.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
# Generate configs with local/prepare_experiment_configs.py
resume=
langs_config=
experiment="Default"
recog_function="recog_ctconly"
mask_phoneme=true
lang2ph="phones/lang2ph.json"
lang_label=true
wav2vec_feature=true


fi


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
        --train-dtype O1 \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --sortagrad 0 \
        --experiment ${experiment} \
        --irm-model-regularization $irm_model_regularization \
        --irm-penalty-multiplier $irm_penalty_multiplier \
        --irm-phone-aware $irm_phone_aware \
        --dro-hard-choice $dro_hard_choice \
        --dro-model-regularization $dro_model_regularization
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    if [ $exp == irmatt ]; then
        nj=48
        ngpu=0
    else
        nj=1
        ngpu=1
    fi
    extra_opts=""
    if ${use_lm}; then
      extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    babel_recog="101 203 103 107 206 307 402 404 505"
    gp_recog="Spanish Polish Croatian Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
    # Generate configs with local/prepare_experiment_configs.py

    recog_set=""
    for l in ${babel_recog} ${gp_recog}; do
        recog_set="eval_${l} ${recog_set}"
    done
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

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --recog-function ${recog_function} \
            --recog-size ${recog_size} \
            --embedding-save-dir ${expdir}/${decode_dir}/embedding.JOB.json \
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


# Stage 5

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    babel_recog="101 203 103 107 206 307 402 404 505"
    gp_recog="Spanish Polish Croatian Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
    # Generate configs with local/prepare_experiment_configs.py
    resume=
    langs_config=
    recog_function="recog_ctconly"

    recog_set=""
    for l in ${babel_recog} ${gp_recog}; do
    recog_set="${recog_set} eval_${l}"
    done

    recog_size=200


    echo "stage 5: Plotting"
    if [ $exp == irmatt ]; then
        nj=48
        ngpu=0
    else
        nj=1
        ngpu=1
    fi

    extra_opts=""
    if ${use_lm}; then
    extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi
    mask_phoneme=true
    if ${mask_phoneme}; then
        for rtask in ${recog_set}; do
            ngpu=1
            pids=() # initialize pids
            plot_dir=plot_mask_${rtask}_$(basename ${decode_config%.*})
            mkdir -p ${expdir}/${plot_dir}
            echo "Saving in ${expdir}/${plot_dir}"
            
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            if [ -f "${feat_recog_dir}/data.json.npy" ]; then
        	    echo "Use ${feat_recog_dir}/data.json.npy"
                recog_json="${feat_recog_dir}/data.json.npy"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json.npy
            else
            	recog_json="${feat_recog_dir}/data.json"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
            fi
            
            
            #recog_json="${feat_recog_dir}/data.json"
            #concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
            echo "expdir ${expdir}"
            for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\."); do
            (
                if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                    echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                    
                else
                    echo "Evaluating $recog_model"
                    mkdir -p ${expdir}/${plot_dir}/${recog_model}
                    ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                        asr_recog.py \
                        --config ${decode_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --recog-json ${expdir}/${plot_dir}/data.merged.json \
                        --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                        --model ${expdir}/results/${recog_model}  \
                        --recog-function ${recog_function} \
                        --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                        --recog-size ${recog_size} \
                        --lang2ph ${lang2ph} \
                        --lang-label true \
                        --mask-phoneme true \
                        ${extra_opts}
                    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
                fi
            ) &
            pids+=($!) # store background pids
            wait $!

            done
            i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
            [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
            echo "Finished Computing CER for ${rtask}"
        done
    
    fi
    for rtask in ${recog_set}; do
        ngpu=1
        pids=() # initialize pids
        plot_dir=plot_${rtask}_$(basename ${decode_config%.*})
        mkdir -p ${expdir}/${plot_dir}
        echo "Saving in ${expdir}/${plot_dir}"
        
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        recog_json="${feat_recog_dir}/data.json"
        concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
        echo "expdir ${expdir}"
        for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\."); do
        (
            echo "Evaluating $recog_model"
            if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                
            else
                mkdir -p ${expdir}/${plot_dir}/${recog_model}
                ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                    asr_recog.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --recog-json ${expdir}/${plot_dir}/data.merged.json \
                    --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                    --model ${expdir}/results/${recog_model}  \
                    --recog-function ${recog_function} \
                    --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                    --recog-size ${recog_size} \
                    ${extra_opts}
                score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
            fi
        ) &
        pids+=($!) # store background pids
        wait $!

        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished Computing CER for ${rtask}"
    done
    for rtask in ${recog_set}; do
        echo "rtask ${rtask}"
        plot_dir=plot_${rtask}_$(basename ${decode_config%.*})
        local/plot_test.py -I ${expdir}/${plot_dir} -O "${expdir}/results/cer_test_${rtask}.png"
    done

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    babel_recog=""
    gp_recog="Croatian Polish Spanish 203 101"
    # Generate configs with local/prepare_experiment_configs.py
    resume=
    langs_config=
    recog_function="recog_ctconly"

    recog_set=""
    for l in ${babel_recog} ${gp_recog}; do
    recog_set="eval_${l} ${recog_set}"
    done

    recog_size=200


    echo "stage 6: Plotting with Fake Label"
    if [ $exp == irmatt ]; then
        nj=48
        ngpu=0
    else
        nj=1
        ngpu=1
    fi

    extra_opts=""
    if ${use_lm}; then
    extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    for rtask in ${recog_set}; do
        fake_lang_labels="CR PL SP PO TU GE BG TH CH FR CZ 203 101 N 404 402 307 206 107 103"
        for fake_lang_label in ${fake_lang_labels}; do
            ngpu=1
            pids=() # initialize pids
            plot_dir=plot_${rtask}_${fake_lang_label}_$(basename ${decode_config%.*})
            mkdir -p ${expdir}/${plot_dir}
            echo "Saving in ${expdir}/${plot_dir}"
            
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        	if [ -f "${feat_recog_dir}/data.json.npy" ]; then
        	    echo "Use ${feat_recog_dir}/data.json.npy"
                recog_json="${feat_recog_dir}/data.json.npy"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json.npy
            else
            	recog_json="${feat_recog_dir}/data.json"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
            fi

            echo "expdir ${expdir}"
            for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\."); do
            (
                echo "Evaluating $recog_model"
                if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                    echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                    
                else
                    mkdir -p ${expdir}/${plot_dir}/${recog_model}
                    ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                        asr_recog.py \
                        --config ${decode_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --recog-json ${expdir}/${plot_dir}/data.merged.json \
                        --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                        --model ${expdir}/results/${recog_model}  \
                        --recog-function ${recog_function} \
                        --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                        --recog-size ${recog_size} \
                        --mask-phoneme false \
                        --lang-label ${lang_label} \
                        --fake-lang-label ${fake_lang_label}
                        ${extra_opts}
                    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
                fi
            ) &
            pids+=($!) # store background pids
            wait $!

            done
            i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
            [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
            echo "Finished Computing CER for ${rtask} ${fake_lang_label}"
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

    babel_recog=""
    gp_recog="Croatian Polish Spanish 203 101"
    # Generate configs with local/prepare_experiment_configs.py
    resume=
    langs_config=
    recog_function="recog_ctconly"

    recog_set=""
    for l in ${babel_recog} ${gp_recog}; do
    recog_set="eval_${l} ${recog_set}"
    done

    recog_size=200


    echo "stage 7: Plotting with Fake Label and mask"
    if [ $exp == irmatt ]; then
        nj=48
        ngpu=0
    else
        nj=1
        ngpu=1
    fi

    extra_opts=""
    if ${use_lm}; then
    extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    for rtask in ${recog_set}; do
        fake_lang_labels="CR PL SP PO TU GE BG TH CH FR CZ 203 101 N 404 402 307 206 107 103"
        for fake_lang_label in ${fake_lang_labels}; do
            ngpu=1
            pids=() # initialize pids
            plot_dir=plot_mask_${rtask}_${fake_lang_label}_$(basename ${decode_config%.*})
            mkdir -p ${expdir}/${plot_dir}
            echo "Saving in ${expdir}/${plot_dir}"
            
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            if [ -f "${feat_recog_dir}/data.json.npy" ]; then
                echo "Use ${feat_recog_dir}/data.json.npy"
                recog_json="${feat_recog_dir}/data.json.npy"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json.npy
            else
            	recog_json="${feat_recog_dir}/data.json"
            	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
            fi
            echo "expdir ${expdir}"
            for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\."); do
            (
                echo "Evaluating $recog_model"
                if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                    echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                    
                else
                    mkdir -p ${expdir}/${plot_dir}/${recog_model}
                    ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                        asr_recog.py \
                        --config ${decode_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --recog-json ${expdir}/${plot_dir}/data.merged.json \
                        --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                        --model ${expdir}/results/${recog_model}  \
                        --recog-function ${recog_function} \
                        --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                        --recog-size ${recog_size} \
                        --mask-phoneme true \
                        --lang-label ${lang_label} \
                        --lang2ph ${lang2ph} \
                        --fake-lang-label ${fake_lang_label}
                        ${extra_opts}
                    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
                fi
            ) &
            pids+=($!) # store background pids
            wait $!

            done
            i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
            [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
            echo "Finished Computing CER for ${rtask} ${fake_lang_label}"
        done
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
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
    
    #babel_recog="Spanish Polish Croatian 101 203 103 107 206 307 402 404 505"
    #gp_recog="Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
    babel_recog="Spanish Polish Croatian 101 203 "
    gp_recog=""
    # Generate configs with local/prepare_experiment_configs.py
    resume=
    langs_config=
    recog_function="recog_ctconly"

    recog_set=""
    for l in ${babel_recog} ${gp_recog}; do
    recog_set="${recog_set} eval_${l}"
    done

    recog_size=200


    echo "stage 8: Plotting with Fake Label and mask"
    if [ $exp == irmatt ]; then
        nj=48
        ngpu=0
    else
        nj=1
        ngpu=1
    fi

    extra_opts=""
    if ${use_lm}; then
    extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    for rtask in ${recog_set}; do
        fake_lang_label="${lang2label[$rtask]}"
        echo  "${rtask} fake_lang_label ${fake_lang_label}"
        ngpu=1
        pids=() # initialize pids
        plot_dir=plot_${rtask}_${fake_lang_label}_$(basename ${decode_config%.*})
        mkdir -p ${expdir}/${plot_dir}
        echo "Saving in ${expdir}/${plot_dir}"
        
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    	if [ -f "${feat_recog_dir}/data.json.npy" ]; then
    	    echo "Use ${feat_recog_dir}/data.json.npy"
            recog_json="${feat_recog_dir}/data.json.npy"
        	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json.npy
        else
        	recog_json="${feat_recog_dir}/data.json"
        	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
        fi

        echo "expdir ${expdir}"
        for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\.30"); do
        (
            echo "Evaluating $recog_model"
            if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                
            else
                mkdir -p ${expdir}/${plot_dir}/${recog_model}
                ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                    asr_recog.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --recog-json ${expdir}/${plot_dir}/data.merged.json \
                    --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                    --model ${expdir}/results/${recog_model}  \
                    --recog-function ${recog_function} \
                    --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                    --recog-size ${recog_size} \
                    --mask-phoneme false \
                    --lang-label ${lang_label} \
                    --lang2ph ${lang2ph} \
                    --fake-lang-label ${fake_lang_label}
                    ${extra_opts}
                score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
            fi
        ) &
        pids+=($!) # store background pids
        wait $!

        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished Computing CER for ${rtask} ${fake_lang_label}"
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
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

    lang_model_weight=0

    #babel_recog="Spanish Polish Croatian 101 203 103 107 206 307 402 404 505"
    #gp_recog="Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
    babel_recog="101 203"
    gp_recog="Spanish Polish Croatian"
    # Generate configs with local/prepare_experiment_configs.py
    resume=
    langs_config=
    recog_function="recog_v2"
    
    
    recog_size=200


    echo "stage 9: Plotting with Language Model and mask"
    if [ $recog_function == recog_v2 ]; then
        nj=1
        ngpu=0
    else
        nj=1
        ngpu=1
    fi

    

    train_set=""
    recog_set=""
    for l in ${babel_recog}; do
        train_set="${train_set} data/${l}/data/train_${l}"
        recog_set="${recog_set} eval_${l}"
    done
    for l in ${gp_recog}; do
        train_set="${train_set} data/GlobalPhone/gp_${l}_train"
        recog_set="${recog_set} eval_${l}"
    done
    
    recog_set=($(echo "$recog_set" | tr ' ' '\n')) # convert string to array
    train_set=($(echo "$train_set" | tr ' ' '\n'))
    echo $train_set
    
    #./local/train_kenlm.sh --babel-recog ${babel_recog} --gp-recog ${gp_recog}

    for i in "${!train_set[@]}"; do
        trset="${train_set[i]}"
        rtask="${recog_set[i]}"
        echo $i ${recog_set[i]} ${trset}
        lang="${lang2label[${rtask}]}"
        lang_model_path=exp/lang_model/"${lang}".arpa
        lmexpdir=exp/rnnlm/${lang}
        extra_opts=""
        use_lm=true
        if ${use_lm}; then
            extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
        fi
        mkdir -p exp/lang_model
        echo "Converting ${lang}: ${trset}/text > ${lang_model_path} ${recog_set[i]}"

        fake_lang_label="${lang2label[$rtask]}"
        echo "${rtask} fake_lang_label ${fake_lang_label}"
        
        ngpu=1
        pids=() # initialize pids
        plot_dir=plot_lm5_mask_${rtask}_${fake_lang_label}_$(basename ${decode_config%.*})
        mkdir -p ${expdir}/${plot_dir}
        echo "Saving in ${expdir}/${plot_dir}"
        
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    	if [ -f "${feat_recog_dir}/data.json.npy" ]; then
    	    echo "Use ${feat_recog_dir}/data.json.npy"
            recog_json="${feat_recog_dir}/data.json.npy"
        	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json.npy
        else
        	recog_json="${feat_recog_dir}/data.json"
        	concatjson.py ${recog_json} > ${expdir}/${plot_dir}/data.merged.json
        fi

        echo "expdir ${expdir}"
        for recog_model in $(ls "${expdir}/results" | grep "snapshot\.ep\.30"); do
        (
            echo "Evaluating $recog_model"
            if [ -f "${expdir}/${plot_dir}/${recog_model}/result.txt" ]; then
                echo "Skip because ${expdir}/${plot_dir}/${recog_model}/result.txt exists"
                
            else
                mkdir -p ${expdir}/${plot_dir}/${recog_model}
                ${decode_cmd} JOB=1:${nj} ${expdir}/${plot_dir}/${recog_model}/log/decode.JOB.log \
                    asr_recog.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --recog-json ${expdir}/${plot_dir}/data.merged.json \
                    --result-label ${expdir}/${plot_dir}/${recog_model}/data.JOB.json \
                    --model ${expdir}/results/${recog_model}  \
                    --recog-function ${recog_function} \
                    --embedding-save-dir ${expdir}/${plot_dir}/${recog_model}/embedding.JOB.json \
                    --recog-size ${recog_size} \
                    --mask-phoneme true \
                    --lang-label ${lang_label} \
                    --lang2ph ${lang2ph} \
                    --lang-model \
                    --lang-model-weight \
                    --fake-lang-label ${fake_lang_label} \
                    --lm-weight 5 \
                    ${extra_opts}
                score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${plot_dir}/${recog_model} ${dict}
            fi
        ) &
        pids+=($!) # store background pids
        wait $!

        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished Computing CER for ${rtask} ${fake_lang_label}"
    done
fi

