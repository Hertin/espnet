#!/bin/bash
. ./path.sh
. ./cmd.sh

# dataset
babel_langs=""
babel_dev=""
babel_recog=""
gp_langs="Czech Bulgarian Polish"
gp_dev="Czech Bulgarian Polish"
gp_recog="Croatian Czech Bulgarian Polish"

set -e
set -u
set -o pipefail

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

# prepare lexicon
local/prepare_align_lexicon.py
utils/prepare_lang.sh data/lang_1char '<oov>' data/dev/ data/lang_1char || exit 1;

# prepare alignment data
for data_split in train dev ${recog_set}; do
    echo "** Prepare alignment data ${data_split} **"
    data_dir=data/${data_split}
    align_data_dir=${data_dir}/align
    align_output_dir=exp/align/${data_split}
    mkdir -p ${align_data_dir}
    mkdir -p ${align_output_dir}

    local/add_space_to_text.py -I ${data_dir}/text -O  ${align_data_dir}/text
    cp ${data_dir}/segments ${align_data_dir}
    cp ${data_dir}/wav.scp ${align_data_dir}
    cp ${data_dir}/utt2spk ${align_data_dir}
    cp ${data_dir}/spk2utt ${align_data_dir}
    
    njob=16 # default number of parallel jobs
    # get number of speakers and set number of jobs no larger than number of speakers
    nspk=$(wc -l ${align_data_dir}/spk2utt | awk  '{ print $1 }')
    njob=$(( njob < nspk ? njob : nspk ))
    echo "Number of speakers:${nspk} Number of jobs: ${njob}"

    utils/fix_data_dir.sh ${align_data_dir}

    # extract mfcc
    mfccdir=mfcc/${data_split}
    mkdir -p $mfccdir
    for x in ${align_data_dir}
    do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $x ${align_output_dir}/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh ${align_data_dir}
        steps/compute_cmvn_stats.sh $x ${align_output_dir}/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh ${align_data_dir}
    done

    # Train monophones
    steps/train_mono.sh --boost-silence 1.25 --nj ${njob} --cmd "$train_cmd" ${align_data_dir} data/lang_1char ${align_output_dir}/mono || exit 1;
    # Align monophones
    steps/align_si.sh --boost-silence 1.25 --nj ${njob} --cmd "$train_cmd" ${align_data_dir} data/lang_1char ${align_output_dir}/mono ${align_output_dir}/mono_ali || exit 1;
    # Train delta-based triphones
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 ${align_data_dir} data/lang_1char ${align_output_dir}/mono_ali ${align_output_dir}/tri1 || exit 1;
    # Align delta-based triphones
    steps/align_si.sh --nj ${njob} --cmd "$train_cmd" ${align_data_dir} data/lang_1char ${align_output_dir}/tri1 ${align_output_dir}/tri1_ali || exit 1;
    # Train delta + delta-delta triphones
    steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 ${align_data_dir} data/lang_1char ${align_output_dir}/tri1_ali ${align_output_dir}/tri2a || exit 1;
    # Align delta + delta-delta triphones
    steps/align_si.sh  --nj ${njob} --cmd "$train_cmd" --use-graphs true ${align_data_dir} data/lang_1char ${align_output_dir}/tri2a ${align_output_dir}/tri2a_ali || exit 1;
    # Train LDA-MLLT triphones
    steps/train_lda_mllt.sh --cmd "$train_cmd" 3500 20000 ${align_data_dir} data/lang_1char ${align_output_dir}/tri2a_ali ${align_output_dir}/tri3a || exit 1;
    # Align LDA-MLLT triphones with FMLLR
    steps/align_fmllr.sh --nj ${njob} --cmd "$train_cmd" ${align_data_dir} data/lang_1char ${align_output_dir}/tri3a ${align_output_dir}/tri3a_ali || exit 1;
    # Train SAT triphones
    steps/train_sat.sh --cmd "$train_cmd" 4200 40000 ${align_data_dir} data/lang_1char ${align_output_dir}/tri3a_ali ${align_output_dir}/tri4a || exit 1;
    # Align SAT triphones with FMLLR
    steps/align_fmllr.sh  --cmd "$train_cmd" ${align_data_dir} data/lang_1char ${align_output_dir}/tri4a ${align_output_dir}/tri4a_ali || exit 1;


    for i in ${align_output_dir}/tri4a_ali/ali.*.gz; do 
        $KALDI_ROOT/src/bin/ali-to-phones --ctm-output ${align_output_dir}/tri4a/final.mdl ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
    done;

    cat ${align_output_dir}/tri4a_ali/*.ctm > ${align_output_dir}/tri4a_ali/merged_alignment.txt

done
