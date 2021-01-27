#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from shutil import move
from sys import stderr

import regex as re
import sys,csv
import os
import codecs
import numpy as np
import codecs

parser = ArgumentParser(
    description="Forced Alignment into segments/text/utt2spk/wavscp"
)
parser.add_argument(
    "--phones_txt",
    default="",
    help="txt file for all the phones",
)
parser.add_argument(
    "--kaldi-egs-dir",
    default="",
    help="Directory for holding exp/gmm/...",
)
parser.add_argument(
    "--espnet-data-dir",
    default="",
    help="Directory for holding espnet/egs/.../data",
)
parser.add_argument(
    "--espnet-resave-dir",
    default="",
    help="Resave directory for holding espnet/egs/.../data",
)
parser.add_argument(
    "--langs",
    default="",
    help="Resave directory for holding espnet/egs/.../data",
)
parser.add_argument(
    "--gp-langs",
    default="",
    help="gp_langs",
)
parser.add_argument(
    "--gp-dev",
    default="",
    help="gp_dev",
)
parser.add_argument(
    "--gp-recog",
    default="",
    help="gp_recog",
)
parser.add_argument(
    "--min-duration",
    default=0.03,
    type=float,
    help="min duration",
)
parser.add_argument(
    "--subsample-factor",
    default=3,
    type=int,
    help="subsample a fraction of the speakers for train and dev",
)

args = parser.parse_args()

results=[]

langs = args.langs.split()

phs_dict = {}

f_phones = codecs.open(args.phones_txt).readlines()

for line in f_phones:
    phs = line.replace('\n','').split(' ')
    phs_dict[phs[1]] = phs[0]

print(phs_dict)

for lang in langs:

    ali_txt = os.path.join(args.kaldi_egs_dir, 'exp/gmm/gp_' + lang + '_train_all/tri5_ali/merged_alignment.txt')

    with open(ali_txt) as f:
        lines = f.readlines()

    name = lines[0].split(" ")[0]
    name_fin = lines[-1].split(" ")[0]
    print(name, name_fin)

    save_folder = os.path.join(args.kaldi_egs_dir, 'exp/gmm/gp_' + lang + '_train_all/tri5_ali/forced_alignment/')
    os.makedirs(save_folder, exist_ok = True)

    with open(ali_txt) as f:
        for line in f.readlines():
            columns=line.replace('\n','').split(" ")
            # print(columns)
            columns[-1] =phs_dict[columns[-1]]
            # print(columns)
            name_prev = name
            name = columns[0]
            if (name_prev != name):

                with open(save_folder + (name_prev)+".txt",'w') as fwrite:
                    fwrite.write("\n".join(results))
                del results[:]
                results = []

                results.append(" ".join(columns[2:]))
            else:
                results.append(" ".join(columns[2:]))

    with open(save_folder + (name_prev)+".txt",'w') as fwrite:
        fwrite.write("\n".join(results))
        print(name_prev)
        del results[:]
        results = []



gp_langs=args.gp_langs.split()
for idx in range(len(gp_langs)):
    gp_langs[idx] = gp_langs[idx] + '_train'

gp_dev=args.gp_dev.split()
for idx in range(len(gp_dev)):
    gp_dev[idx] = gp_dev[idx] + '_dev'

gp_recog=args.gp_recog.split()
for idx in range(len(gp_recog)):
    gp_recog[idx] = gp_recog[idx] + '_eval'




for gp_lang in gp_langs + gp_dev + gp_recog:
    text_lines = []
    segment_lines = []
    utt2spk_lines = []
    wavscp_lines = []

    save_folder = os.path.join(args.kaldi_egs_dir,'exp/gmm/gp_' + gp_lang[:gp_lang.find('_')] + '_train_all/tri5_ali/forced_alignment/')

    files = os.listdir(save_folder)

    text_dir = os.path.join(args.espnet_data_dir, 'GlobalPhone/gp_' + gp_lang, 'text')
    wavscp_dir = os.path.join(args.espnet_data_dir, 'GlobalPhone/gp_' + gp_lang, 'wav.scp')

    f = codecs.open(text_dir).readlines()
    for line in f:
        uttid = line.split(maxsplit = 1)[0].replace('\n','')
        if (uttid + '.txt') in files:
            alignment = codecs.open(os.path.join(save_folder,uttid + '.txt')).readlines()
            for line_idx in range(len(alignment)):
                start_time, dur, phone = alignment[line_idx].replace('\n','').split()
                end_time = round(float(start_time) + float(dur), 2)
                end_time = str(end_time)
                if '<' not in phone and float(dur) > args.min_duration:
                    seg_uttid = uttid + '_' + str(line_idx)
                    spkid = uttid.split("_")[0]
                    if (int(spkid[2:]) % args.subsample_factor == 0) or 'eval' in gp_lang:
                        segment_lines.append(" ".join([seg_uttid, uttid, start_time, end_time]) + '\n')
                        text_lines.append(" ".join([seg_uttid, phone]) + '\n')
                        utt2spk_lines.append(" ".join([seg_uttid, spkid]) + '\n')
               

    f = codecs.open(wavscp_dir).readlines()
    for line in f:
        uttid = line.split(maxsplit = 1)[0].replace('\n','')
        if (uttid + '.txt') in files:
            wavscp_lines.append(line)

    resave_folder = os.path.join(args.espnet_resave_dir,
                                 'GlobalPhone/gp_' + gp_lang)

    os.makedirs(resave_folder, exist_ok = True)

    with open(os.path.join(resave_folder, 'text'), 'w') as f:
        f.writelines(text_lines)

    with open(os.path.join(resave_folder, 'segments'), 'w') as f:
        f.writelines(segment_lines)

    with open(os.path.join(resave_folder, 'utt2spk'), 'w') as f:
        f.writelines(utt2spk_lines)

    with open(os.path.join(resave_folder, 'wav.scp'), 'w') as f:
        f.writelines(wavscp_lines)
