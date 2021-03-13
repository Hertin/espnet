#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from shutil import move
from sys import stderr

import regex as re


remap_dict = {}

remap_dict['cross-lingual'] = {}

remap_dict['cross-lingual']['e'] = 'i' 
remap_dict['cross-lingual']['ʋ'] = 'v'


parser = ArgumentParser(
    description="Filter an IPA text file to remap OOV IPA."
                " Used to avoid inconsistency in training/eval."
)
parser.add_argument("--text")
parser.add_argument(
    "-d", "--data-dir", help="Path to Kaldi data directory with text file."
)
parser.add_argument(
    "-e", "--expname", help="Experiment name for choosing remap_dict"
)

args = parser.parse_args()
text_path = Path(args.text)
norm_text_path = text_path.with_suffix(".remapped")
cur_remap = remap_dict[str(args.expname)]

# Perform IPA remapping
with open(text_path) as fin, open(norm_text_path, "w") as fout:
    for idx, line in enumerate(fin):
        ret = line.strip().split(sep = " ", maxsplit=1)
        if len(ret) == 1:
            continue
        key = ret[0]
        text = ret[1].strip()
        if 'Bulgarian' in str(text_path):
            text = text.replace('∅','')
            
        for symb in cur_remap.keys():
            text = text.replace(symb, cur_remap[symb])
        text = re.sub(' +', ' ', text)
        print(key, text, file=fout)

backup_path = text_path.with_suffix(".remapped.bak")
move(text_path, backup_path)
move(norm_text_path, text_path)

