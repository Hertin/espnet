#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from shutil import move
from sys import stderr

import regex as re


remap_dict = {}

remap_dict['cross-lingual'] = {}

remap_dict['cross-lingual']['ð'] = 'z' 
remap_dict['cross-lingual']['ɱ'] = 'm'
remap_dict['cross-lingual']['ʑ'] = 'z'
remap_dict['cross-lingual']['̪'] = ''
remap_dict['cross-lingual']['β'] = 'v'

remap_dict['leave-Mandarin-out'] = {}
remap_dict['leave-Mandarin-out']['ð'] = 'z' 
remap_dict['leave-Mandarin-out']['ɱ'] = 'm'
remap_dict['leave-Mandarin-out']['ʑ'] = 'z'
remap_dict['leave-Mandarin-out']['̪'] = ''
remap_dict['leave-Mandarin-out']['β'] = 'v'
remap_dict['leave-Mandarin-out']['ɚ'] = 'ə'
remap_dict['leave-Mandarin-out']['ɻ'] = 'ɹ'
remap_dict['leave-Mandarin-out']['ʐ'] = 'v'

remap_dict['leave-Portuguese-out'] = {}
remap_dict['leave-Portuguese-out']['ð'] = 'z' 
remap_dict['leave-Portuguese-out']['ɱ'] = 'm'
remap_dict['leave-Portuguese-out']['ʑ'] = 'z'
remap_dict['leave-Portuguese-out']['̪'] = ''
remap_dict['leave-Portuguese-out']['β'] = 'v'
remap_dict['leave-Portuguese-out']['ʎ'] = 'j'

remap_dict['leave-French-out'] = {}
remap_dict['leave-French-out']['ð'] = 'z' 
remap_dict['leave-French-out']['ɱ'] = 'm'
remap_dict['leave-French-out']['ʑ'] = 'z'
remap_dict['leave-French-out']['̪'] = ''
remap_dict['leave-French-out']['β'] = 'v'
remap_dict['leave-French-out']['R'] = 'ʁ'

remap_dict['leave-Bulgarian-out'] = {}
remap_dict['leave-Bulgarian-out']['ð'] = 'z' 
remap_dict['leave-Bulgarian-out']['ɱ'] = 'm'
remap_dict['leave-Bulgarian-out']['ʑ'] = 'z'
remap_dict['leave-Bulgarian-out']['̪'] = ''
remap_dict['leave-Bulgarian-out']['β'] = 'v'

remap_dict['leave-Czech-out'] = {}
remap_dict['leave-Czech-out']['ð'] = 'z' 
remap_dict['leave-Czech-out']['ɱ'] = 'm'
remap_dict['leave-Czech-out']['ʑ'] = 'z'
remap_dict['leave-Czech-out']['̪'] = ''
remap_dict['leave-Czech-out']['β'] = 'v'
remap_dict['leave-Czech-out']['̝'] = ''


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
        if '206' in str(text_path):
            text = text.replace('ʱ', '̤')
        if 'Bulgarian' in str(text_path):
            text = text.replace('∅','')
        if 'Portuguese' in str(text_path):
            text = text.replace('∅','')
        if 'Portuguese' in str(text_path):
            text = text.replace('ã','a  ̃')
        if 'Portuguese' in str(text_path):
            text = text.replace('õ','o  ̃')
        if 'French' in str(text_path):
            text = text.replace('č', 't ʃ')
            
        for symb in cur_remap.keys():
            text = text.replace(symb, cur_remap[symb])
        text = re.sub(' +', ' ', text)
        print(key, text, file=fout)

backup_path = text_path.with_suffix(".remapped.bak")
move(text_path, backup_path)
move(norm_text_path, text_path)

