#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from shutil import move
from sys import stderr

import regex as re

number = re.compile(r"\d")

# Match every punctuation besides hyphens
# https://stackoverflow.com/questions/21209024/
# python-regex-remove-all-punctuation-except-hyphen-for-unicode-string
punctuation = re.compile(r"[^\P{P}-]+")
mand_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

parser = ArgumentParser(
    description="Filter a Kaldi text file to remove utterances with numbers."
                " Used to avoid inconsistency in training/eval."
)
parser.add_argument("text")
parser.add_argument(
    "--remove-digit-utts",
    action="store_true",
    help="Remove utterances that contain digits.",
)
parser.add_argument(
    "--strip-punctuation",
    action="store_true",
    help="Strip punctuation from utterances.",
)

args = parser.parse_args()
text_path = Path(args.text)
norm_text_path = text_path.with_suffix(".norm")

remove_counter = 0
norm_counter = 0
print(args.remove_digit_utts, args.strip_punctuation,file=stderr)
with open(text_path) as fin, open(norm_text_path, "w") as fout:
    for idx, line in enumerate(fin):
        ret = line.strip().split(sep = " ", maxsplit=1)
        if len(ret) == 1:
            continue
        key = ret[0]
        text = ret[1].strip()
        if 'GlobalPhone' in str(text_path):
            text = text.replace('<', '')
            text = text.replace('->', '')
            text = text.replace('>', '')
            if 'French' in str(text_path):
                if 'ë' in text:
                    remove_counter += 1
                    continue
                if 'ü' in text:
                    remove_counter += 1
                    continue
                text = text.replace('ď','ï')
                text = text.replace('ę', 'ê')
                text = text.replace('ŕ', 'à')
                text = text.replace('ů', 'u')
                text = text.replace('ű','û')
            if 'Turkish' in str(text_path):
                if 'â' in text:
                    remove_counter += 1
                    continue
                if 'í' in text:
                    remove_counter += 1
                    continue
                if 'î' in text:
                    remove_counter += 1
                    continue
                text = text.replace('ď','ï')
                text = text.replace('ę', 'ê')
                text = text.replace('ŕ', 'à')
                text = text.replace('ů', 'u')
                text = text.replace('ű','û')     
 
        else:
            text = text.replace('<hes>','')
            text = text.replace('<noise>','')
            text = text.replace('<silence>','')
            text = text.replace('<unk>','')
            text = text.replace('<v-noise>','')
        text = re.sub(' +', ' ', text)
        if args.remove_digit_utts and number.search(text):
            remove_counter += 1
            continue
        if args.strip_punctuation:
            text = punctuation.sub("", text)
            if 'Mandarin' in str(text_path):
                text = re.sub(r"[%s]+" % mand_punc, "", text)
            norm_counter += 1
        print(key, text, file=fout)

backup_path = text_path.with_suffix(".norm.bak")
move(text_path, backup_path)
move(norm_text_path, text_path)

print(
    f"Inputs utts: {idx + 1} -- removed: {remove_counter} --"
    f" normalized: {norm_counter}",
    file=stderr,
)
