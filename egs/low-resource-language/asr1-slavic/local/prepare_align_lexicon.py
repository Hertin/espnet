#!/usr/bin/env python3
# encoding: utf-8

with open('data/lang_1char/train_units.txt',  'r') as fin, open('data/lang_1char/lexicon.txt', 'w') as fout:
    fout.write('<oov> <oov>\n')
    fout.write('<sil> <sil>\n')
    for l in fin:
        ph, _ = l.strip().split(maxsplit=1)
        fout.write(f'{ph} {ph}\n')

with open('data/lang_1char/silence_phones.txt', 'w') as f:
    f.write('<sil>\n<space>\n')

with open('data/lang_1char/lexicon.txt',  'r') as fin, open('data/lang_1char/nonsilence_phones.txt', 'w') as fout:
    for l in fin:
        ph, _ = l.strip().split(maxsplit=1)
        if ph != '<space>' and ph != '<sil>':
            fout.write(f'{ph}\n')

with open('data/lang_1char/optional_silence.txt', 'w') as f:
    f.write('<sil>\n')
