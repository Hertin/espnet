#!/usr/bin/env python3
# encoding: utf-8

import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="plot recog pter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-I", dest="input_text", type=str, help="Input text")
    parser.add_argument("-O", dest="output_text", type=str, help="Output text")
    return parser

def convert(text):
    s = ''
    for c in text:
        if c == '*':
            ch = '<unk>'
        elif c == ' ':
            ch = '<space>'
        else:
            ch = c
        s += ' ' + ch
    return s.strip()

if __name__ == "__main__":
    args = get_parser().parse_args()
    with open(args.input_text, 'r') as fin, open(args.output_text, 'w') as fout:
        for l in fin:
            utt, text = l.split(' ', maxsplit=1)
            text = convert(text.strip())
            fout.write(f'{utt} {text}\n')
