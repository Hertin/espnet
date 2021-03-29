import os
import numpy as np
import json
from ipywidgets import interact
import matplotlib.pyplot as plt
import torch
import pickle as pk
from espnet.asr.pytorch_backend.asr import load_trained_model

import matplotlib
lembs = []

import argparse

parser = argparse.ArgumentParser(description='Extract language embeddings')
parser.add_argument('--exp-dir', type=str,
                    help='path to experiment result directory')

args = parser.parse_args()
lang_labels = [
    'CR', 'PL', 'SP', 'PO', 'TU', 'GE', 'BG', 'TH', 'CH', 'FR', 'CZ', 
    '203', '101', 'N', '404', '402', '307', '206', '107', '103'
]
lembs = {}
for snapshot in os.list(args.exp_dir):
    if not snapshot.startswith('snapshot.ep.'):
        continue
    ep = int(snapshot.replace('snapshot.ep.', ''))
    model, train_args = load_trained_model(f'{args.exp_dir}/{snapshot}')
    device = torch.device('cuda')
    model = model.float()
    model = model.to(device)
    with torch.no_grad():
        if 'lgcn' in args.exp_dir:
            lemb = model.lgcn(lang_labels)
        elif 'lemb' in args.exp_dir:
            lemb = model.lemb(lang_labels)
        else:
            raise ValueErorr(f'model for {args.exp} not implemented')
        lemb = lemb.detach().cpu().numpy()
        lembs[ep] = lemb
    del model
with open(f'{args.exp_dir}/language_emb.pk', 'wb') as f: 
    pk.dump([embs, lang_labels], f)
