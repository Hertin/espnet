import os
import numpy as np
import json
from ipywidgets import interact
import matplotlib.pyplot as plt
import torch
import random
import re
from tqdm import tqdm
from espnet.asr.pytorch_backend.asr import load_trained_model

import matplotlib
matplotlib.use('TkAgg')


i = 30
model, train_args = load_trained_model(f'exp/train_pytorch_wav2vecfex/results/snapshot.ep.{i}')
device = torch.device('cuda')
model = model.float()
model = model.to(device)

with open('dump/train/deltafalse/data.json.npy') as f:
    js = json.load(f)['utts']
    
def get_lang(d):
    s = d.split('_')[0]
    s = re.sub(r'\d+$', '', s.split('-')[0]) if re.search('[a-zA-Z]+', s) else s
    return s

def to_onehot(ys):
    uniq = sorted(list(np.unique(ys)))
    
    out = np.zeros((len(ys), len(uniq)))
    l2int = {l: i for i, l in enumerate(uniq)}
    print(l2int)
    for i, l in enumerate(ys):
        out[i, l2int[l]] = 1
    return out

def to_int(ys):
    uniq = sorted(list(np.unique(ys)))
    
#     out = np.zeros((len(ys), len(uniq)))
    l2int = {l: i for i, l in enumerate(uniq)}
    print(l2int)
    out = np.array([l2int[l] for i, l in enumerate(ys)])
    return out
n_sample = 1000
frame_ratio = 0.1

xs = []
ys = []
random.seed(1)
with torch.no_grad():
    for k, v in tqdm(random.sample(js.items(), n_sample)):
        lang = get_lang(k)
        x = torch.FloatTensor(np.load(v['input'][0]['feat'])).unsqueeze(0).to(device)
        features = model.feature_extractor(x).transpose(1, 2)
        
        features = features.squeeze(0).detach().cpu().numpy()
        T, d = features.shape
        n_frame = int(frame_ratio * T)
        ys.extend([lang] * n_frame)
        idx = random.sample(list(range(T)), n_frame)
        xs.append(features[idx])
#         break
    xs = np.vstack(xs)
    
    
X = np.array(xs)
Y = to_int(ys)
print(X.shape, Y.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clf = LinearSVC(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc)

