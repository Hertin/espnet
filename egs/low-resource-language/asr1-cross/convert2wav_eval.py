import os
import re
import json
import time
import librosa
import subprocess
import shlex
import torch
import fairseq
import numpy as np
import soundfile as sf
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing


eval_folders = [f for f in os.listdir('data') if 'eval' in f]

for eval_folder in eval_folders:
    print('#'*50, eval_folder)
    data_dir = 'data'
    wav_scp = f'{data_dir}/{eval_folder}/wav.scp'
    audio_folder = 'w2vaudio_wav'
    feat_folder = 'w2vaudio_npy'
    seen_utt = set()
    def line_count(filename):
        return int(subprocess.check_output(['wc', '-l', filename]).split()[0])
    nline = line_count(wav_scp)

    lines = []
    with open(wav_scp, 'r') as f:
        for l in tqdm(f, total=nline):
            lines.append(l)

    def process(i, l):
        if i % 100 == 0:
            print(i, 'th line')
        uttid, cmd = l.strip().strip('- |').split(' ', maxsplit=1)
        audio_path = f'{audio_folder}/{uttid}.wav'
        cmd = f' wav {audio_path} '.join(cmd.rsplit(' wav - ', maxsplit=1))

        if not os.path.isfile(audio_path):
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if stderr:
                if 'WARN' in str(stderr):
                    pass
                else:
                    assert False

    num_cores = multiprocessing.cpu_count() - 2
    print('num_cores', num_cores)

    results = Parallel(n_jobs=num_cores)(delayed(process)(i, l) for i, l in enumerate(lines))

seen_utt = set()
for eval_folder in eval_folders:
    print(eval_folder)
    import time
    time.sleep(1)
    segment_file = f'{data_dir}/{eval_folder}/segments'
    nline = line_count(segment_file)
    sr = 16000
    # seen_utt = set()
    with open(segment_file, 'r') as f:
        for l in tqdm(f, total=nline):
            segid, audioid, start, end = l.split(' ')
            start, end = float(start), float(end)
            audio_path = f'{audio_folder}/{audioid}.wav'
            feat_path = f'{feat_folder}/{segid}.npy'
            if not os.path.isfile(feat_path):
                feats, curr_sample_rate = librosa.load(audio_path, sr=sr, offset=start, duration=end-start)
                assert curr_sample_rate == sr
                if len(feats.shape) == 2:
                    feats = feats.mean(-1)
                    assert len(feats) != 2
                np.save(feat_path, feats)

            assert segid not in seen_utt
            seen_utt.add(segid)
 
 def convert_json_data(input_json_path):
    
    output_json_path = f'{input_json_path}.npy'
    if os.path.isfile(output_json_path):
        print(f'{output_json_path} already exists, skip')
        return
    with open(input_json_path, 'r') as fin:
        js = json.load(fin)["utts"]
        for uttid, v in tqdm(js.items(), total=len(js)):
            npy_file = f'{feat_folder}/{uttid}.npy'
            assert 'shape' in v['input'][0]
            assert 'feat' in v['input'][0]
            v['input'][0]['feat'] = npy_file
            v['input'][0]['filetype'] = 'npy'
            shape = np.load(npy_file, mmap_mode='r').shape[0]
            if type(shape) is int:
                shape =(shape, 1)
            elif len(shape) == 1:
                shape = (shape[0], 1)
            v['input'][0]['shape'] = shape
        print(f'saving modified npy json {output_json_path}')
        with open(output_json_path, 'w') as fout:
            json.dump({'utts': js}, fout, indent=2)
    
eval_folders = [f for f in os.listdir('data') if 'eval' in f]
feat_folder = 'w2vaudio_npy'
for eval_folder in eval_folders:
    input_json_path = f'dump/{eval_folder}/deltafalse/data.json'
    convert_json_data(input_json_path)
