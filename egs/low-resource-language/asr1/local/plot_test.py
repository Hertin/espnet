#!/usr/bin/env python3
# encoding: utf-8

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(
        description="plot recog pter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-I", dest="plot_dir", type=str, help="Input result folder")
    parser.add_argument("-O", dest="plot_save_pth", type=str, help="Output plot")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    data = []
    for folder in os.listdir(args.plot_dir):
        if folder.startswith('snapshot.ep.'):
            ep = int(folder.strip('snapshot.ep.'))
            with open(f'{args.plot_dir}/{folder}/result.txt', 'r', encoding="utf-8") as f:
                for l in f:
                    if 'Sum/Avg' in l:
                        pter = float(l.strip().strip('|').strip().split()[-2])
                        data.append((ep, pter))
                        break

    data = sorted(data, key=lambda x: x[0])
    
    eps, pters = zip(*data)
    minid = np.argmin(pters)
    print(f'min cer occurs at ep {eps[minid]} cer {pters[minid]}')

    plt.figure(dpi=100)
    plt.plot(eps, pters)
    plt.title('test pter')
    plt.grid()
    plt.savefig(args.plot_save_pth)
