#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""pytorch dataset and dataloader implementation for chainer training."""

import torch
import torch.utils.data
import logging
import numpy as np
import string
import re
import torch.nn.functional as F


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform
        # logging.warning(f'TransformDataset {data[0]}')
        # logging.warning(f'TransformDataset {self.transform}')

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        xs_pad, ilens, ys_pad = self.transform(self.data[idx])
        # logging.warning(f"TransformDataset __getitem__ {idx} [total] {len(self.data[idx])}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [0] {self.data[idx][0]}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [1] {self.data[idx][1]}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [lang] {[d[0].split('_')[0] for d in self.data[idx]]}")
        # logging.warning(f'TransformDataset __getitem__ {idx} {xs_pad.size(), ilens.size(), ys_pad.size()}')
        # raise 
        return self.transform(self.data[idx])


class TransformDatasetRandomFlip(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform
        # logging.warning(f'TransformDataset {data[0]}')
        # logging.warning(f'TransformDataset {self.transform}')

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        xs_pad, ilens, ys_pad = self.transform(self.data[idx])
        # logging.warning(f"TransformDataset __getitem__ {idx} [total] {len(self.data[idx])}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [0] {self.data[idx][0]}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [1] {self.data[idx][1]}")
        # logging.warning(f"TransformDataset __getitem__ {idx} [lang] {[d[0].split('_')[0] for d in self.data[idx]]}")
        # logging.warning(f'TransformDataset __getitem__ {idx} {xs_pad.size(), ilens.size(), ys_pad.size()}')
        # raise 
        return self.transform(self.data[idx])


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform, utt=False, lang=False, lang_onehot=False, num_langs=None):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform
        self.utt, self.lang, self.lang_onehot = utt, lang, lang_onehot
        if self.lang or self.lang_onehot:
            self.all_lang = set()
            for dt in self.data:
                self.all_lang.update([self.get_lang(d) for d in dt])
            from collections import Counter
            
            cnt = Counter()
            for dt in self.data:
                cnt.update([d[0].split('_')[0] for d in dt])
            # logging.warning(f'TransformDataset [counter] {cnt}')

            self.num_langs = num_langs
            self.lang2int = {l: i for i, l in enumerate(sorted(self.all_lang))}
            self.int2lang = {i: l for l, i in self.lang2int.items()}
            logging.warning(f'TransformDatasetEar [all_lang] {self.all_lang}')
            logging.warning(f'TransformDatasetEar [lang2int] {self.lang2int}')
        # logging.warning(f'TransformDataset {data[0]}')
        # logging.warning(f'TransformDataset {self.transform}')

    def __len__(self):
        """Len function."""
        return len(self.data)

    def get_lang(self, d):
        s = d[0].split('_')[0]
        s = re.sub(r'\d+$', '', s.split('-')[0]) if re.search('[a-zA-Z]+', s) else s
        return s

    def __getitem__(self, idx):
        """[] operator."""
        return self.custom_transform(self.data[idx])
        # items = []
        # xs_pad, ilens, ys_pad = self.transform(self.data[idx])
        # if self.lang:
        #     langs = torch.from_numpy(np.array([
        #             self.lang2int[self.get_lang(d)] for d in self.data[idx]
        #     ])).long()
        #     items.append(langs)
        # if self.lang_onehot:
        #     langs_onehot = F.one_hot(torch.from_numpy(np.array([
        #         self.lang2int[self.get_lang(d)] for d in self.data[idx]
        #     ])), num_classes=self.num_langs).float()
        #     items.append(langs_onehot)
        # if self.utt:
        #     utts = [d[0] for d in self.data[idx]]
        #     items.append(utts)
        # items.extend([xs_pad, ilens, ys_pad])

        # return tuple(items)

    def custom_transform(self, data):
        items = []
        xs_pad, ilens, ys_pad = self.transform(data)
        if self.lang:
            langs = torch.from_numpy(np.array([
                    self.lang2int[self.get_lang(d)] for d in data
            ])).long()
            items.append(langs)
        if self.lang_onehot:
            langs_onehot = F.one_hot(torch.from_numpy(np.array([
                self.lang2int[self.get_lang(d)] for d in data
            ])), num_classes=self.num_langs).float()
            items.append(langs_onehot)
        if self.utt:
            utts = [d[0] for d in data]
            items.append(utts)
        items.extend([xs_pad, ilens, ys_pad])
        return tuple(items)

class TransformDatasetEar(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """
    def get_lang(self, d):
        s = d[0].split('_')[0]
        s = re.sub(r'\d+$', '', s.split('-')[0]) if re.search('[a-zA-Z]+', s) else s
        return s

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

        self.all_lang = set()
        for dt in self.data:
            self.all_lang.update([self.get_lang(d) for d in dt])
            # ll = []
            # for d in dt:
                
            #     s = d[0].split('_')[0]
            #     s = re.sub(r'\d+$', '', s.split('-')[0]) if re.search('[a-zA-Z]+', s) else s
            #     ll.append(s)
            # self.all_lang.update(ll)
            #     [
            #     re.sub(r'\d+$', '', d[0].split('_')[0]) if re.search('[a-zA-Z]+',d[0].split('_')[0]) else d[0].split('_')[0]
            #     for d in dt
            # ]
        from collections import Counter
        
        cnt = Counter()
        for dt in self.data:
            cnt.update([d[0].split('_')[0] for d in dt])
        # logging.warning(f'TransformDataset [counter] {cnt}')

        self.lang2int = {l: i for i, l in enumerate(sorted(self.all_lang))}
        self.int2lang = {i: l for l, i in self.lang2int.items()}
        logging.warning(f'TransformDatasetEar [all_lang] {self.all_lang}')
        logging.warning(f'TransformDatasetEar [lang2int] {self.lang2int}')
        # logging.warning(f'TransformDatasetEar [int2lang] {self.int2lang}')
        # logging.warning(f'TransformDatasetEar {self.transform}')

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        xs_pad, ilens, ys_pad = self.transform(self.data[idx])
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [total] {len(self.data[idx])}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [0] {self.data[idx][0]}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [1] {self.data[idx][1]}")
        lang = torch.from_numpy(np.array(
            [
                self.lang2int[self.get_lang(d)] for d in self.data[idx]
            ]
        )).long()

        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [lang] len {len(lang)} {lang}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [lang] {[d[0].split('_')[0] for d in self.data[idx]]}")
        # logging.warning(f'TransformDatasetEar __getitem__ {idx} {xs_pad.size(), ilens.size(), ys_pad.size()}') 
        return lang, xs_pad, ilens, ys_pad

class TransformDatasetEval(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """
    def get_lang(self, d):
        s = d[0].split('_')[0]
        s = re.sub(r'\d+$', '', s.split('-')[0]) if re.search('[a-zA-Z]+', s) else s
        return s

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

        self.all_lang = set()
        for dt in self.data:
            self.all_lang.update([self.get_lang(d) for d in dt])

        from collections import Counter
        
        cnt = Counter()
        for dt in self.data:
            cnt.update([d[0].split('_')[0] for d in dt])
        # logging.warning(f'TransformDataset [counter] {cnt}')

        self.lang2int = {l: i for i, l in enumerate(sorted(self.all_lang))}
        self.int2lang = {i: l for l, i in self.lang2int.items()}
        # logging.warning(f'TransformDatasetEar [all_lang] {self.all_lang}')
        # logging.warning(f'TransformDatasetEar [lang2int] {self.lang2int}')
        # logging.warning(f'TransformDatasetEar [int2lang] {self.int2lang}')
        # logging.warning(f'TransformDatasetEar {self.transform}')

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        xs_pad, ilens, ys_pad = self.transform(self.data[idx])
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [total] {len(self.data[idx])}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [0] {self.data[idx][0]}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [1] {self.data[idx][1]}")
        # lang = torch.from_numpy(np.array(
        #     [
        #         self.lang2int[d[0].split('_')[0]] for d in self.data[idx]
        #     ]
        # )).long()
        utts = [d[0] for d in self.data[idx]]
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [lang] {lang}")
        # logging.warning(f"TransformDatasetEar __getitem__ {idx} [lang] {[d[0].split('_')[0] for d in self.data[idx]]}")
        # logging.warning(f'TransformDatasetEar __getitem__ {idx} {xs_pad.size(), ilens.size(), ys_pad.size()}') 
        return utts, xs_pad, ilens, ys_pad

class ChainerDataLoader(object):
    """Pytorch dataloader in chainer style.

    Args:
        all args for torch.utils.data.dataloader.Dataloader

    """

    def __init__(self, **kwargs):
        """Init function."""
        self.loader = torch.utils.data.dataloader.DataLoader(**kwargs)
        self.len = len(kwargs["dataset"])
        self.current_position = 0
        self.epoch = 0
        self.iter = None
        self.kwargs = kwargs

    def next(self):
        """Implement next function."""
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch = self.epoch + 1
            self.current_position = 0
        return ret

    def synchronize(self, epoch):
        self.current_position = 0
        self.epoch = epoch

    def __iter__(self):
        """Implement iter function."""
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        """Epoch_detail required by chainer."""
        return self.epoch + self.current_position / self.len

    def serialize(self, serializer):
        """Serialize and deserialize function."""
        epoch = serializer("epoch", self.epoch)
        current_position = serializer("current_position", self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        """Shuffle function for sortagrad."""
        self.kwargs["shuffle"] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        """Implement finalize function."""
        del self.loader
