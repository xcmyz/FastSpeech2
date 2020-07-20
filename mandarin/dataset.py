from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from utils import pad_1D_tensor, pad_2D_tensor
from tqdm import tqdm

import os
import torch
import math
import time
import audio
import utils
import hparams as hp
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_to_buffer():
    buffer = list()
    phone_idx = dict()
    duration_idx = dict()

    with open(os.path.join("data", "phone_idx.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("|")
            file_id = line_split[0]
            p_idx = line_split[1][:-1].split(" ")
            for i, phone in enumerate(p_idx):
                p_idx[i] = int(phone)
            phone_idx.update({file_id: p_idx})

    with open(os.path.join("data", "duration_idx.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("|")
            file_id = line_split[0]
            d_idx = line_split[1][:-1].split(" ")
            for i, duration in enumerate(d_idx):
                d_idx[i] = int(duration)
            duration_idx.update({file_id: d_idx})

    start = time.perf_counter()
    for key in tqdm(phone_idx):
        mel_file_path = os.path.join(hp.mel_path, key+".npy")
        mel = np.load(mel_file_path)
        phone = phone_idx[key]
        duration = duration_idx[key]
        len_dur = len(duration)
        duration = utils.parse_duration(duration, phone)
        if len_dur != np.sum(np.array(duration)):
            raise Exception("data processing error")
        buffer.append({"text": torch.Tensor(np.array(phone)),
                       "duration": torch.Tensor(np.array(duration)),
                       "mel_target": torch.Tensor(mel)})
    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))
    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hp.batch_expand_size

    cut_list = list()
    for i in range(hp.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(hp.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
