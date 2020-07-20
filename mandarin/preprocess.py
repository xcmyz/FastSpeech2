import torch
import os
import shutil
import numpy as np
import hparams as hp

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from data.bznsyp import process
from frontend.get_pinyin import get_pyin
from frontend.parse_interval import parse_interval


def preprocess_bznsyp():
    executor = ProcessPoolExecutor(max_workers=4)
    futures = []

    os.makedirs(hp.mel_path, exist_ok=True)
    data_path = os.path.join(os.path.join("data", "BZNSYP"), "PhoneLabeling")
    file_list = os.listdir(data_path)
    for file_name in file_list:
        info_dict = parse_interval(os.path.join(data_path, file_name))
        futures.append(executor.submit(partial(process, info_dict)))
    return [future.result() for future in tqdm(futures)]


if __name__ == "__main__":
    idxs = preprocess_bznsyp()
    with open(os.path.join("data", "phone_idx.txt"), "w", encoding="utf-8") as f:
        for idx in idxs:
            f.write(idx[0]+"\n")
    with open(os.path.join("data", "duration_idx.txt"), "w", encoding="utf-8") as f:
        for idx in idxs:
            f.write(idx[1]+"\n")
