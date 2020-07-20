import os
import audio
import numpy as np
import hparams as hp

from frontend.parse_interval import parse_interval
from utils import get_phone_map

phone_map = get_phone_map()


def process(info_dict):
    wav_path = os.path.join(hp.data_path, "Wave")
    wav_file_name = os.path.join(wav_path, info_dict["sentence_id"]+".wav")
    wav = audio.load_wav(wav_file_name)
    mel = audio.melspectrogram(wav).T
    mel_file_path = os.path.join(hp.mel_path, info_dict["sentence_id"]+".npy")
    np.save(mel_file_path, mel)

    phone_idx = info_dict["sentence_id"] + "|"
    for phone_duration in info_dict["alignment"]:
        phone_idx += str(phone_map[phone_duration[0]]) + " "

    duration_idx = info_dict["sentence_id"] + "|"
    length_mel = mel.shape[0]
    length_phone_list = len(info_dict["alignment"])
    cur_pointer = 0
    for frame_id in range(length_mel):
        added = False
        cur_time = hp.frame_length_ms / 2 + frame_id * hp.frame_shift_ms
        cur_time = cur_time / 1000.0
        for i in range(cur_pointer, length_phone_list):
            if cur_time >= info_dict["alignment"][i][1][0] and cur_time < info_dict["alignment"][i][1][1]:
                phone_id = phone_map[info_dict["alignment"][i][0]]
                duration_idx += str(phone_id) + " "
                cur_pointer = i
                added = True
                break
        if not added:
            phone_id = phone_map[info_dict["alignment"][cur_pointer][0]]
            duration_idx += str(phone_id) + " "

    return phone_idx[:-1], duration_idx[:-1]
