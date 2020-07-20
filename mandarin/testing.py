import os
import hparams as hp

# from data.bznsyp import process_utterance
from data.bznsyp import process
from frontend.parse_interval import parse_interval
from utils import get_phone_map

if __name__ == "__main__":
    data_path = os.path.join(os.path.join("data", "BZNSYP"), "PhoneLabeling")
    file_list = os.listdir(data_path)
    info_list = list()

    print(get_phone_map())

    os.makedirs(hp.mel_path, exist_ok=True)

    for file_name in file_list:
        # info_dict = parse_interval(os.path.join(data_path, file_name))
        # print(info_dict)
        # phone_idx, duration_idx = process(info_dict)
        # print(phone_idx)
        # print(len(duration_idx.split("|")[1].split(" ")))
        # print()
        # phone_idx = process_utterance(info_dict)
        # print(phone_idx.split("|")[0], len(phone_idx.split("|")[1].split(" ")))
        # print(phone_idx.split("|")[1].split(" ")[-1])
        # print(parse_interval(info_dict))
        pass
