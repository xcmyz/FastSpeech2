import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os
import hparams as hp
import audio
import utils
import dataset
import model as M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    test1 = [172, 190, 39, 189, 249, 141, 125, 126, 180, 257, 254,
             90, 48, 123, 231, 141, 39, 141, 250, 78, 91, 115, 172]
    test2 = [172, 71, 117, 52, 53, 13, 135, 179, 136, 252, 245, 7, 71, 101,
             83, 257, 225, 245, 7, 257, 127, 71, 101, 83, 257, 225, 13, 231, 172]
    test3 = [172, 237, 27, 136, 249, 141, 246, 233, 135, 245, 251, 245, 251,
             213, 4, 213, 4, 80, 231, 123, 135, 237, 154, 90, 101, 136, 249, 172]
    test4 = [172, 80, 191, 13, 186, 81, 162, 91, 198, 80, 231, 128, 80, 38, 180,
             141, 37, 24, 80, 125, 80, 125, 91, 178, 260, 226, 245, 149, 35, 132, 172]
    test5 = [172, 213, 111, 190, 1, 233, 215, 35,
             246, 24, 257, 126, 81, 5, 118, 233, 27, 172]
    test6 = [172, 213, 204, 80, 21, 190, 24, 136, 69, 147, 23, 260, 7, 1, 180, 45, 86, 170, 71,
             115, 73, 185, 136, 203, 142, 52, 17, 136, 203, 180, 257, 72, 260, 64, 123, 132, 123, 60, 172]
    data_list = [test1, test2, test3, test4, test5, test6]
    return data_list


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    print("use griffin-lim")
    model = get_DNN(args.step)
    data_list = get_data()
    for i, phn in enumerate(data_list):
        mel, _ = synthesis(model, phn, args.alpha)
        if not os.path.exists("results"):
            os.mkdir("results")
        wav = audio.inv_mel_spectrogram(mel)
        audio.save_wav(wav, "results/"+str(args.step)+"_"+str(i)+".wav")
        print("Done", i + 1)
