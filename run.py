import argparse
import itertools
import os
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck
from utils import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

import torch.nn as nn

print("torch.cuda.is_available()", torch.cuda.is_available())


def main():
    n_segments = 5
    n_samples = 16000
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    gpu = False

    model.load_state_dict(
        torch.load(
            "./models/weights/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")
    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    embedding = extract_speaker_embd(
        model,
        fn="test_data/id01_1.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )

    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for i in range(0, n_segments):
        output = cos(embedding[i], embedding2[i])
        output = float(output)
        print(output)


def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
        )

    if len(audio) < n_samples:  # RawNet3 was trained using utterances of 3 seconds
        shortage = n_samples - len(audio) + 1
        audio = np.pad(audio, (0, shortage), "wrap")

    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf) : int(asf) + n_samples])

    audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
    if gpu:
        audios = audios.to("cuda")
    with torch.no_grad():
        output = model(audios)

    return output


if __name__ == "__main__":
    main()
