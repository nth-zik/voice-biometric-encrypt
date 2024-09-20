import argparse
import itertools
import os
import sys
import hashlib
import struct
import graycode
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
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from sklearn.metrics.pairwise import cosine_similarity

print("torch.cuda.is_available()", torch.cuda.is_available())


DECIMAL_KEEP = 10**8  # Hằng số để loại bỏ phần thập phân

def float_to_gray_with_sign(value):
    """Chuyển đổi float thành Gray code với 1 bit dấu và 63 bit giá trị."""
    sign_bit = '0' if value >= 0 else '1'  # 0 cho dương, 1 cho âm
    abs_value = abs(value) * DECIMAL_KEEP  # Nhân với hằng số để loại bỏ phần thập phân
    int_value = int(abs_value)  # Chuyển thành số nguyên
    gray_value = graycode.tc_to_gray_code(int_value)  # Chuyển thành Gray code
    value_63bit = "{:063b}".format(gray_value)  # Chuyển thành chuỗi nhị phân với 63 bit
    return sign_bit + value_63bit  # 1 bit dấu + 63 bit giá trị Gray code

def embedding_to_gray_with_sign(embedding_np):
    """Chuyển đổi toàn bộ embedding thành chuỗi Gray code với 63 bit giá trị và 1 bit dấu."""
    return "".join(float_to_gray_with_sign(x) for x in embedding_np.ravel())


def pad_hex_strings(bin1, bin2):
    """Hàm bổ sung để đảm bảo hai chuỗi nhị phân có cùng độ dài."""
    max_len = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_len)
    bin2 = bin2.zfill(max_len)
    return bin1, bin2

def hamming_distance(hex1, hex2):
    """Tính khoảng cách Hamming giữa hai chuỗi hexadecimal."""
    # Convert hex to binary strings
    bin1 = bin(int(hex1, 16))[2:]
    bin2 = bin(int(hex2, 16))[2:]
    
    # Ensure both binary strings have the same length
    bin1, bin2 = pad_hex_strings(bin1, bin2)

    # Compute Hamming distance
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

def cosine_similarity_hex(hex1, hex2):
    """Tính cosine similarity giữa hai chuỗi hex."""
    # Convert hex to binary strings
    bin1 = bin(int(hex1, 16))[2:]
    bin2 = bin(int(hex2, 16))[2:]
    
    # Ensure equal length by padding
    bin1, bin2 = pad_hex_strings(bin1, bin2)
    
    # Convert to integer vectors (list of 0s and 1s)
    vec1 = np.array([int(bit) for bit in bin1]).reshape(1, -1)
    vec2 = np.array([int(bit) for bit in bin2]).reshape(1, -1)
    
    # Calculate cosine similarity
    return cosine_similarity(vec1, vec2)[0][0]

def euclidean_distance_hex(hex1, hex2):
    """Tính khoảng cách Euclidean giữa hai chuỗi hex."""
    bin1 = bin(int(hex1, 16))[2:]
    bin2 = bin(int(hex2, 16))[2:]
    
    # Đảm bảo độ dài bằng nhau bằng cách đệm thêm các số 0
    bin1, bin2 = pad_hex_strings(bin1, bin2)
    
    # Tính khoảng cách Euclidean
    return sum((int(c1) - int(c2))**2 for c1, c2 in zip(bin1, bin2))**0.5

def euclidean_distance(vec1, vec2):
    """Tính khoảng cách Euclidean giữa hai vector."""
    return np.linalg.norm(vec1 - vec2)

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
    # if torch.cuda.is_available():
    #     print("Cuda available, conducting inference on GPU")
    #     model = model.to("cuda")
    #     gpu = True

    embedding = extract_speaker_embd(
        model,
        fn="test_data/id01_1.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )

    # Mã hóa và lưu tạm xuống file (xem như db)
    embedding_np = embedding.numpy()
    print(len(embedding_np.ravel()))
    binary_representation = embedding_to_gray_with_sign(embedding_np)
    hex1 = hex(int(binary_representation, 2))
    
    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    print(len(embedding2.ravel()))
    
    binary_representation_2 = embedding_to_gray_with_sign(embedding2.numpy())
    hex2 = hex(int(binary_representation_2, 2))
    
    hex1, hex2 = pad_hex_strings(hex1[2:], hex2[2:])  
    
    distance = hamming_distance(hex1[2:], hex2[2:])
    print(f"Hamming Distance: {distance}/{len(hex1)}  {len(hex1)} {len(hex2)}")
    print(f"cosine similarity: {cosine_similarity_hex(hex1, hex2)}")
    # Euclidean raw
    euclidean_distances = []
    for i in range(n_segments):
        dist = euclidean_distance(embedding[i].numpy(), embedding2[i].numpy())
        euclidean_distances.append(dist)
    
    print(f"Khoảng cách Euclidean giữa hai embedding gốc: {euclidean_distances}")
    
    print(f"Euclidean : {euclidean_distance_hex(hex1, hex2)}")
    # So sánh gốc
    print("Original embedding")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first_outputs = []
    for i in range(0, n_segments):
        output = cos(embedding[i], embedding2[i])
        first_outputs.append(float(output))
    print(first_outputs)



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
