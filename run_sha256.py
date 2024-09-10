import argparse
import itertools
import os
import sys
import hashlib
import struct
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
from simhash import Simhash

print("torch.cuda.is_available()", torch.cuda.is_available())


def sha256_hash_hex(hex_string):
    """Băm chuỗi hexadecimal bằng SHA-256"""
    return hashlib.sha256(hex_string.encode("utf-8")).hexdigest()


def hamming_distance(bin_str1, bin_str2):
    """Tính khoảng cách Hamming giữa hai chuỗi nhị phân"""
    return sum(c1 != c2 for c1, c2 in zip(bin_str1, bin_str2))


def float_to_binary_with_sign(value):
    """Chuyển số float sang chuỗi binary, thêm bit lưu dấu"""
    # Chuyển giá trị float thành chuỗi nhị phân
    if value >= 0:
        sign_bit = "0"  # Thêm 0 nếu số dương
    else:
        sign_bit = "1"  # Thêm 1 nếu số âm
        value = -value  # Lấy giá trị tuyệt đối cho phần nhị phân
    # Chuyển đổi phần giá trị tuyệt đối thành 64-bit nhị phân
    binary_representation = format(
        struct.unpack("!Q", struct.pack("!d", value))[0], "064b"
    )
    # Thêm bit dấu vào trước
    return binary_representation


def vector_to_binary_hex(vector):
    """Chuyển vector đặc trưng sang binary rồi chuyển sang hexadecimal"""
    # Chuyển đổi vector thành binary
    binary_representation = "".join(
        float_to_binary_with_sign(x) for x in vector.ravel()
    )
    import sys
    sys.set_int_max_str_digits(0)
    print(f'int(binary_representation, 2) {int(binary_representation, 2)}')
    # Chuyển đổi từ binary sang hexadecimal
    hex_representation = hex(int(binary_representation, 2))[2:]  # Bỏ '0x' của hex
    print(hex_representation)
    return hex_representation


def main():
    n_segments = 10
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
    # simhash1 = vector_to_binary_hex(binary_representation1)
    hash1 = sha256_hash_hex(vector_to_binary_hex(embedding.numpy()))

    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        # fn="test_data/00004_old.wav",
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    hash2 = sha256_hash_hex(vector_to_binary_hex(embedding2.numpy()))

    hamming_dist = hamming_distance(hash1, hash2)
    print(f"len hash {len(hash1)}")
    print(f"hash1 {hash1}")
    print(f"hash2 {hash2}")
    print(f"Hamming Distance {hamming_dist} {hamming_dist/len(hash1)}")
    with open("embeddingraw1", "w") as f:
        f.write(str(embedding.numpy().tolist()))
    with open("embeddingraw2", "w") as f:
        f.write(str(embedding2.numpy().tolist()))
    # So sánh gốc
    print("Original embedding")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first_outputs = []
    # for i in range(0, n_segments):
    #     output = cos(embedding[i], embedding2[i])
    #     first_outputs.append(float(output))
    # print(first_outputs)
    print(float(cos(embedding[0], embedding2[0])))

    # print("encrypt embedding")
    # second_outputs = []
    # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # for i in range(0, n_segments):
    #     original_embedding_tensor = torch.tensor(
    #         original_embedding_np[i], dtype=torch.float32
    #     )
    #     output = cos(original_embedding_tensor, embedding2[i])
    #     second_outputs.append(float(output))
    # print(second_outputs)
    # print(np.array(second_outputs) / np.array(first_outputs))
    # with open("embeddingraw_afteraes.txt", "w") as f:
    #     f.write(str(original_embedding_np.tolist()))


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

    # if len(audio) < n_samples:  # RawNet3 was trained using utterances of 3 seconds
    #     shortage = n_samples - len(audio) + 1
    #     audio = np.pad(audio, (0, shortage), "wrap")

    # audios = []
    # startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    # for asf in startframe:
    #     audios.append(audio[int(asf) : int(asf) + n_samples])

    # audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
    audios = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if gpu:
        audios = audios.to("cuda")
    with torch.no_grad():
        output = model(audios)

    return output


if __name__ == "__main__":
    main()
