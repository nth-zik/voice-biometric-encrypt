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


def float_to_bin(value):
    """Chuyển đổi float sang chuỗi nhị phân theo chuẩn IEEE 754."""
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return f"{d:064b}"


def embedding_to_binary(embedding_np):
    """Chuyển đổi toàn bộ embedding thành chuỗi nhị phân."""
    return "".join(float_to_bin(x) for x in embedding_np.ravel())


def binary_to_simhash(binary_string):
    """Chuyển đổi chuỗi nhị phân thành một mã SimHash."""
    features = [binary_string[i : i + 8] for i in range(0, len(binary_string), 8)]
    return Simhash(features)


def simhash_to_hex_string(simhash):
    """Chuyển đổi SimHash thành chuỗi thập lục phân."""
    return hex(simhash.value)[2:].zfill(16)


def save_simhash_to_file(simhash, filename):
    """Lưu SimHash xuống file dưới dạng chuỗi thập lục phân."""
    with open(filename, "w") as f:
        f.write(simhash_to_hex_string(simhash))


def load_simhash_from_file(filename):
    """Đọc SimHash từ file và chuyển đổi lại thành đối tượng SimHash."""
    with open(filename, "r") as f:
        hex_value = f.read().strip()
        simhash_value = int(hex_value, 16)
        return Simhash(simhash_value)


def embedding_to_simhash(embedding_np):
    """Chuyển đổi embedding thành một mã SimHash."""
    # Chuyển đổi embedding thành một danh sách các tính năng (danh sách các hash giá trị)
    embedding_flat = embedding_np.ravel()
    features = [(str(i), float(x)) for i, x in enumerate(embedding_flat)]
    return Simhash(features)


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
    return sign_bit + binary_representation


def vector_to_binary_hex(vector):
    """Chuyển vector đặc trưng sang binary rồi chuyển sang hexadecimal"""
    # Chuyển đổi vector thành binary
    binary_representation = "".join(
        float_to_binary_with_sign(x) for x in vector.ravel()
    )

    # Chuyển đổi từ binary sang hexadecimal
    hex_representation = hex(int(binary_representation, 2))[2:]  # Bỏ '0x' của hex
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
    # binary_representation1 = vector_to_binary_hex(embedding)
    # simhash1 = binary_to_simhash(binary_representation1)
    simhash1 = embedding_to_simhash(embedding.numpy())
    save_simhash_to_file(simhash1, "simhash1.txt")
    print("SimHash has been saved to simhash1.txt")

    # Đọc lại SimHash từ file
    loaded_simhash = load_simhash_from_file("simhash1.txt")
    print(f"Loaded SimHash: {loaded_simhash.value}")

    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        # fn="test_data/id01_2.wav",
        fn="test_data/00004_old.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    # giải mã và so sánh
    # binary_representation2 = vector_to_binary_hex(embedding2)
    # simhash2 = binary_to_simhash(binary_representation2)
    simhash2 = embedding_to_simhash(embedding2.numpy())

    hamming_dist = loaded_simhash.distance(simhash2)
    print(simhash1.value)
    print(simhash2.value)
    # print(f"Total len {simhash1.batch_size}")
    print(f"Hamming Distance Raw {simhash1.distance(simhash2)}")
    print(f"Hamming Distance from file {hamming_dist}")
    with open("embeddingraw1", "w") as f:
        f.write(str(embedding.numpy().tolist()))
    with open("embeddingraw2", "w") as f:
        f.write(str(embedding2.numpy().tolist()))
    # So sánh gốc
    print("Original embedding")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first_outputs = []
    for i in range(0, n_segments):
        output = cos(embedding[i], embedding2[i])
        first_outputs.append(float(output))
    print(first_outputs)

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
