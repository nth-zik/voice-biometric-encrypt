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

print("torch.cuda.is_available()", torch.cuda.is_available())


def float_to_bin(value):
    """Chuyển đổi float sang chuỗi nhị phân theo chuẩn IEEE 754."""
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return f"{d:064b}"


def embedding_to_binary(embedding_np):
    """Chuyển đổi toàn bộ embedding thành chuỗi nhị phân."""
    return "".join(float_to_bin(x) for x in embedding_np.ravel())


def hash_binary(binary_string):
    """Tạo giá trị băm SHA-256 từ chuỗi nhị phân."""
    binary_bytes = int(binary_string, 2).to_bytes(
        (len(binary_string) + 7) // 8, byteorder="big"
    )
    sha256_hash = hashlib.sha256(binary_bytes).hexdigest()
    return sha256_hash


def hamming_distance(hash1, hash2):
    """Tính khoảng cách Hamming giữa hai giá trị băm"""
    bin1 = bin(int(hash1, 16))[2:].zfill(256)
    bin2 = bin(int(hash2, 16))[2:].zfill(256)
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


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

    DECIMAL_KEEP = 10**8

    # Mã hóa và lưu tạm xuống file (xem như db)
    embedding_np = embedding.numpy()
    # save raw embedding to file
    with open("embeddingraw.txt", "w") as f:
        f.write(str(embedding_np.tolist()))

    signs = np.sign(embedding_np)  # Lưu trữ dấu (-1 hoặc 1)
    original_shape = embedding_np.shape
    abs_embedding = np.abs(embedding_np)  # Lấy giá trị tuyệt đối
    # chuyển sang số nguyên và giữ 10 chữ số thập phân
    integer_representation = (abs_embedding * DECIMAL_KEEP).astype(int)
    ravel_length = len(integer_representation.ravel())

    # Chuyển mỗi giá trị float sang nhị phân
    binary_representation = "".join(
        format(x, "064b") for x in integer_representation.ravel()
    )

    # Mã hóa chuỗi nhị phân với AES

    # Chuyển đổi từng phần sang số nguyên và mã hóa bằng AES
    key = b"lEI9KdGi2j!48XSi"
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = b""

    binary_data = int(binary_representation, 2).to_bytes(
        (len(binary_representation) + 7) // 8, byteorder="big"
    )
    ciphertext += cipher.encrypt(pad(binary_data, AES.block_size))

    # Lưu IV (Initialization Vector) và ciphertext xuống file
    iv = cipher.iv
    with open("encrypted_voice_data.bin", "wb") as f:
        f.write(np.array(original_shape, dtype=np.int32).tobytes())
        f.write(np.array(ravel_length, dtype=np.int32).tobytes())
        f.write(signs.tobytes())
        f.write(iv)
        f.write(ciphertext)

    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    # giải mã và so sánh

    with open("encrypted_voice_data.bin", "rb") as f:
        stored_original_shape = tuple(np.frombuffer(f.read(8), dtype=np.int32))
        stored_ravel_length = int(np.frombuffer(f.read(4), dtype=np.int32))
        stored_signs = np.frombuffer(
            f.read(embedding_np.size * signs.itemsize), dtype=signs.dtype
        )  # Đọc dấu từ tệp
        stored_iv = f.read(16)  # Đọc IV (16 byte tiếp theo)
        stored_ciphertext = f.read()  # Đọc phần còn lại là ciphertext

    # Giải mã dữ liệu
    cipher = AES.new(key, AES.MODE_CBC, iv=stored_iv)
    decrypted_data = unpad(cipher.decrypt(stored_ciphertext), AES.block_size)
    # Chuyển chuỗi nhị phân giải mã về dạng số nguyên
    binary_string = bin(int.from_bytes(decrypted_data, byteorder="big"))[2:].zfill(
        stored_ravel_length * 64
    )
    original_integer_representation = np.array(
        [int(binary_string[i : i + 64], 2) for i in range(0, len(binary_string), 64)]
    )

    # Chuyển đổi số nguyên về lại float

    original_abs_embedding_np = (
        original_integer_representation.astype(float) / DECIMAL_KEEP
    )
    original_abs_embedding_np = original_abs_embedding_np.reshape(stored_original_shape)
    # Áp dụng dấu đã lưu để khôi phục giá trị gốc
    original_embedding_np = original_abs_embedding_np * stored_signs.reshape(
        original_abs_embedding_np.shape
    )
    # So sánh gốc
    print("Original embedding")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first_outputs = []
    for i in range(0, n_segments):
        output = cos(embedding[i], embedding2[i])
        first_outputs.append(float(output))
    print(first_outputs)

    print("encrypt embedding")
    second_outputs = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for i in range(0, n_segments):
        original_embedding_tensor = torch.tensor(
            original_embedding_np[i], dtype=torch.float32
        )
        output = cos(original_embedding_tensor, embedding2[i])
        second_outputs.append(float(output))
    print(second_outputs)
    print(np.array(second_outputs) / np.array(first_outputs))
    with open("embeddingraw_afteraes.txt", "w") as f:
        f.write(str(original_embedding_np.tolist()))


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
