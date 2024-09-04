import ssdeep

import numpy as np
import soundfile as sf
import torch

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck

import torch.nn as nn
from lshashpy3 import LSHash

print("torch.cuda.is_available()", torch.cuda.is_available())


def fuzzy_hash_vector(vector):
    """Sử dụng fuzzy hashing với SSDEEP để băm vector đặc trưng"""
    vector_bytes = vector.tobytes()  # Chuyển đổi vector thành chuỗi bytes
    fuzzy_hash = ssdeep.hash(vector_bytes)
    return fuzzy_hash


def compare_fuzzy_hashes(hash1, hash2):
    """So sánh hai giá trị băm fuzzy hash"""
    return ssdeep.compare(hash1, hash2)


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
    ).numpy()

    embedding1_avg = np.mean(embedding, axis=0)
    print(embedding1_avg.shape)
    print(embedding.flatten().shape)
    print(embedding.shape[1])
    dim = embedding.flatten().shape[0]  # Số chiều của vector đặc trưng (embedding)
    num_hash_bits = 64  # Số lượng bits trong hàm băm LSH, bạn có thể điều chỉnh
    lsh = LSHash(num_hash_bits, dim)
    # print(embedding.flatten())
    lsh.index(embedding.flatten(), extra_data="audio1")
    # lsh.index(embedding1_avg, extra_data="audio1")

    print("SimHash has been saved to simhash1.txt")

    embedding2 = extract_speaker_embd(
        model,
        # fn='test_data/ts1_eSKwFJL.000000000.wav',
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    ).numpy()

    embedding2_avg = np.mean(embedding2, axis=0)

    results = lsh.query(embedding2.flatten(), num_results=1, distance_func="euclidean")
    # results = lsh.query(embedding2_avg, num_results=1, distance_func="euclidean")
    print("results", results)
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
