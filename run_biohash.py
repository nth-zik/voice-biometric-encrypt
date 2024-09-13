import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def biohashing(embedding, random_matrix):
    """
    Áp dụng BioHashing lên vector đặc trưng.

    Args:
        embedding (np.ndarray): Vector đặc trưng.
        random_matrix (np.ndarray): Ma trận ngẫu nhiên (khóa).

    Returns:
        np.ndarray: Mã băm sinh trắc học.
    """
    # Nhân embedding với random_matrix
    transformed = np.dot(embedding, random_matrix)
    # Áp dụng hàm dấu để lượng tử hóa
    biohash = np.sign(transformed)
    # Chuyển từ -1, 0, 1 thành 0, 1
    biohash[biohash >= 0] = 1
    biohash[biohash < 0] = 0
    return biohash.astype(int)


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

    embedding_np = embedding.numpy()
    # Tạo ma trận ngẫu nhiên (khóa)
    np.random.seed(42)  # Trong thực tế, nên lưu trữ khóa này một cách an toàn
    random_matrix = np.random.randn(embedding_np.shape[1], embedding_np.shape[1])

    # Áp dụng BioHashing lên embedding
    biohash = biohashing(embedding_np, random_matrix)

    # Lưu trữ biohash xuống file
    with open("stored_biohash.npy", "wb") as f:
        np.save(f, biohash)

    # Trong quá trình xác thực
    # Trích xuất embedding mới
    embedding2 = extract_speaker_embd(
        model,
        fn="test_data/id01_2.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    embedding2_np = embedding2.numpy()

    # Áp dụng BioHashing lên embedding mới
    biohash2 = biohashing(embedding2_np, random_matrix)

    # Tải biohash đã lưu trữ
    with open("stored_biohash.npy", "rb") as f:
        stored_biohash = np.load(f)

    # So sánh hai mã băm
    hamming_dist = np.sum(np.abs(stored_biohash - biohash2))
    total_bits = stored_biohash.size
    hamming_percentage = (hamming_dist / total_bits) * 100

    print(
        f"Hamming distance between stored biohash and new biohash: {hamming_dist} {hamming_percentage}"
    )

    # Thiết lập ngưỡng để xác định người dùng
    threshold = 10  # Ngưỡng tùy chọn
    if hamming_dist <= threshold:
        print("Authentication successful!")
    else:
        print("Authentication failed!")


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

    if len(audio) < n_samples:  # RawNet3 được huấn luyện với đoạn âm thanh 1 giây
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
