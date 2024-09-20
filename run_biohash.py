import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def quantize_embedding(embedding_np, num_bits=16):
    """
    Lượng tử hóa embedding thành biểu diễn nhị phân với số bit xác định.

    Args:
        embedding_np (np.ndarray): Vector đặc trưng.
        num_bits (int): Số bit để lượng tử hóa mỗi giá trị (tối đa 8).

    Returns:
        np.ndarray: Mảng nhị phân sau khi lượng tử hóa.
    """
    if num_bits > 32:
        raise ValueError("num_bits phải nhỏ hơn hoặc bằng 32.")

    # Chuẩn hóa embedding về khoảng [0, 1]
    min_val = embedding_np.min()
    max_val = embedding_np.max()
    embedding_norm = (embedding_np - min_val) / (max_val - min_val)

    # Lượng tử hóa thành số mức 2^num_bits
    quantized = np.floor(embedding_norm * (2**num_bits - 1)).astype(np.uint8)

    # Chuyển đổi thành mảng nhị phân
    binary_array = np.unpackbits(quantized[:, :, np.newaxis], axis=2, bitorder="big")[
        :, :, -num_bits:
    ]
    binary_array = binary_array.reshape(-1)

    return binary_array


def biohashing(binary_array, random_matrix):
    """
    Áp dụng BioHashing lên mảng nhị phân.

    Args:
        binary_array (np.ndarray): Mảng nhị phân.
        random_matrix (np.ndarray): Ma trận ngẫu nhiên (khóa).

    Returns:
        np.ndarray: Mã băm sinh trắc học.
    """
    # Chuyển đổi binary_array thành vector số thực (-1 và 1)
    binary_vector = binary_array * 2 - 1  # Chuyển 0 thành -1, 1 giữ nguyên
    # Áp dụng BioHashing với ma trận ngẫu nhiên nhỏ hơn
    transformed = np.dot(binary_vector, random_matrix)
    # Lượng tử hóa kết quả
    biohash = (transformed > 0).astype(int)
    return biohash


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
    print("RawNet3 đã được khởi tạo và tải trọng số!")
    if torch.cuda.is_available():
        print("Cuda khả dụng, thực hiện suy luận trên GPU")
        model = model.to("cuda")
        gpu = True

    # Trích xuất embedding
    embedding = extract_speaker_embd(
        model,
        fn="test_data/id01_1.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )

    embedding_np = embedding.numpy()

    # Lượng tử hóa embedding thành mảng nhị phân
    num_bits = 8  # Số bit lượng tử hóa
    binary_array = quantize_embedding(embedding_np, num_bits=num_bits)

    # Tạo ma trận ngẫu nhiên (khóa) nhỏ hơn
    output_dim = 2048  # Kích thước mã băm mong muốn
    np.random.seed(42)  # Trong thực tế, nên lưu trữ khóa này một cách an toàn
    random_matrix = np.random.randn(len(binary_array), output_dim)

    # Áp dụng BioHashing lên mảng nhị phân
    biohash = biohashing(binary_array, random_matrix)

    # Lưu trữ biohash xuống file
    with open("stored_biohash.npy", "wb") as f:
        np.save(f, biohash)

    # Trong quá trình xác thực
    # Trích xuất embedding mới
    embedding2 = extract_speaker_embd(
        model,
        # fn="test_data/id01_2.wav",
        fn="test_data/00004_old.wav",
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=gpu,
    )
    embedding2_np = embedding2.numpy()

    # Lượng tử hóa embedding mới thành mảng nhị phân
    binary_array2 = quantize_embedding(embedding2_np, num_bits=num_bits)

    # Áp dụng BioHashing lên mảng nhị phân mới
    biohash2 = biohashing(binary_array2, random_matrix)

    # Tải biohash đã lưu trữ
    with open("stored_biohash.npy", "rb") as f:
        stored_biohash = np.load(f)

    # Tính khoảng cách Hamming giữa hai biohash
    hamming_dist = np.sum(np.abs(stored_biohash - biohash2))
    total_bits = stored_biohash.size
    hamming_percentage = (hamming_dist / total_bits) * 100
    print(
        f"Khoảng cách Hamming giữa biohash lưu trữ và biohash mới: {hamming_percentage:.2f}% {hamming_dist} / {stored_biohash.size}"
    )

    # Thiết lập ngưỡng để xác định người dùng (ví dụ: 10% sai khác)
    threshold_percentage = 10.0  # Ngưỡng tùy chọn
    if hamming_percentage <= threshold_percentage:
        print("Xác thực thành công!")
    else:
        print("Xác thực thất bại!")


def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 chỉ hỗ trợ đầu vào mono. Dữ liệu đầu vào có shape {audio.shape}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"RawNet3 chỉ hỗ trợ tần số lấy mẫu 16k. Tần số của dữ liệu đầu vào là {sample_rate}."
        )

    if len(audio) < n_samples:
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
