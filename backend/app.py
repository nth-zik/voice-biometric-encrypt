from flask import Flask, request, jsonify
import torch
import numpy as np
import soundfile as sf
from models.models_utils import (  # Import các hàm từ mã nguồn
    extract_speaker_embd,
    embedding_to_gray_with_sign,
    jaccard_similarity,
    load_embeddings,
    save_embeddings,
)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Đường dẫn tới mô hình đã huấn luyện
MODEL_PATH = "./models/weights/model.pt"
model = None


# Tải mô hình khi khởi động API
def load_model():
    global model
    if model is None:
        from models.RawNet3 import RawNet3
        from models.RawNetBasicBlock import Bottle2neck

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
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))["model"]
        )
        model.eval()
        print("Model loaded!")


load_model()


@app.route("/api/extract", methods=["POST"])
def extract_embedding():
    # Nhận file âm thanh từ yêu cầu
    file = request.files.get("file")
    print("Received files:", file)
    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        # Thực hiện trích xuất vector đặc trưng
        embedding = extract_speaker_embd(model, file, n_samples=16000, n_segments=10)
        embedding_np = embedding.cpu().numpy()

        # Chuyển đổi sang chuỗi hexadecimal
        decimal_keep = 10**1  # Sử dụng hệ số lượng tử hóa của bạn
        binary_representation = embedding_to_gray_with_sign(
            embedding_np.ravel(), decimal_keep
        )
        hex_representation = hex(int(binary_representation, 2))

        return jsonify({"hex_embedding": hex_representation}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_embeddings():
    # Nhận hai chuỗi hexadecimal để so sánh
    data = request.get_json()
    hex1 = data.get("hex1")
    hex2 = data.get("hex2")

    if not hex1 or not hex2:
        return jsonify({"error": "Both hex1 and hex2 are required"}), 400

    try:
        # Tính độ tương đồng Jaccard
        similarity = jaccard_similarity(hex1, hex2)
        return jsonify({"jaccard_similarity": similarity}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


eer_threshold = 0.3543


@app.route("/api/verify", methods=["POST"])
def verify():
    if "file" not in request.files or "reference_hex" not in request.form:
        return jsonify({"error": "Both file and reference_hex must be provided"}), 400

    file = request.files.get("file")
    reference_hex = request.form["reference_hex"]

    try:
        # Đọc file âm thanh với soundfile và trích xuất embedding
        embedding = extract_speaker_embd(model, file, n_samples=16000, n_segments=10)
        embedding_np = embedding.cpu().numpy()

        decimal_keep = 10**1
        binary_representation = embedding_to_gray_with_sign(
            embedding_np.ravel(), decimal_keep
        )
        new_hex = hex(eval("0b" + binary_representation))

        # Tính độ tương đồng Jaccard và so sánh với ngưỡng EER
        similarity = jaccard_similarity(new_hex, reference_hex)

        if similarity >= eer_threshold:
            return jsonify(
                {
                    "verified": True,
                    "similarity": similarity,
                    "new_hex": new_hex,
                    "eer_threshold": eer_threshold,
                }
            )
        else:
            return jsonify(
                {
                    "verified": False,
                    "similarity": similarity,
                    "new_hex": new_hex,
                    "eer_threshold": eer_threshold,
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
