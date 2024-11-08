# Sử dụng image chính thức của Python
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file yêu cầu vào container và cài đặt thư viện
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Thiết lập biến môi trường cho Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port 5000 để truy cập Flask app từ bên ngoài
EXPOSE 5000

# Chạy Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
