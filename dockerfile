# Gunakan image dasar yang sesuai
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME /app

# Set working directory
WORKDIR $APP_HOME

# Salin requirements.txt dan install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . ./

# Jalankan aplikasi menggunakan gunicorn
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
