FROM python:3.11

# Mengatur working directory
WORKDIR /app

# Meng-copy dan menginstall requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Meng-copy semua file project ke dalam container
COPY . .

# Hugging Face Spaces secara default mengekspos port 7860
ENV PORT=7860
EXPOSE 7860

# Menjalankan aplikasi Flask menggunakan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
