from ultralytics import YOLO

# Memuat model dasar yolov8n (nano), yang cepat dan ringan.
# Untuk akurasi lebih baik, Anda bisa coba 'yolov8s.pt' atau 'yolov8m.pt'
model = YOLO('yolov8n.pt')

# Jalankan training
if __name__ == '__main__':
    # Pastikan Anda sudah mengaktifkan .venv sebelum menjalankan ini
    results = model.train(
        data='kartu.yaml',  # File konfigurasi dataset Anda
        epochs=100,         # Jumlah siklus training (bisa dimulai dari 50 atau 100)
        imgsz=640,          # Ukuran gambar input standar untuk YOLOv8
        batch=8,            # Kurangi jika VRAM GPU Anda kecil (misal: 4)
        name='latihan_deteksi_kartu' # Nama folder untuk menyimpan hasil
    )