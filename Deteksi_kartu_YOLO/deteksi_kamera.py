import cv2
from ultralytics import YOLO

# --- PATH MODEL ---
# Setelah training selesai, path ini akan menunjuk ke model terbaik Anda.
# Contoh: 'runs/detect/latihan_deteksi_kartu/weights/best.pt'
# Pastikan Anda mengganti path ini sesuai dengan hasil training Anda.
model_path = 'runs/detect/latihan_deteksi_kartu/weights/best.pt'

# Coba muat model, jika gagal, beri pesan error
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error memuat model: {e}")
    print(f"Pastikan path '{model_path}' sudah benar dan file model ada.")
    exit()

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Kamera tidak ditemukan atau tidak bisa dibuka.")
    exit()

print("Kamera siap. Arahkan ke kartu dan tekan 'q' untuk keluar.")

# Loop utama untuk memproses setiap frame dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Lakukan deteksi objek pada frame
    results = model(frame)

    # Gambar hasil deteksi pada frame
    # .plot() adalah fungsi bawaan ultralytics untuk menggambar kotak, label, dan keyakinan
    annotated_frame = results[0].plot()

    # Tampilkan frame yang sudah dianotasi
    cv2.imshow("Deteksi Kartu Real-time - Tekan 'q' untuk Keluar", annotated_frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan koneksi kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()