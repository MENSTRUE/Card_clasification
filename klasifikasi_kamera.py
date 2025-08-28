import cv2
import torch
from torchvision import transforms
from PIL import Image
from models import create_vgg_model  # Menggunakan model dari file models.py Anda

CHECKPOINT_LOAD_PATH = 'card_classifier_vgg.pth'
TRAIN_DATASET_PATH = 'D:/00. codingan/AI/Uas Cnn Card/dataset/train'  # Path untuk mengambil nama kelas

def load_class_names(train_path):
    import os
    try:
        class_names = sorted([d.name for d in os.scandir(train_path) if d.is_dir()])
        if not class_names:
            print(f"Error: Tidak ada folder kelas ditemukan di {train_path}")
            return None
        return class_names
    except FileNotFoundError:
        print(f"Error: Direktori training tidak ditemukan di {train_path}")
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # Muat nama kelas
    class_names = load_class_names(TRAIN_DATASET_PATH)
    if class_names is None:
        return
    num_classes = len(class_names)
    print(f"Ditemukan {num_classes} kelas: {class_names[:5]}...")  # Tampilkan 5 kelas pertama

    # Buat dan muat model VGG yang sudah dilatih
    model = create_vgg_model(num_classes)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_LOAD_PATH, map_location=device))
        print(f"Model berhasil dimuat dari: {CHECKPOINT_LOAD_PATH}")
    except FileNotFoundError:
        print(
            f"Error: File checkpoint '{CHECKPOINT_LOAD_PATH}' tidak ditemukan. Pastikan Anda sudah menjalankan train.py")
        return
    except RuntimeError as e:
        print(f"Error saat memuat model: {e}")
        print("Ini mungkin terjadi jika jumlah kelas tidak cocok dengan model yang disimpan.")
        return

    model = model.to(device)
    model.eval()  # Set model ke mode evaluasi

    # Transformasi untuk gambar input dari kamera
    # HARUS SAMA dengan transformasi validasi saat training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka.")
        return

    print("\nKamera siap. Posisikan satu kartu di dalam kotak hijau.")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tentukan area target (Region of Interest - ROI) di tengah layar
        h, w, _ = frame.shape
        roi_size = 300  # Ukuran kotak
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Gambar kotak ROI di frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ambil gambar dari dalam kotak (ROI)
        roi = frame[y1:y2, x1:x2]

        # Konversi ROI dari format OpenCV (BGR) ke PIL (RGB)
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Terapkan transformasi
        image_tensor = transform(roi_pil).unsqueeze(0).to(device)

        # Lakukan prediksi
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_label = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        # --- Blok Kode Baru untuk Menampilkan Teks dengan Latar Belakang ---

        # Buat teks yang akan ditampilkan
        text = f"Prediksi: {predicted_label} ({confidence_score:.2f}%)"

        # Tentukan font, ukuran, dan ketebalan
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Hitung ukuran kotak yang dibutuhkan untuk teks
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Tentukan posisi untuk kotak latar belakang
        box_y1 = y1 - text_height - 15  # 15 adalah padding/jarak dari kotak hijau
        box_y2 = y1 - 5

        # Gambar kotak latar belakang (warna hijau, terisi penuh)
        cv2.rectangle(frame, (x1, box_y1), (x1 + text_width + 10, box_y2), (0, 150, 0), -1)

        # Tulis teks di atas kotak latar belakang (warna putih, lebih kontras)
        cv2.putText(frame, text, (x1 + 5, y1 - 10), font, font_scale, (255, 255, 255), font_thickness)
        # --- Akhir Blok Kode Baru ---

        # Tampilkan frame
        cv2.imshow('Klasifikasi Kartu Real-time', frame)

        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()