import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from models import create_vgg_model

# --- KONFIGURASI ---
# <-- BARU: Tambahkan path ke folder training Anda untuk mendapatkan daftar nama kelas
TRAIN_DATASET_PATH = 'D:/00. codingan/AI/Uas Cnn Card/dataset/train'
TEST_IMAGE_FOLDER = 'D:/00. codingan/AI/Uas Cnn Card/dataset/test_other' # <-- DIUBAH: Path ke folder berisi gambar
CHECKPOINT_LOAD_PATH = 'card_classifier_vgg.pth'


def predict_random_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    try:
        class_names = sorted([d.name for d in os.scandir(TRAIN_DATASET_PATH) if d.is_dir()])
        num_classes = len(class_names)
        if num_classes == 0:
            print(f"Error: Tidak ada folder kelas yang ditemukan di direktori training '{TRAIN_DATASET_PATH}'")
            print("Pastikan path training sudah benar.")
            return
        print(f"Ditemukan {num_classes} kelas dari folder training: {class_names}")
    except FileNotFoundError:
        print(f"Error: Direktori dataset training tidak ditemukan di '{TRAIN_DATASET_PATH}'")
        return

    model = create_vgg_model(num_classes)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_LOAD_PATH, map_location=device))
        print(f"Model berhasil dimuat dari: {CHECKPOINT_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: File checkpoint '{CHECKPOINT_LOAD_PATH}' tidak ditemukan.")
        return
    except RuntimeError as e:
        print(f"Error saat memuat model state_dict: {e}")
        print(f"Ini mungkin terjadi jika jumlah kelas ({num_classes}) tidak cocok dengan model yang disimpan.")
        return

    model = model.to(device)
    model.eval()

    try:
        image_files = [f for f in os.listdir(TEST_IMAGE_FOLDER) if os.path.isfile(os.path.join(TEST_IMAGE_FOLDER, f))]
        if not image_files:
            print(f"Error: Tidak ada file gambar di folder {TEST_IMAGE_FOLDER}")
            return
        random_image_name = random.choice(image_files)
        image_path = os.path.join(TEST_IMAGE_FOLDER, random_image_name)
    except FileNotFoundError:
        print(f"Error: Direktori gambar tes tidak ditemukan di '{TEST_IMAGE_FOLDER}'")
        return

    print(f"\nMenguji gambar acak: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    plt.imshow(image)
    plt.axis('off')
    plt.title(
        f"Prediksi: {predicted_label}\n"
        f"Keyakinan: {confidence_percent:.2f}%",
        fontsize=12
    )
    plt.show()


if __name__ == '__main__':
    predict_random_image()