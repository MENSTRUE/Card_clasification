import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from models import create_vgg_model
from dataset import create_dataloaders  # Kita akan gunakan sebagian kecil dari ini

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- KONFIGURASI EVALUASI ---
MODEL_PATH = 'card_classifier_vgg.pth'
TEST_DATA_PATH = 'D:/00. codingan/AI/Uas Cnn Card/dataset/test'
BATCH_SIZE = 32


# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Siapkan Test Loader
    # Transformasi untuk data test/validasi (tanpa augmentasi)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Dataset test dimuat. Ditemukan {num_classes} kelas.")

    # 2. Muat Model yang Sudah Dilatih
    model = create_vgg_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model {MODEL_PATH} berhasil dimuat.")

    # 3. Lakukan Evaluasi
    print("\n--- Memulai Evaluasi pada Test Set ---")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Mengevaluasi"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Tampilkan Hasil Evaluasi
    accuracy = 100 * correct / total
    print(f"\n--- Hasil Evaluasi Selesai ---")
    print(f"Akurasi Final di Test Set: {accuracy:.2f}%")

    print("\nLaporan Klasifikasi di Test Set:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix di Test Set:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Test Set')
    plt.show()


if __name__ == '__main__':
    main()