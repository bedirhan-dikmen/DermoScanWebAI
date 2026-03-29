import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os

# --- AYARLAR ---
MODEL_PATH = 'final_model_3class.keras'  # Modelin yolu
TEST_DIR = 'dataset/test'                # Test verilerinin yolu
SAVE_DIR = 'static/images'               # Kaydedilecek yer

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- MODELİ VE VERİYİ YÜKLE ---
print("Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Grafikler için sıralama bozulmamalı
)

# Sınıf isimleri: ['benign', 'malignant', 'normal']
class_names = list(test_generator.class_indices.keys())
n_classes = len(class_names)

# --- TAHMİN YAP ---
print("Tahminler yapılıyor...")
y_pred_prob = model.predict(test_generator, verbose=1) # Olasılıklar (ROC için lazım)
y_pred_class = np.argmax(y_pred_prob, axis=1)          # Sınıf Tahmini (Matris için lazım)
y_true = test_generator.classes                        # Gerçekler

# --- GRAFİK 1: CONFUSION MATRIX (KARIŞIKLIK MATRİSİ) ---
print("Matris çiziliyor...")
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/matrix.png') # KAYDET
print("matrix.png kaydedildi.")

# --- GRAFİK 2: ROC CURVE (ÇOK SINIFLI) ---
print("ROC Eğrisi çiziliyor...")
# Etiketleri binarize et (Her sınıf için 0 ve 1 mantığına çevir)
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))

colors = ['green', 'red', 'blue'] # Benign, Malignant, Normal renkleri
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2) # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm)')
plt.ylabel('True Positive Rate (Doğru Tespit)')
plt.title('ROC Curve (Model Hassasiyeti)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/roc_curve.png') # KAYDET
print("roc_curve.png kaydedildi.")

# --- GRAFİK 3: ACCURACY (DOĞRULUK) ---
# NOT: Bu grafik sadece eğitim (training) anında oluşur.
# Elindeki 'save1.png' veya 'grafik_sonuc.png' dosyasını
# static/images klasörüne 'accuracy.png' adıyla kopyalaman yeterlidir.
print(f"Lütfen eğitimden kalan doğruluk grafiğini '{SAVE_DIR}/accuracy.png' olarak adlandırın.")