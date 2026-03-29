import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# --- 1. AYARLAR ---
# ResNet modeli 224x224 piksellik resimlerle çalışmayı sever.
IMG_SIZE = (224, 224) 
# Her bir öğrenme adımında modele kaç resim gösterileceği (Ram gücüne göre değişir)
BATCH_SIZE = 32
# Tüm veri setinin modelin üzerinden kaç kez geçeceği
EPOCHS = 10

# Veri setimizin bulunduğu klasör yolları
train_dir = 'dataset/train' 
test_dir = 'dataset/test'

# --- 2. VERİ HAZIRLIĞI VE ÇOĞALTMA (DATA AUGMENTATION) ---
# Eğitim verileri için zenginleştirme yapıyoruz. 
# Bu, modelin ezberlemesini (overfitting) engeller ve farklı açılardan tanımasını sağlar.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Resmi ResNet'in matematiksel formatına sokar (-1 ile 1 arası)
    rotation_range=40,      # Resmi rastgele 40 derece döndür
    width_shift_range=0.2,  # Resmi sağa sola kaydır
    height_shift_range=0.2, # Resmi yukarı aşağı kaydır
    shear_range=0.2,        # Resmi yamult (perspektif)
    zoom_range=0.2,         # Resme yakınlaş/uzaklaş
    horizontal_flip=True,   # Resmi aynala (yatay çevir)
    vertical_flip=True,     # Resmi dikey çevir
    fill_mode='nearest'     # Döndürme sonrası oluşan boşlukları en yakın renkle doldur
)

# Test/Doğrulama verileri için SADECE ön işleme yapıyoruz. 
# Test verilerini bozmamalıyız, orijinal halleriyle test etmeliyiz.
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("Veriler Yükleniyor...")
# Klasördeki resimleri okuyup modele besleyen "akış" (generator) oluşturuyoruz.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 3 sınıf olduğu için kategorik mod
    shuffle=True # Eğitimde verilerin sırasını karıştırmak iyidir
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Test ederken sırayı bozmuyoruz
)

# Hangi sınıfın hangi sayıya denk geldiğini göster (Örn: {'benign': 0, 'malignant': 1...})
print(f"Sınıf Haritası: {train_generator.class_indices}")

# --- 3. MODEL MİMARİSİ (TRANSFER LEARNING) ---
# ResNet50V2 motorunu yüklüyoruz (Weights='imagenet' diyerek hazır bilgiyle başlatıyoruz)
# include_top=False: ResNet'in sonundaki 1000 sınıflık karar katmanını atıyoruz, çünkü biz 3 sınıf istiyoruz.
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# base_model.trainable = False: Hazır modelin ağırlıklarını donduruyoruz.
# Böylece o mükemmel özellik çıkarıcı yapıyı bozmuyoruz, sadece kendi eklediğimiz son kısmı eğitiyoruz.
base_model.trainable = False 

# Kendi sınıflandırma katmanlarımızı ekliyoruz (Modelin "Kafası")
x = base_model.output
x = GlobalAveragePooling2D()(x) # Resim özelliklerini tek bir vektöre indirger
x = Dense(256, activation='relu')(x) # Öğrenme kapasitesini artıran ara katman
x = Dropout(0.5)(x) # Nöronların %50'sini rastgele kapatır (Ezberlemeyi önler)
predictions = Dense(3, activation='softmax')(x) # ÇIKIŞ KATMANI: 3 Sınıf için olasılık üretir

# Eski gövde ile yeni kafayı birleştiriyoruz
model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derliyoruz (Nasıl öğreneceğini söylüyoruz)
model.compile(optimizer=Adam(learning_rate=0.0001), # Adam optimizasyon algoritması (hassas öğrenme hızıyla)
              loss='categorical_crossentropy',      # Çok sınıflı sınıflandırma hatası hesaplama yöntemi
              metrics=['accuracy'])                 # Başarıyı 'doğruluk' oranıyla ölç

# --- 4. EĞİTİM (CALLBACKS - YARDIMCILAR) ---
# ModelCheckpoint: Eğitim sırasında en yüksek doğrulama başarısına sahip modeli kaydeder.
checkpoint = ModelCheckpoint('final_model_3class.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# EarlyStopping: Eğer 3 epoch boyunca gelişme olmazsa eğitimi durdur (Boşuna vakit harcama).
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ReduceLROnPlateau: Eğer model öğrenememeye başlarsa (takılırsa), öğrenme hızını düşürerek daha hassas ilerle.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

print(f"\n--- {EPOCHS} EPOCHLUK HIZLI EĞİTİM BAŞLIYOR ---\n")
# Eğitimi başlatıyoruz
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop, reduce_lr] # Yardımcıları ekledik
)

# --- 5. GRAFİKLER ---
# Eğitim bittikten sonra başarı grafiğini çiziyoruz
plt.figure(figsize=(12, 6))

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama (Test) Başarısı')
plt.title('Model Doğruluğu')
plt.legend()

# Kayıp (Loss) Grafiği - Hata oranı
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama (Test) Kaybı')
plt.title('Model Kaybı')
plt.legend()

plt.tight_layout()
plt.savefig('grafik_sonuc.png') # Grafiği dosyaya kaydet
plt.show() # Ekranda göster

print("Hızlı eğitim tamamlandı! 'final_model_3class.keras' hazır.")