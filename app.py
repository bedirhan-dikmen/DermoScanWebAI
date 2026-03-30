import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- SAYFA AYARLARI (Kriter 17: Estetik Görünüm) ---
st.set_page_config(
    page_title="DermoScan AI | Teşhis Destek Sistemi",
    page_icon="🔬",
    layout="wide"
)

# --- MODEL YÜKLEME (RAM Tasarrufu İçin Cache Kullanımı) ---
@st.cache_resource
def load_my_model():
    # Model dosya adının doğru olduğundan emin ol
    model_path = 'final_model_3class.keras'
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenemedi! Lütfen dosya adını kontrol edin. Hata: {e}")

# --- SINIF ETİKETLERİ VE RENKLER ---
LABELS = {
    0: {"text": "BENIGN (İyi Huylu)", "color": "green"},
    1: {"text": "MALIGNANT (Riskli / Kötü Huylu)", "color": "red"},
    2: {"text": "NORMAL / LEZYON YOK", "color": "blue"}
}

# --- YAN PANEL: PROJE BİLGİLERİ (Kriter 1-5: Tanım ve Veri Seti) ---
with st.sidebar:
    st.title("🔬 Proje Hakkında")
    st.info("""
    **Problem Tanımı:** Deri kanserinin erken teşhisi için geliştirilmiş yapay zeka destekli analiz sistemi.
    
    **Model Mimarisi:** ResNet50V2 (Transfer Learning)
    
    **Veri Seti:** 3 Sınıflı dermatoskopik görüntüler (224x224 piksel).
    """)
    st.divider()
    st.write("© 2026 Akademik Sunum")

# --- ANA SAYFA TASARIMI ---
st.title("DermoScan AI: Deri Kanseri Analiz Paneli")
st.write("Sınav Değerlendirme Kriterlerine Uygun Teknik Altyapı")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Görüntü Yükleme")
    # Kriter 18: Kullanılabilirlik
    uploaded_file = st.file_uploader("Analiz edilecek fotoğrafı seçin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Yüklenen Görsel', use_container_width=True)
        
        # ANALİZ BUTONU
        if st.button('🧠 ANALİZİ BAŞLAT', use_container_width=True):
            with st.spinner('Model tahmin yürütüyor...'):
                # 1. Ön İşleme (Kriter 6: Normalizasyon)
                img = image.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                
                # ResNetV2 preprocess_input mantığı (-1 ile 1 arası)
                img_array = (img_array / 127.5) - 1.0 
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. Tahmin (Kriter 13: Sonuçların Sunumu)
                preds = model.predict(img_array)
                idx = np.argmax(preds)
                confidence = preds[0][idx] * 100
                
                # SONUÇ EKRANI
                res = LABELS[idx]
                st.success("Analiz Tamamlandı!")
                st.markdown(f"### Sonuç: :{res['color']}[{res['text']}]")
                st.metric(label="Güven Oranı", value=f"%{confidence:.2f}")

with col2:
    st.subheader("📊 Model Performans Metrikleri")
    # Kriter 15-16: Grafiklerin Kullanımı ve Kalitesi
    # Bu dosyaların static/images içinde olduğundan emin ol
    tab1, tab2, tab3 = st.tabs(["Doğruluk", "Matris", "ROC Eğrisi"])
    
    with tab1:
        st.image("static/images/accuracy.png", caption="Eğitim vs Doğruluk (Accuracy)")
        st.caption("**Kriter 14 Yorumu:** Eğitim ve test başarılarının paralelliği modelin genelleme yeteneğini gösterir.")
        
    with tab2:
        st.image("static/images/matrix.png", caption="Karmaşıklık Matrisi (Confusion Matrix)")
        
    with tab3:
        st.image("static/images/roc_curve.png", caption="ROC Eğrisi (Hassasiyet)")

st.divider()
# Kriter 20: Sonuç ve Bütünlük
st.write("**Teknik Not:** Bu sistem bir karar destek mekanizmasıdır. Kesin teşhis için uzman doktor onayı gereklidir.")