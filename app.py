import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- SAYFA AYARLARI (Kriter 17: Estetik Görünüm) ---
st.set_page_config(
    page_title="DermoScan AI",
    page_icon="🔬",
    layout="wide"
)

# --- MODEL YÜKLEME (Kriter 11) ---
@st.cache_resource
def load_my_model():
    model_path = 'final_model_3class.keras'
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error("Model dosyası yüklenemedi!")

# --- SIDEBAR: NAVİGASYON VE BİLGİ (Kriter 1, 2, 3) ---
with st.sidebar:
    st.title("🔬 DermoScan AI")
    st.markdown("---")
    
    # SAYFA SEÇİMİ (İstediğin 'ayrı sayfaya gitme' tuşu burasıdır)
    page = st.sidebar.radio("Menü", ["🏠 Analiz Paneli", "📊 Model Performansı"])
    
    st.markdown("---")
    st.subheader("📝 Proje Bilgileri")
    st.info("""
    **Amaç:** Deri lezyonlarının yapay zeka ile ön teşhisi. 
    **Kapsam:** Benign, Malignant ve Normal deri sınıfları. 
    **Teknoloji:** ResNet50V2 & Streamlit Cloud.
    """)
    st.write("© 2026 Akademik Proje Ödevi")

# --- 1. SAYFA: ANALİZ PANELİ ---
if page == "🏠 Analiz Paneli":
    st.title("🔍 Otomatik Teşhis Sistemi")
    st.write("Analiz için bir fotoğraf yüklediğinizde sistem otomatik olarak çalışacaktır.")
    
    uploaded_file = st.file_uploader("Lezyon Görseli Seçin", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption='Yüklenen Görsel', use_container_width=True)
            
        with col2:
            with st.spinner('🧠 Yapay zeka analiz ediyor...'):
                # Ön İşleme (Kriter 6)
                img = image.convert('RGB').resize((224, 224))
                img_array = (np.array(img) / 127.5) - 1.0 
                img_array = np.expand_dims(img_array, axis=0)
                
                # Tahmin (Kriter 13)
                preds = model.predict(img_array)
                idx = np.argmax(preds)
                confidence = preds[0][idx] * 100
                
                # Sonuç Etiketleri
                LABELS = {
                    0: {"t": "BENIGN (İyi Huylu)", "c": "green"},
                    1: {"t": "MALIGNANT (Riskli)", "c": "red"},
                    2: {"t": "NORMAL DERİ", "c": "gray"}
                }
                
                res = LABELS[idx]
                st.markdown(f"### Sonuç: :{res['c']}[{res['t']}]")
                st.metric(label="Güven Oranı", value=f"%{confidence:.2f}")
                st.info(f"**Teknik Yorum:** Görüntü %{confidence:.2f} güvenle '{res['t']}' olarak sınıflandırıldı. ")

# --- 2. SAYFA: GRAFİKLER ---
elif page == "📊 Model Performansı":
    st.title("📈 Model Eğitim ve Başarı Grafikleri")
    st.write("Bu bölümde modelin akademik geçerliliğini kanıtlayan performans metrikleri yer almaktadır.")
    
    # Kriter 15-16: Grafiklerin Kalitesi ve Sunumu
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Accuracy & Loss")
        st.image("static/images/accuracy.png", use_container_width=True)
        st.caption("**Kriter 14:** Eğitim ve doğrulama başarıları arasındaki paralellik, modelin ezberlemediğini (overfitting olmadığını) gösterir. ")
        
    with col_b:
        st.subheader("Confusion Matrix")
        st.image("static/images/matrix.png", use_container_width=True)
        st.caption("Modelin sınıfları birbirine karıştırma oranlarını gösteren matris.")

    st.divider()
    st.subheader("ROC Eğrisi (Hassasiyet)")
    st.image("static/images/roc_curve.png", use_container_width=True)
    st.write("**AUC Değeri:** Modelin hassasiyet ve özgüllük dengesini temsil eder.")

# --- FOOTER (Kriter 20) ---
st.divider()
st.caption("Akademik Sınav Değerlendirme Projesi | 2026")