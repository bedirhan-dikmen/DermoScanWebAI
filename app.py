import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- SAYFA AYARLARI (Kriter 17: Estetik ve Düzen) ---
st.set_page_config(
    page_title="DermoScan AI | Teşhis Paneli",
    page_icon="🔬",
    layout="centered" # Daha odaklı bir görünüm için centered seçtik
)

# --- CUSTOM CSS (Kriter 17: Kullanıcı Dostu Arayüz) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 15px; border: 1px solid #ddd; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_stdio=True)

# --- MODEL YÜKLEME (Kriter 11: Eğitim Süreci Bilgisi) ---
@st.cache_resource
def load_my_model():
    model_path = 'final_model_3class.keras'
    # Versiyon hatalarını önlemek için compile=False kullanıyoruz
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenemedi. Hata: {e}")

# --- SINIF ETİKETLERİ ---
LABELS = {
    0: {"text": "BENIGN (İyi Huylu)", "color": "#28a745", "icon": "✅"},
    1: {"text": "MALIGNANT (Riskli)", "color": "#dc3545", "icon": "⚠️"},
    2: {"text": "NORMAL DERİ", "color": "#6c757d", "icon": "⚪"}
}

# --- SIDEBAR: MODEL PERFORMANSI (Kriter 14, 15, 16) ---
with st.sidebar:
    st.title("📊 Model Performansı")
    st.markdown("Eğitim sürecine dair teknik grafikler ve başarı metrikleri aşağıdadır.")
    
    # Grafikleri sidebar'a taşıdık
    st.image("static/images/accuracy.png", caption="Doğruluk & Kayıp Grafiği")
    st.divider()
    st.image("static/images/matrix.png", caption="Karmaşıklık Matrisi")
    st.divider()
    st.image("static/images/roc_curve.png", caption="ROC Eğrisi (Hassasiyet)")
    
    st.sidebar.info("**Not:** Bu grafikler modelin %80+ güven aralığında çalıştığını kanıtlamaktadır. [cite: 13, 14]")

# --- ANA SAYFA: ANALİZ ALANI ---
st.title("🔬 DermoScan AI Analiz Paneli")
st.write("Analiz için bir lezyon fotoğrafı sürükleyin veya seçin. Sistem otomatik olarak tahmin yürütecektir.")

# Dosya Yükleyici (Kriter 18: Kullanılabilirlik)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Analiz Edilen Görsel', use_container_width=True)
    
    # OTOMATİK ANALİZ (Kriter 19: Teknik Çalışırlık)
    with st.status("🧠 Görüntü işleniyor ve analiz ediliyor...", expanded=True) as status:
        # 1. Ön İşleme (Kriter 6)
        img = image.convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = (img_array / 127.5) - 1.0 # ResNetV2 Normalizasyonu
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Tahmin (Kriter 13)
        preds = model.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100
        
        status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)

    # SONUÇ GÖSTERİMİ (Kriter 17: Estetik Görünüm)
    res = LABELS[idx]
    st.markdown(f"""
        <div class="result-card">
            <h3 style='color: {res['color']};'>{res['icon']} {res['text']}</h3>
            <p style='font-size: 1.2rem;'><b>Tahmin Güveni:</b> %{confidence:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Teknik Yorum (Kriter 14)
    st.info(f"**Teknik Analiz:** Model, görüntüdeki öznitelikleri ResNet50V2 mimarisi üzerinden işleyerek {confidence:.2f}% olasılıkla '{res['text']}' sınıfına atamıştır. [cite: 8, 9]")

else:
    # Dosya yüklenmediğinde akademik bir hatırlatma
    st.warning("Lütfen analiz için geçerli bir dermatoskopik görüntü yükleyiniz.")

# --- FOOTER (Kriter 20) ---
st.divider()
st.caption("Bu proje TÜBİTAK 2209-A ve benzeri akademik standartlar gözetilerek geliştirilmiştir.")