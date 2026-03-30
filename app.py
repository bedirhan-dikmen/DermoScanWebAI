import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- SAYFA AYARLARI (Kriter 17: Estetik ve Düzen) ---
st.set_page_config(
    page_title="DermoScan AI | Teşhis Paneli",
    page_icon="🔬",
    layout="centered"
)

# --- CSS DÜZELTMESİ (Hata Alınan Kısım Düzeltildi) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .result-card { 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 10px solid #007bff;
        background-color: white; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True) # Parametre düzeltildi

# --- MODEL YÜKLEME (Kriter 11: Eğitim Süreci) ---
@st.cache_resource
def load_my_model():
    model_path = 'final_model_3class.keras'
    # Versiyon hatalarını önlemek için compile=False kritik öneme sahip
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenemedi. Lütfen 'final_model_3class.keras' dosyasının ana dizinde olduğunu kontrol edin.")

# --- SINIF ETİKETLERİ ---
LABELS = {
    0: {"text": "BENIGN (İyi Huylu)", "color": "#28a745", "icon": "✅"},
    1: {"text": "MALIGNANT (Riskli / Kötü Huylu)", "color": "#dc3545", "icon": "⚠️"},
    2: {"text": "NORMAL DERİ / LEZYON YOK", "color": "#6c757d", "icon": "⚪"}
}

# --- SIDEBAR: MODEL PERFORMANSI (Kriter 14, 15, 16) ---
with st.sidebar:
    st.title("📊 Model Performansı")
    st.markdown("Akademik değerlendirme kriterlerine uygun başarı grafikleri:")
    
    # Görsellerin klasör yolu senin yapına göre güncellendi
    try:
        st.image("static/images/accuracy.png", caption="Eğitim Doğruluk Grafiği")
        st.divider()
        st.image("static/images/matrix.png", caption="Karmaşıklık Matrisi")
        st.divider()
        st.image("static/images/roc_curve.png", caption="ROC Eğrisi (Hassasiyet)")
    except:
        st.warning("Grafik dosyaları 'static/images/' klasöründe bulunamadı.")

# --- ANA SAYFA: ANALİZ ALANI ---
st.title("🔬 DermoScan AI")
st.write("Deri lezyonu fotoğrafını yükleyin, sistem otomatik olarak analiz edecektir.")

# Kriter 18: Kullanılabilirlik (Dosya yükleyici)
uploaded_file = st.file_uploader("Görsel seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görsel', use_container_width=True)
    
    # OTOMATİK ANALİZ (Kriter 19: Teknik Çalışırlık)
    with st.spinner('🧠 Yapay zeka öznitelikleri çıkartıyor...'):
        # 1. Ön İşleme (Kriter 6)
        img = image.convert('RGB').resize((224, 224))
        img_array = np.array(img)
        # ResNetV2 Normalizasyonu (-1, 1 aralığı)
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Tahmin (Kriter 13)
        preds = model.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100
        
        # SONUÇ KARTI (Kriter 17)
        res = LABELS[idx]
        st.markdown(f"""
            <div class="result-card">
                <h3 style='color: {res['color']}; margin-top:0;'>{res['icon']} {res['text']}</h3>
                <p style='font-size: 1.1rem; margin-bottom:0;'><b>Analiz Güveni:</b> %{confidence:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Teknik Yorum (Kriter 14)
        st.info(f"**Teknik Analiz:** ResNet50V2 mimarisi görüntüyü %{confidence:.2f} güven oranıyla '{res['text']}' olarak sınıflandırdı.")

else:
    st.caption("Analiz için lütfen bir fotoğraf yükleyiniz.")

# --- FOOTER (Kriter 20) ---
st.divider()
st.caption("Bu proje akademik bir değerlendirme için hazırlanmıştır. Gerçek teşhis için doktor onayı şarttır.")