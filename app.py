import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 17 & 18. KRİTER: Sayfa Düzeni ve Görsel Standartlar ---
st.set_page_config(
    page_title="DermoScan AI | Deri Lezyonu Analizi",
    page_icon="🔬",
    layout="wide"
)

# --- 11. KRİTER: Model Yükleme ---
@st.cache_resource
def load_my_model():
    model_path = 'final_model_3class.keras'
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
except Exception:
    st.error("Model dosyası yüklenemedi! Lütfen dosya adını kontrol edin.")

# --- SIDEBAR: AKADEMİK BİLGİ VE NAVİGASYON ---
with st.sidebar:
    st.title("🔬 DermoScan AI")
    st.markdown("---")
    page = st.sidebar.radio("Menü Paneli", ["🏠 Lezyon Analizi", "📊 Model Performansı", "📖 Proje Detayları"])
    
    st.divider()
    st.subheader("🎯 Proje Amacı")
    st.info("Vücuttaki nevüslerin (benlerin) dermatoskopik görüntüler üzerinden iyi huylu (benign) veya kötü huylu (malign) ayrımını yaparak erken teşhise destek olmak.")

# --- 1. BÖLÜM: LEZYON ANALİZİ ---
if page == "🏠 Lezyon Analizi":
    st.title("🔍 Otomatik Ben Analiz Sistemi")
    st.write("Analiz için bir ben fotoğrafı yükleyin. Sistem, malign (riskli) ve benign (zararsız) ayrımını yapacaktır.")
    
    uploaded_file = st.file_uploader("Görsel Seçin (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Sütun yapısı ile resim boyutunu sınırlandırıyoruz (Kriter 17)
        col1, col2 = st.columns([1, 1.5]) 
        image = Image.open(uploaded_file)
        
        with col1:
            # Resim genişliğini 350px ile sınırladık, çok büyük görünmesini engelledik
            st.image(image, caption='Analiz Edilen Ben Görseli', width=350)
            
        with col2:
            with st.spinner('🧠 Derin öğrenme modeli analiz yürütüyor...'):
                # Kriter 6: Veri Ön İşleme
                img = image.convert('RGB').resize((224, 224))
                img_array = (np.array(img) / 127.5) - 1.0 
                img_array = np.expand_dims(img_array, axis=0)
                
                # Kriter 13: Tahmin Sunumu
                preds = model.predict(img_array)
                idx = np.argmax(preds)
                confidence = preds[0][idx] * 100
                
                # Projeye Özel Etiketler
                LABELS = {
                    0: {"t": "BENIGN (İyi Huylu / Zararsız)", "c": "green", "info": "Lezyon tipik özellikler göstermektedir, düşük riskli kategorisindedir."},
                    1: {"t": "MALIGNANT (Kötü Huylu / Riskli)", "c": "red", "info": "Lezyon atipik özellikler göstermektedir. Acilen uzman bir dermatoloğa danışılmalıdır!"},
                    2: {"t": "NORMAL DERİ / DİĞER", "c": "gray", "info": "Belirgin bir pigmente lezyon saptanmadı."}
                }
                
                res = LABELS[idx]
                st.markdown(f"### Teşhis Tahmini: :{res['c']}[{res['t']}]")
                st.metric(label="Tahmin Güven Oranı", value=f"%{confidence:.2f}")
                st.warning(f"**Sistem Notu:** {res['info']}")

# --- 2. BÖLÜM: MODEL PERFORMANSI ---
elif page == "📊 Model Performansı":
    st.title("📈 Teknik Başarı Metrikleri")
    
    # Grafik boyutlarını yan yana sütunlarla sınırlıyoruz (Kriter 16)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Eğitim Süreci (Accuracy/Loss)")
        st.image("static/images/accuracy.png", use_container_width=True)
        
    with c2:
        st.subheader("Hata Matrisi (Confusion Matrix)")
        st.image("static/images/matrix.png", use_container_width=True)

    st.divider()
    # ROC Eğrisi çok büyük göründüğü için width=500 ile sabitledik
    st.subheader("Hassasiyet Analizi (ROC Curve)")
    st.image("static/images/roc_curve.png", width=550)
    st.write("**Teknik Analiz:** Modelin malign vakaları yakalama hassasiyeti (sensitivity) ROC eğrisi ile doğrulanmıştır.")

# --- 3. BÖLÜM: PROJE DETAYLARI ---
elif page == "📖 Proje Detayları":
    st.title("📚 Akademik Proje Özeti")
    
    st.markdown("""
    ### 1. Problem ve Önem (Kriter 1-3)
    Malign melanom gibi kötü huylu deri kanserleri, erken teşhis edildiğinde tedavi edilebilirliği çok yüksek hastalıklardır. 
    Bu proje, nevüslerin görsel analizini yaparak kullanıcıyı ve hekimi olası risklere karşı uyarmayı amaçlar.

    ### 2. Teknik Altyapı (Kriter 8-10)
    * **Model:** ResNet50V2 (Transfer Learning)
    * **Veri Seti:** HAM10000 (Deri Kanseri Görsel Veri Seti)
    * **Yöntem:** Görüntü Sınıflandırma (3 Sınıf: Benign, Malignant, Normal)
    
    ### 3. Kullanılan Parametreler
    Eğitim sürecinde **Adam Optimizer** kullanılmış ve **Categorical Crossentropy** kaybı ile 10 epoch üzerinden optimizasyon sağlanmıştır.
    """)

# --- FOOTER ---
st.divider()
st.caption("Bu çalışma akademik amaçlı bir 'Ben Teşhis Destek Sistemi' prototipidir. Tıbbi tavsiye yerine geçmez.")