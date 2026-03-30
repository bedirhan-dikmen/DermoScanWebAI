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
    st.title("📈 Model Teknik Başarı Analizi")
    st.write("Eğitim ve test süreçleri sonunda elde edilen akademik metrikler aşağıdadır.")

    # --- METRİK KARTLARI (Kriter 13: Sonuçların Düzenli Sunumu) ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="Genel Doğruluk (Accuracy)", value="%82.4")
    with col_m2:
        st.metric(label="Hassasiyet (Recall)", value="%79.8", delta="Kritik Metrik")
    with col_m3:
        st.metric(label="Keskinlik (Precision)", value="%81.2")
    with col_m4:
        st.metric(label="F1-Skoru", value="0.80")

    st.divider()

    # --- METRİK AÇIKLAMALARI (Kriter 14: Sonuçların Yorumlanması) ---
    with st.expander("📝 Metriklerin Teknik Açıklamalarını Görüntüle"):
        st.markdown("""
        * **Accuracy (Doğruluk):** Toplam tahminler içinde doğru bilinenlerin oranıdır. Modelin genel başarısını ifade eder.
        * **Precision (Keskinlik):** 'Malignant' (Kötü Huylu) olarak etiketlenen sonuçların gerçekte ne kadarının doğru olduğunu ölçer. Yanlış alarm oranını minimize eder.
        * **Recall (Duyarlılık):** **(En Önemli Metrik)** Gerçekte kötü huylu olan vakaların ne kadarının sistem tarafından yakalandığını gösterir. Tıbbi teşhiste vaka atlamamak için bu değerin yüksek olması hayati önem taşır.
        * **F1-Score:** Precision ve Recall değerlerinin harmonik ortalamasıdır; modelin dengeli performansını temsil eder.
        * **AUC (Area Under Curve):** ROC eğrisinin altında kalan alandır. 1.0'a ne kadar yakınsa, modelin iyi ve kötü huylu vakaları birbirinden ayırma yeteneği o kadar mükemmeldir (Projemizde: **0.88**).
        * **Loss (Kayıp):** Modelin eğitim sırasındaki hata miktarıdır. Bu değerin grafiklerde azaldığı gözlemlenmiştir, bu da öğrenmenin gerçekleştiğinin kanıtıdır.
        """)
    
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
    st.title("📖 Akademik Proje Özeti")
    
    st.markdown(f"""
    ### 1. Problem Tanımı ve Önem
    Deri lezyonlarının erken teşhisi hayati önem taşır. Bu proje, iyi huylu (benign) ve kötü huylu (malignant) benleri 
    ayırarak erken teşhis sürecine teknolojik destek sağlamayı amaçlar.

    ### 2. Veri Seti Bilgileri
    * **Format:** Veriler, doğrudan ham **.jpg** formatında görüntülerden oluşmaktadır.
    * **İçerik:** Toplam 4000 adet dermatoskopik görüntü (1500 Benign, 1500 Malignant,1000 Normal) kullanılmıştır.
    * **Ön İşleme:** Görüntüler model girişine uygun olarak 224x224 boyutuna getirilmiş ve piksel değerleri normalize edilmiştir.

    ### 3. Teknik Mimari
    * **Model:** ResNet50V2 (Transfer Learning)
    * **Hiperparametreler:** Adam Optimizer, 1e-4 Learning Rate, 10 Epoch.
    """)
    
    st.divider()
    st.markdown("### 📚 Kaynakça")
    st.write("1. Kaggle Dataset: [Skin Cancer: Malignant vs. Benign by Claudio Fanconi](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)")
    st.write("2. **Veri Seti:** Fanconi, C. Skin Cancer: Malignant vs. Benign. Kaggle (2019).")
    st.write("3. **Mimari:** He, K. et al. Deep Residual Learning for Image Recognition. CVPR (2016).")
    st.write("4. **Tıbbi Referans:** Esteva, A. et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature (2017).")
    st.write("5. **Kütüphane:** TensorFlow & Keras Documentation (2026).")
# --- FOOTER ---
st.divider()
st.caption("Bu çalışma akademik amaçlı bir 'Ben Teşhis Destek Sistemi' prototipidir. Tıbbi tavsiye yerine geçmez.")