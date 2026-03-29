import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)

# --- AYARLAR ---
# Model dosyanın 'final_model_3class.keras' adıyla aynı klasörde olduğundan emin ol.
MODEL_PATH = 'final_model_3class.keras' 
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Sınıf Etiketleri (Arayüzde daha profesyonel görünmesi için düzenlendi)
LABELS = {
    0: "BENIGN (İyi Huylu)",
    1: "MALIGNANT (Kötü Huylu - Riskli)",
    2: "NORMAL DERİ / LEZYON YOK"
}

def model_predict(img_path, model):
    # 1. Resmi yükle ve ResNet boyutu olan 224x224'e getir (Kriter 6)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # 2. ResNetV2 için normalizasyon (-1 ile 1 arası) uygula (Kriter 6)
    x = preprocess_input(x) 

    # 3. Tahmin yap
    preds = model.predict(x)
    result_index = np.argmax(preds)
    
    # Güven oranını sayısal olarak al (Progress bar için %)
    confidence_value = preds[0][result_index] * 100
    
    sonuc_metni = LABELS[result_index]
    
    # Bootstrap renklerini belirle
    if result_index == 1: 
        renk = "danger"
    elif result_index == 0: 
        renk = "success"
    else: 
        renk = "primary"

    # HTML'de hem metin hem de saf yüzde değeri kullanabilmek için tuple dönüyoruz
    return sonuc_metni, f"%{confidence_value:.2f}", renk

# --- ANA ROTA ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    image_loc = None
    color = None
    scroll_to_result = False

    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            # Güvenli dosya ismi ve kaydetme
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            
            # Analiz fonksiyonunu çağır
            prediction, probability, color = model_predict(file_path, model)
            
            # Statik klasöründeki yolu HTML'e uygun hale getir
            image_loc = file_path.replace('\\', '/') 
            scroll_to_result = True 

    return render_template('index.html', 
                           prediction=prediction, 
                           probability=probability, 
                           image_loc=image_loc,
                           color=color,
                           scroll_to_result=scroll_to_result)

if __name__ == '__main__':
    # Geliştirme aşamasında debug=True kalsın, sunuma geçerken kapatabilirsin.
    app.run(debug=True, port=5000)