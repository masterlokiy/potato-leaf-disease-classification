import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import time
import os
from skimage.feature import local_binary_pattern

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Daun Kentang",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database informasi penyakit dengan warna spesifik
DISEASE_INFO = {
    "Healthy": {
        "description": "Tanaman dalam kondisi sehat tanpa tanda-tanda penyakit",
        "symptoms": "Warna hijau seragam di semua bagian daun dan bentuk daun sempurna tanpa cacat",
        "treatment": "Pertahankan perawatan rutin dan pemantauan",
        "prevention": "Pemupukan teratur, penyiraman yang cukup, dan inspeksi berkala",
        "color": "#4CAF50",  # Hijau
        "icon": "✅"
    },
    "Virus": {
        "description": "Infeksi virus pada tanaman",
        "symptoms": "Ukuran daun menyusut dan keriput, bercak ringan atau mosaik, nekrosis",
        "treatment": "Pemusnahan tanaman terinfeksi, penggunaan varietas tahan virus, kontrol vektor",
        "prevention": "Sanitasi kebun, kontrol serangga vektor, penggunaan alat steril",
        "color": "#9C27B0",  # Ungu
        "icon": "🦠"
    },
    "Phytophthora": {
        "description": "Penyakit busuk daun dan akar yang disebabkan oleh organisme mirip jamur",
        "symptoms": "Lesi coklat tua hingga hitam pada daun yang dapat meluas menjadi bercak nekrotik melingkar",
        "treatment": "Fungisida sistemik, perbaikan drainase, pemangkasan bagian terinfeksi",
        "prevention": "Rotasi tanaman, hindari kelembaban berlebih, penggunaan mulsa",
        "color": "#795548",  # Coklat
        "icon": "🍂"
    },
    "Nematode": {
        "description": "Serangan cacing nematoda pada sistem perakaran",
        "symptoms": "Daun menguning dengan gejala mirip kekurangan air dan nutrisi",
        "treatment": "Nematisida, solarisasi tanah, penanaman tanaman penangkal",
        "prevention": "Penggunaan bibit bebas nematoda, rotasi tanaman, peningkatan bahan organik tanah",
        "color": "#FFEB3B",  # Kuning
        "icon": "🪱"
    },
    "Fungi": {
        "description": "Infeksi jamur pada jaringan tanaman",
        "symptoms": "Bercak melingkar di tepi daun dan/atau bercak daun agak cekung dengan pinggiran kuning dan lingkaran konsentris, dan/atau daun kuning dengan bercak tepung",
        "treatment": "Fungisida kontak/sistemik, pemangkasan daun terinfeksi",
        "prevention": "Sirkulasi udara baik, hindari penyiraman daun, sanitasi kebun",
        "color": "#FF9800",  # Oranye
        "icon": "🍄"
    },
    "Bacteria": {
        "description": "Infeksi bakteri pada tanaman",
        "symptoms": "Daun layu tanpa mengering atau nekrosis; daun awalnya tidak menguning",
        "treatment": "Bakterisida tembaga, pemangkasan bagian sakit",
        "prevention": "Penggunaan benih sehat, alat steril, rotasi tanaman",
        "color": "#2196F3",  # Biru
        "icon": "🧫"
    },
    "Pest": {
        "description": "Serangan serangga atau arthropoda lainnya",
        "symptoms": "Jaringan daun terdistorsi dan/atau berlubang dan/atau daun berbintik dengan warna keperakan atau bahan klorotik dan/atau dengan jalur galian pada daun",
        "treatment": "Insektisida selektif, kontrol biologis, pembuangan manual",
        "prevention": "Monitoring rutin, tanaman perangkap, menjaga kebersihan kebun",
        "color": "#F44336",  # Merah
        "icon": "🐛"
    }
}

# Navigasi halaman
page = st.sidebar.radio("Menu", ["🔍 Deteksi", "📌 Batasan Masalah"])

if page == "📌 Batasan Masalah":
    st.title("📌 Batasan Masalah")

    # CSS styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .main { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
    }

    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: transparent;
    }

    .header {
        color: #2c3e50;
        padding: 1.5rem 0;
        border-bottom: 1px solid #eee;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # Info section
    st.info("""
    1. 🌦️ **Lingkungan Tidak Terkontrol**  
       Sistem dipengaruhi oleh pencahayaan yang kurang, bayangan, latar belakang yang ramai, dan posisi kamera yang tidak konsisten. Hal ini menyulitkan model dalam mengenali ciri-ciri penyakit secara akurat.

    2. 🧩 **Keterbatasan Fitur Ekstraksi**  
       Saat ini, sistem menggunakan metode seperti *Enhanced Color Correlogram* dan *MCLBP*. Namun metode ini belum cukup kuat dalam mengenali pola visual kompleks. Alternatif yang lebih baik seperti **CNN (Convolutional Neural Network)** dapat memberikan hasil yang lebih presisi.

    3. 📚 **Terbatas pada Data Latih**  
       Model hanya mengenali penyakit yang tersedia dalam dataset pelatihan. Penyakit baru atau gejala yang tidak terdapat dalam data kurang dapat terdiagnosis dengan benar.

    📷 **Tips Pengguna:**  
    Gunakan gambar dengan pencahayaan terang, latar belakang polos, dan fokus yang jelas untuk meningkatkan akurasi diagnosis.
    """)

    st.stop()

    
# CSS kustom dengan animasi dan warna penyakit
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    .main {{ 
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
    }}
    
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
        background-color: transparent;
    }}
    
    .header {{
        color: #2c3e50;
        padding: 1.5rem 0;
        border-bottom: 1px solid #eee;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }}
    
    .header h1 {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .result-box {{
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        animation: slideIn 0.5s ease-out;
    }}
    
    .confidence-bar {{
        height: 30px;
        border-radius: 8px;
        margin: 1rem 0;
        background: linear-gradient(90deg, #e0e0e0 0%, #e0e0e0 var(--confidence), var(--disease-color) var(--confidence), var(--disease-color) 100%);
        transition: all 0.3s ease;
    }}
    
    .confidence-bar:hover {{
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    .file-uploader {{
        border: 2px dashed #4e73df;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background: rgba(78, 115, 223, 0.05);
        transition: all 0.3s ease;
    }}
    
    .file-uploader:hover {{
        background: rgba(78, 115, 223, 0.1);
    }}
    
    .disease-card {{
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border-left: 5px solid var(--disease-color);
    }}
    
    .disease-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }}
    
    .disease-header {{
        border-left: 5px solid var(--disease-color);
        padding-left: 1rem;
        margin-bottom: 1rem;
    }}
    
    .footer {{
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #eee;
        font-size: 0.9rem;
        color: #7f8c8d;
        text-align: center;
        background: white;
        border-radius: 12px;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    .tab-content {{
        padding: 1.5rem 0;
    }}
    
    /* Warna khusus untuk setiap penyakit */
    .healthy-color {{ color: {DISEASE_INFO['Healthy']['color']}; }}
    .virus-color {{ color: {DISEASE_INFO['Virus']['color']}; }}
    .phytophthora-color {{ color: {DISEASE_INFO['Phytophthora']['color']}; }}
    .nematode-color {{ color: {DISEASE_INFO['Nematode']['color']}; }}
    .fungi-color {{ color: {DISEASE_INFO['Fungi']['color']}; }}
    .bacteria-color {{ color: {DISEASE_INFO['Bacteria']['color']}; }}
    .pest-color {{ color: {DISEASE_INFO['Pest']['color']}; }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_dict = joblib.load("model_new.pkl")
        required_keys = ['svm', 'scaler', 'label_encoder']
        if not all(key in model_dict for key in required_keys):
            missing = [key for key in required_keys if key not in model_dict]
            raise ValueError(f"Model kehilangan komponen penting: {missing}")
        return {
            'svm': model_dict['svm'],
            'scaler': model_dict['scaler'],
            'le': model_dict['label_encoder'],
            'class_names': model_dict.get('class_names', [])
        }
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

def enhanced_color_correlogram(img, distances=[1, 3, 5, 7], n_samples=1000):
    """Ekstraksi fitur Enhanced Color Correlogram"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    quantized = []
    for cspace in [hsv, lab]:
        q = (cspace // 64).astype(np.uint8)
        quantized.append(q)
    
    rows, cols = img.shape[:2]
    total_pixels = rows * cols
    n_samples = min(n_samples, total_pixels)
    
    features = []
    for q_img in quantized:
        for d in distances:
            rng = np.random.RandomState(42)
            indices = rng.choice(total_pixels, n_samples, replace=False)
            
            for c in range(3):
                channel = q_img[..., c]
                hist = np.zeros(4, dtype=np.float32)
                count = 0
                
                i, j = np.unravel_index(indices, (rows, cols))
                neighbors = [
                    np.clip(i - d, 0, rows-1), j,
                    np.clip(i + d, 0, rows-1), j,
                    i, np.clip(j - d, 0, cols-1),
                    i, np.clip(j + d, 0, cols-1)
                ]
                
                for ni, nj in zip(neighbors[::2], neighbors[1::2]):
                    mask = (channel[i, j] == channel[ni, nj])
                    np.add.at(hist, channel[i, j][mask], 1)
                    count += len(mask)
                
                if count > 0:
                    hist /= count
                features.extend(hist)
    
    return np.array(features)

def robust_mclbp(img, radii=[1, 2, 3], n_points=8):
    """Ekstraksi fitur Multi-Channel LBP"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    channels = [gray, hsv[..., 0], hsv[..., 1], hsv[..., 2], 
               lab[..., 0], lab[..., 1], lab[..., 2]]
    
    features = []
    for channel in channels:
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        for radius in radii:
            lbp = local_binary_pattern(channel, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                 range=(0, n_points + 2), density=True)
            hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 3), 0.5).flatten()
            features.extend(hist)
    
    return np.array(features)

def extract_features(image):
    image_resized = cv2.resize(image, (224, 224))
    cc_feat = enhanced_color_correlogram(image_resized)
    mclbp_feat = robust_mclbp(image_resized)
    return np.concatenate([cc_feat, mclbp_feat])

def predict_image(image, model_dict):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        features = extract_features(image_cv)
        features_scaled = model_dict['scaler'].transform([features])
        
        pred = model_dict['svm'].predict(features_scaled)[0]
        if hasattr(model_dict['svm'], "predict_proba"):
            proba = model_dict['svm'].predict_proba(features_scaled)[0]
        else:
            proba = np.zeros(len(model_dict['class_names']))
            pred_idx = np.where(model_dict['le'].classes_ == pred)[0][0]
            proba[pred_idx] = 1.0
        
        if hasattr(model_dict['le'], "inverse_transform"):
            try:
                pred_label = model_dict['le'].inverse_transform([pred])[0]
            except:
                pred_label = pred
        else:
            pred_label = pred
            
        class_names = model_dict.get('class_names', model_dict['le'].classes_ if hasattr(model_dict['le'], 'classes_') else [])
        
        confidence_scores = {}
        for i, score in enumerate(proba):
            class_name = class_names[i] if len(class_names) > i else f"Kelas {i}"
            confidence_scores[class_name] = float(score)
        
        return pred_label, confidence_scores
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return None, None

def show_disease_info(disease_name):
    """Menampilkan informasi detail tentang penyakit dengan warna yang sesuai"""
    info = DISEASE_INFO.get(disease_name, {
        "description": "Informasi tidak tersedia",
        "symptoms": "Tidak diketahui",
        "treatment": "Konsultasikan dengan ahli tanaman",
        "prevention": "Pemantauan rutin disarankan",
        "color": "#666666",
        "icon": "❓"
    })
    
    disease_color = info["color"]
    disease_icon = info["icon"]
    disease_class = f"{disease_name.lower().replace(' ', '-')}-color"
    
    with st.expander(f"{disease_icon} Informasi Detail tentang {disease_name}", expanded=True):
        st.markdown(f"""
            <div class='disease-header' style='--disease-color: {disease_color}'>
                <h3 style='color: {disease_color}; margin-bottom: 0.5rem;'>{disease_name}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Deskripsi", "Gejala", "Perawatan", "Pencegahan"])
        
        with tab1:
            st.markdown(f"**{disease_icon} Apa itu {disease_name}?**")
            st.info(info["description"])
            try:
                st.image(f"assets/{disease_name.lower().replace(' ', '_')}.jpg", 
                        use_container_width=True, 
                        caption=f"Contoh {disease_name}")
            except:
                colored_box = Image.new('RGB', (300, 200), color=disease_color)
                st.image(colored_box, use_container_width=True, caption="Ilustrasi")
        
        with tab2:
            st.markdown(f"**{disease_icon} Gejala Utama:**")
            st.markdown(f"<div style='background-color: {disease_color}20; padding: 1rem; border-radius: 8px;'>"
                        f"{info['symptoms']}</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown(f"**{disease_icon} Cara Mengatasi:**")
            st.markdown(f"<div style='background-color: {disease_color}10; padding: 1rem; border-radius: 8px;'>"
                        f"{info['treatment']}</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown(f"**{disease_icon} Tindakan Pencegahan:**")
            st.markdown(f"<div style='background-color: {disease_color}15; padding: 1rem; border-radius: 8px;'>"
                        f"{info['prevention']}</div>", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div class='header'>
            <h1>🌿 Deteksi Penyakit Daun Kentang</h1>
            <p>Identifikasi penyakit tanaman dengan kecerdasan buatan</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Animasi loading model
    with st.spinner('Memuat model AI...'):
        model_dict = load_model()
        time.sleep(1)
    
    if model_dict is None:
        return
    
    with st.sidebar:
        # Gunakan placeholder jika gambar tidak ada
        try:
            st.image("logo.png", use_container_width=True)
        except:
            st.image(Image.new('RGB', (300, 200), color='lightgray'), caption="Icon tanaman", use_container_width=True)
        
        st.header("ℹ️ Tentang Sistem")
        st.markdown("""
            **Teknologi:**
            - 🧠 Machine Learning (SVM)
            - 🖼️ Computer Vision
            - 🌈 Analisis Warna & Tekstur
            
            **Akurasi Sistem:** 89%
            
            **Penyakit yang Dikenali:**
            - Healthy
            - Bacteria
            - Fungi
            - Nematode
            - Pest
            - Phytophthora
            - Virus
        """)
        
        st.markdown("---")
        st.header("🛠️ Bantuan")
        st.info("""
            Unggah gambar daun tanaman yang ingin diperiksa. 
            Sistem akan menganalisis dan memberikan diagnosis.
        """)

        st.markdown("---")
        st.header("🔗 Drive Test Foto")
        st.info("""
            https://drive.google.com/drive/folders/1r-54JIZZovgGaPEDzbVgHS6kwCNqsyHm?usp=sharing
        """)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Unggah Gambar Daun")
        uploaded_file = st.file_uploader(
            "Seret dan lepas gambar atau klik untuk memilih",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
                
                # Analisis gambar
                with st.expander("🔍 Analisis Gambar"):
                    st.write(f"**Format:** {image.format}")
                    st.write(f"**Dimensi:** {image.size[0]} × {image.size[1]} piksel")
                    st.write(f"**Mode Warna:** {image.mode}")
                    
                    # Visualisasi channel warna
                    st.markdown("**Analisis Channel Warna:**")
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:  # Jika gambar berwarna
                        tabs = st.tabs(["Red", "Green", "Blue"])
                        for i, tab in enumerate(tabs):
                            with tab:
                                st.image(img_array[:,:,i], use_container_width=True, clamp=True)
            
            except Exception as e:
                st.error(f"Gagal memuat gambar: {str(e)}")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔬 Hasil Diagnosis")
            
            with st.spinner('Menganalisis gambar...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                pred_label, confidence_scores = predict_image(image, model_dict)
                
                if pred_label is not None:
                    # Dapatkan warna dan ikon penyakit
                    disease_info = DISEASE_INFO.get(pred_label, {})
                    disease_color = disease_info.get("color", "#4e73df")
                    disease_icon = disease_info.get("icon", "🌿")
                    
                    # Hasil utama dengan animasi
                    st.markdown(f"""
                        <div class='result-box'>
                            <h3 style='color:#2c3e50; margin-top:0;'>Diagnosis Utama</h3>
                            <h2 style='color: {disease_color};'>{disease_icon} {pred_label}</h2>
                            <p style='color:#7f8c8d;'>Sistem mendeteksi kondisi tanaman dengan akurasi tinggi</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualisasi kepercayaan
                    st.markdown("**Tingkat Keyakinan Sistem:**")
                    for class_name, score in confidence_scores.items():
                        confidence_percent = score * 100
                        disease_color = DISEASE_INFO.get(class_name, {}).get("color", "#4e73df")
                        
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{class_name}**")
                        with col_b:
                            st.write(f"{confidence_percent:.1f}%")
                        
                        st.markdown(
                            f"""<div class="confidence-bar" style="--confidence:{confidence_percent}%; --disease-color: {disease_color}"></div>""",
                            unsafe_allow_html=True
                        )
                    
                    # Informasi penyakit
                    show_disease_info(pred_label)
                    
                    # Rekomendasi tindakan
                    with st.expander("🚀 Rekomendasi Tindakan", expanded=True):
                        if pred_label == "Healthy":
                            st.success("""
                            ✅ **Lanjutkan perawatan rutin**  
                            • Pemupukan seimbang  
                            • Penyiraman teratur  
                            • Pemantauan berkala
                            """)
                        else:
                            st.warning(f"""
                            ⚠️ **Tindakan segera diperlukan**  
                            • Isolasi tanaman yang terinfeksi  
                            • Potong bagian yang sakit  
                            • Terapkan {disease_info.get('treatment', 'pengobatan yang sesuai')}  
                            • Tingkatkan sanitasi kebun
                            """)
    
    # Footer dengan efek
    st.markdown("""
        <div class='footer'>
            <p>Sistem Deteksi Penyakit Daun Kentang • © 2025</p>
            <div style='font-size:0.7rem; color:#aaa;'>
                Dibangun dengan Streamlit dan Scikit-learn
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
