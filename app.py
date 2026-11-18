import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Klasifikasi Batu Gunting Kertas",
    page_icon="✌️",
    layout="centered"
)

st.title("✌️ Klasifikasi Gerakan Tangan (Batu - Gunting - Kertas)")
st.write("Unggah gambar tangan (Batu, Gunting, atau Kertas) untuk memprediksi kelasnya menggunakan model ResNet18.")

# Urutan kelas sesuai ImageFolder (alfabetis: paper, rock, scissors)
CLASS_NAMES = ['paper', 'rock', 'scissors']
LABEL_ID = {
    'paper': 'Kertas (Paper) ✋',
    'rock': 'Batu (Rock) ✊',
    'scissors': 'Gunting (Scissors) ✌️'
}

# 2. Memuat Model Hasil Training
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Memuat bobot model terbaik
    model.load_state_dict(torch.load("model_terbaik.pth", map_location=device))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat file 'model_terbaik.pth'. Pastikan file berada di folder yang sama dengan app.py. Detail: {e}")
    st.stop()

# 3. Transformasi Gambar (Harus sama dengan saat validasi/test di Colab)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Antarmuka Upload Gambar atau Kamera
pilihan_input = st.radio("Pilih metode input gambar:", ("Unggah File Gambar", "Ambil Foto dari Kamera"))

image_source = None
if pilihan_input == "Unggah File Gambar":
    image_source = st.file_uploader("Pilih gambar (.jpg, .jpeg, .png)...", type=["jpg", "jpeg", "png"])
else:
    image_source = st.camera_input("Arahkan tangan ke kamera lalu ambil foto")

# 5. Proses Prediksi
if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Gambar yang diuji", use_container_width=True)

    with st.spinner("Sedang menganalisis gambar..."):
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_pct = confidence.item() * 100

    st.success(f"**Hasil Prediksi:** {LABEL_ID[predicted_class]}")
    st.info(f"**Tingkat Keyakinan (Confidence):** {confidence_pct:.2f}%")

    # Tampilkan rincian probabilitas semua kelas
    st.subheader("Rincian Probabilitas:")
    for idx, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[idx].item() * 100
        st.write(f"- {LABEL_ID[class_name]}: **{prob:.2f}%**")
        st.progress(int(prob))
