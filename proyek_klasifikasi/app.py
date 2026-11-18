import os
import gdown
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image

st.title("Klasifikasi Batu - Gunting - Kertas")
st.write("Unggah gambar tanganmu untuk diprediksi oleh model PyTorch!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1p0xXm2rrvERsEm5vcZxzLgHNuh72U3zK"
    output = "model_terbaik.pth"
    
    # Download otomatis dari Google Drive jika belum ada
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True)
        
    # Inisialisasi arsitektur & load state_dict
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(output, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    with st.spinner("Sedang mengunduh dan memuat model dari Google Drive..."):
        model = load_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("Siap diprediksi!")
