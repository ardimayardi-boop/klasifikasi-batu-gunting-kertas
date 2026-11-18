import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Judul Aplikasi
st.title("Klasifikasi Batu - Gunting - Kertas")
st.write("Unggah gambar tanganmu untuk diprediksi oleh model PyTorch!")

# 2. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Fungsi Load Model (karena satu folder, cukup nama file-nya saja)
@st.cache_resource
def load_model():
    # Sesuaikan dengan arsitektur model yang kamu pakai saat training (contoh: ResNet18)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3) # 3 kelas: batu, gunting, kertas
    
    # Load bobot model
    model.load_state_dict(torch.load("model_terbaik.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    model = load_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 4. Upload Gambar dari User
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    # Tambahkan proses preprocessing & prediksi model di sini sesuai script aslimu
    st.write("Hasil prediksi akan muncul di sini...")
