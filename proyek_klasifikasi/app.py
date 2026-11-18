import os
import gdown
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.title("Klasifikasi Batu - Gunting - Kertas")
st.write("Unggah gambar tanganmu untuk diprediksi oleh model PyTorch!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1p0xXm2rrvERsEm5vcZxzLgHNuh72U3zK"
    output = "model_terbaik.pth"
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(output, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    with st.spinner("Memuat model..."):
        model = load_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    # Transformasi gambar agar sesuai dengan input ResNet (biasanya 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Daftar kelas (sesuaikan urutan kelas saat training, misal: Batu, Gunting, Kertas)
    class_names = ['Batu', 'Gunting', 'Kertas'] # Pastikan urutannya pas
    
    with st.spinner("Sedang memprediksi..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            result = class_names[class_idx]
            
    st.markdown(f"### Hasil Prediksi: **{result}** 🎉")
