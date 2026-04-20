import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np
import easyocr

# Configuração de Interface
st.set_page_config(page_title="Scanner Jumbo CDP", layout="centered")
st.title("⚡ Scanner Inteligente Jumbo CDP")
st.write("Identificação de campos + Código de Barras")

# 1. Carregar Modelos (IA e Texto) com Cache para não gastar energia à toa
@st.cache_resource
def load_tools():
    model = YOLO('best.pt') 
    reader = easyocr.Reader(['pt'])
    return model, reader

model, reader = load_tools()

# 2. Ativar Câmera ao vivo
picture = st.camera_input("Aponte para o Pedido e capture")

if picture:
    img = Image.open(picture)
    img_np = np.array(img)

    # --- PARTE 1: CÓDIGO DE BARRAS ---
    barcodes = decode(img)
    if barcodes:
        st.subheader("📊 Rastreio Detectado")
        for barcode in barcodes:
            st.success(f"CÓDIGO: {barcode.data.decode('utf-8')}")
    
    # --- PARTE 2: IA + OCR ---
    st.subheader("🔍 Dados Extraídos")
    results = model(img)
    
    for result in results:
        # Mostra a foto com os quadrados
        st.image(result.plot(), caption="Visualização da IA")
        
        # Extrai o texto de cada campo
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            # Corta a imagem (Crop)
            crop = img_np[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            
            # OCR no corte
            txt_result = reader.readtext(crop, detail=0)
            label = model.names[int(box.cls[0])]
            st.write(f"**{label.upper()}:** {' '.join(txt_result)}")
