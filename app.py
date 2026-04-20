import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np
import easyocr

# Configuração de Interface
st.set_page_config(page_title="Scanner Jumbo CDP", layout="centered")
st.title("⚡ Scanner Inteligente Jumbo CDP")
st.write("IA para campos + Leitor de Rastreio")

# 1. Carregar Modelos (IA e Texto)
@st.cache_resource
def load_tools():
    model = YOLO('best.pt') # Seu cérebro treinado
    reader = easyocr.Reader(['pt']) # Leitor de texto em português
    return model, reader

model, reader = load_tools()

# 2. Ativar Câmera
picture = st.camera_input("Aponte para o Pedido")

if picture:
    img = Image.open(picture)
    img_np = np.array(img)

    # --- PARTE 1: CÓDIGO DE BARRAS (RASTREIO) ---
    barcodes = decode(img)
    st.subheader("📊 Código de Rastreio")
    if barcodes:
        for barcode in barcodes:
            codigo = barcode.data.decode("utf-8")
            st.success(f"✅ Rastreio Detectado: {codigo}")
    else:
        st.info("Nenhum código de barras na área.")

    # --- PARTE 2: IA + EXTRAÇÃO DE TEXTO ---
    st.subheader("🔍 Dados do Pedido")
    results = model(img)
    
    for result in results:
        # Mostra a imagem com os quadrados da IA
        st.image(result.plot(), caption="Detecção da IA")
        
        # Lógica para ler o texto dentro de cada quadrado detectado
        for box in result.boxes:
            # Coordenadas do quadrado
            xyxy = box.xyxy[0].cpu().numpy()
            # Corta a imagem apenas no campo detectado (ex: campo 'pedido')
            crop = img_np[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            
            # Lê o texto dentro desse corte
            txt_result = reader.readtext(crop, detail=0)
            
            label = model.names[int(box.cls[0])]
            texto_extraido = " ".join(txt_result)
            
            if texto_extraido:
                st.write(f"**{label.upper()}:** {texto_extraido}")
