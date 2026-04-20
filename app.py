import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np

# Título focado na operação da Jumbo CDP
st.set_page_config(page_title="Scanner Vivo Jumbo", layout="centered")
st.title("⚡ Scanner em Tempo Real")
st.write("Aponte a câmera para o pedido e depois para o código de barras.")

# 1. Carregar o 'Cérebro' da IA
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

# 2. Câmera ao vivo (Abre direto no celular do funcionário)
picture = st.camera_input("Alinhe o pedido na marcação")

if picture:
    # Transformar a imagem da câmera para o formato que a IA entende
    img = Image.open(picture)
    img_array = np.array(img)

    # --- PASSO 1: DETECTAR CAMPOS DA FOLHA ---
    st.subheader("🔍 Identificação da IA")
    results = model(img)
    
    for result in results:
        # Mostra a imagem com os campos destacados (Unidade, Detento, etc.)
        res_plotted = result.plot()
        st.image(res_plotted, caption='Visualização da IA')

    # --- PASSO 2: IDENTIFICAR CÓDIGO DE BARRAS ---
    st.subheader("📊 Leitura de Rastreio")
    barcodes = decode(img)
    
    if barcodes:
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            st.success(f"🎯 **Rastreio Identificado:** {barcode_data}")
            # Dica: Aqui podemos colocar um botão para enviar ao n8n
            if st.button(f"Confirmar Checkout Pedido {barcode_data}"):
                st.info("Enviando dados para o sistema central...")
    else:
        st.warning("Aproxime a câmera do código de barras para capturar o rastreio.")
