import streamlit as st
import easyocr
import numpy as np
import cv2
import re
from PIL import Image
from pyzbar.pyzbar import decode  # Biblioteca para ler barras

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['pt'])

reader = load_ocr()

st.title("📦 Scanner Híbrido (Texto + Barras)")

img_file = st.camera_input("Scanear Folha e Etiqueta")

if img_file:
    image = Image.open(img_file)
    img_array = np.array(image)
    
    with st.spinner("Processando..."):
        # 1. Tenta ler Códigos de Barras (Rastreio)
        # Se o operador colocar a etiqueta de rastreio perto da folha, ele lê os dois de uma vez!
        barcodes = decode(img_array)
        rastreio_detectado = barcodes[0].data.decode('utf-8') if barcodes else ""

        # 2. OCR - Leitura de Texto (Pedido, Nome, Fone)
        resultados = reader.readtext(img_array, detail=0)
        texto_completo = " ".join(resultados)
        
        # Regex (Padrões que definimos antes)
        match_pedido = re.search(r'Pedido N[°º:]\s*(\d+)', texto_completo, re.IGNORECASE)
        match_nome = re.search(r'Nome:\s*(.*?)\s*E-mail:', texto_completo, re.IGNORECASE)
        match_fone = re.search(r'Fone:\s*(\(?\d{2}\)?\s?\d{4,5}-?\d{4})', texto_completo, re.IGNORECASE)

    # 3. INTERFACE DE CONFERÊNCIA
    with st.form("checkout"):
        st.subheader("Dados da Folha")
        res_pedido = st.text_input("Nº Pedido", value=match_pedido.group(1) if match_pedido else "")
        res_nome = st.text_input("Comprador", value=match_nome.group(1).strip() if match_nome else "")
        res_fone = st.text_input("Fone", value=match_fone.group(1) if match_fone else "")
        
        st.divider()
        st.subheader("Rastreio")
        # Se o código de barras foi lido, ele já aparece aqui preenchido!
        res_rastreio = st.text_input("Código de Rastreio", value=rastreio_detectado)

        if st.form_submit_button("🚀 ENVIAR TUDO AO N8N"):
            # Envio para o Webhook...
            st.success("Enviado!")
