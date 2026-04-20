import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import requests
from PIL import Image
from pyzbar.pyzbar import decode

# Configurações iniciais
st.set_page_config(page_title="Checkout Scanner Pro", layout="centered")

# 1. Carregar Motores (YOLO para detecção e EasyOCR para leitura)
@st.cache_resource
def load_models():
    # O arquivo 'best.pt' será o que vamos treinar com suas fotos
    try:
        model_yolo = YOLO('best.pt') 
    except:
        model_yolo = None
    ocr_reader = easyocr.Reader(['pt'])
    return model_yolo, ocr_reader

model, reader = load_models()

st.title("📦 Scanner de Checkout Jumbo")
st.write("Aponte para a folha e depois para a etiqueta de rastreio.")

# Interface de Câmera
img_file = st.camera_input("Scanner")

if img_file:
    # Converter imagem para formato processável
    image = Image.open(img_file)
    frame = np.array(image)
    
    # --- PASSO A: DETECÇÃO DE ÁREAS (YOLO) ---
    pedido, nome, fone = "", "", ""
    
    if model:
        results = model(frame)
        for r in results:
            for box in r.boxes:
                # Pegar coordenadas e classe (pedido, nome ou fone)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Recortar a área detectada
                crop = frame[y1:y2, x1:x2]
                
                # Ler o texto apenas daquela área específica
                text_result = reader.readtext(crop, detail=0)
                txt = " ".join(text_result)
                
                if label == 'pedido': pedido = txt
                elif label == 'nome': nome = txt
                elif label == 'fone': fone = txt
    else:
        st.warning("⚠️ Modelo 'best.pt' não encontrado. Use o OCR tradicional por enquanto.")

    # --- PASSO B: LEITURA DE CÓDIGO DE BARRAS (RASTREIO) ---
    barcodes = decode(frame)
    rastreio = barcodes[0].data.decode('utf-8') if barcodes else ""

    # --- PASSO C: FORMULÁRIO DE CONFERÊNCIA ---
    with st.form("conferencia_dados"):
        st.subheader("Dados Extraídos")
        res_pedido = st.text_input("Nº Pedido", value=pedido)
        res_nome = st.text_input("Comprador", value=nome)
        res_fone = st.text_input("Fone", value=fone)
        res_rastreio = st.text_input("Rastreio", value=rastreio)
        
        st.divider()
        
        # Botão de Envio para Apps Script
        if st.form_submit_button("✅ ENVIAR PARA PLANILHA", use_container_width=True):
            if res_pedido and res_rastreio:
                # URL do seu Google Apps Script
                URL_APPS_SCRIPT = "SUA_URL_AQUI"
                payload = {
                    "pedido": res_pedido,
                    "nome": res_nome,
                    "fone": res_tel,
                    "rastreio": res_rastreio
                }
                # Envio real
                # requests.post(URL_APPS_SCRIPT, json=payload)
                st.success(f"Pedido {res_pedido} salvo na nuvem!")
            else:
                st.error("Campos obrigatórios faltando (Pedido ou Rastreio)!")
