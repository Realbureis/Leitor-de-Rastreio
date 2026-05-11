import streamlit as st
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
from pyzbar.pyzbar import decode
import cv2

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Jumbo CDP - Scanner Inteligente",
    page_icon="🚀",
    layout="wide"
)

# --- CARREGAMENTO DOS MODELOS (CACHE) ---
@st.cache_resource
def load_ia_models():
    # Carrega o YOLO treinado (seu arquivo best.pt)
    yolo_model = YOLO('best.pt')
    # Carrega o EasyOCR para Português (gpu=False se a VPS não tiver placa de vídeo)
    ocr_reader = easyocr.Reader(['pt'], gpu=False)
    return yolo_model, ocr_reader

try:
    model, reader = load_ia_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")
    st.stop()

# --- INTERFACE ---
st.title("🚀 Scanner Inteligente Jumbo CDP")
st.markdown("### IA para Pedidos e Extração de Dados (Sulfite A4)")

# Sidebar para ajustes finos
st.sidebar.header("⚙️ Configurações da IA")
conf_level = st.sidebar.slider("Confiança do YOLO", 0.1, 1.0, 0.45)
apply_filter = st.sidebar.checkbox("Aplicar Limpeza de Imagem (Binarização)", value=True)

# --- CAPTURA DE IMAGEM ---
st.markdown("---")
img_file = st.camera_input("Capture a folha de pedido")

if not img_file:
    img_file = st.file_uploader("Ou envie uma foto da sulfite", type=['jpg', 'jpeg', 'png'])

if img_file:
    # 1. Preparação da Imagem
    img_pil = Image.open(img_file)
    img_cv = np.array(img_pil)
    
    # Converter para escala de cinza para facilitar leitura do OCR e Barcode
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    with st.spinner("📦 Processando Pedido e Extraindo Dados..."):
        # --- PARTE 1: CÓDIGO DE BARRAS (PyZbar) ---
        barcodes = decode(img_pil)
        rastreio = barcodes[0].data.decode('utf-8') if barcodes else "Não encontrado"

        # --- PARTE 2: IA (DETECÇÃO DE CAMPOS YOLO) ---
        results = model.predict(img_pil, conf=conf_level)
        annotated_img = results[0].plot() # Imagem visual com os boxes

        # --- PARTE 3: EXTRAÇÃO DE TEXTO (OCR POR CAMPO) ---
        extracted_data = []
        
        # Iterar sobre cada caixa detectada pelo YOLO
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            # Adicionar margem (padding) para não cortar letras nas bordas
            pad = 5
            h, w = img_gray.shape
            crop = img_gray[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
            
            # Filtro de Limpeza (Opcional)
            if apply_filter:
                crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Executar o OCR no recorte do campo
            ocr_result = reader.readtext(crop, detail=0)
            texto_lido = " ".join(ocr_result)
            
            extracted_data.append({
                "campo": label.upper(),
                "texto": texto_lido
            })

    # --- EXIBIÇÃO DO RESULTADO ---
    st.markdown("---")
    col_img, col_data = st.columns([1, 1])

    with col_img:
        st.subheader("🔍 Visão da Inteligência")
        st.image(annotated_img, use_container_width=True, caption="Campos Identificados")

    with col_data:
        st.subheader("📝 Dados Extraídos")
        
        # Bloco de Rastreio
        st.metric("📦 CÓDIGO DE RASTREIO", rastreio)
        
        # Listagem dos Campos do Modelo YOLO
        if extracted_data:
            for item in extracted_data:
                with st.expander(f"Campo: {item['campo']}", expanded=True):
                    if item['texto'].strip():
                        st.info(item['texto'])
                    else:
                        st.warning("Campo detectado, mas texto ilegível.")
        else:
            st.error("Nenhum campo detectado. Verifique o enquadramento da folha sulfite.")

    # --- RODAPÉ TÉCNICO ---
    if st.checkbox("Ver Resumo Técnico (JSON)"):
        st.json({
            "rastreio": rastreio,
            "deteccoes": extracted_data
        })

st.markdown("---")
st.caption("Jumbo CDP Scanner v2.1 - Processamento via YOLOv8 + EasyOCR")
