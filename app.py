import streamlit as st
import easyocr
import numpy as np
import cv2
import re
from PIL import Image

# Configuração da página para parecer um App
st.set_page_config(page_title="Checkout Scanner", layout="centered", page_icon="📦")

# Inicializa o leitor de OCR (em português) - em cache para performance
@st.cache_resource
def load_ocr():
    # Isso baixa os modelos na primeira vez, pode demorar um pouco
    return easyocr.Reader(['pt'])

reader = load_ocr()

st.title("📦 Checkout Express")
st.write("Aponte a câmera para o cabeçalho da folha de pedido.")

# 1. CAPTURA DE IMAGEM
# O Streamlit ajusta automaticamente para a câmera traseira em celulares
img_file = st.camera_input("Scanear Folha")

if img_file:
    # Processamento da Imagem
    image = Image.open(img_file)
    img_array = np.array(image)
    
    with st.spinner("Lendo informações da folha..."):
        # 2. OCR - Leitura de todo o texto da imagem
        resultados = reader.readtext(img_array, detail=0)
        
        # Unimos tudo em uma string para facilitar a busca
        texto_completo = " ".join(resultados)
        
        # --- 3. INTELIGÊNCIA DE EXTRAÇÃO (REGEX ESPECÍFICO) ---
        
        # A) Pedido Nº: Procura por "Pedido N:" seguido de números
        # O EasyOCR pode ler "Nº" como "N", então prevemos isso
        match_pedido = re.search(r'Pedido N[°º:]\s*(\d+)', texto_completo, re.IGNORECASE)
        
        # B) Nome do Comprador: Procura o texto entre "Nome:" e "E-mail:"
        match_nome = re.search(r'Nome:\s*(.*?)\s*E-mail:', texto_completo, re.IGNORECASE)
        
        # C) Fone do Comprador: Procura por "Fone:" seguido de DDD e números
        match_fone = re.search(r'Fone:\s*(\(?\d{2}\)?\s?\d{4,5}-?\d{4})', texto_completo, re.IGNORECASE)

    # --- 4. INTERFACE DE CONFERÊNCIA ---
    st.subheader("Conferência de Dados")
    
    # Preenchemos o formulário com o que foi detectado ou vazio se falhar
    res_pedido = st.text_input("Nº Pedido Confirmado", value=match_pedido.group(1) if match_pedido else "")
    res_nome = st.text_input("Comprador Confirmado", value=match_nome.group(1).strip() if match_nome else "")
    res_fone = st.text_input("Fone Confirmado", value=match_fone.group(1) if match_fone else "")

    st.divider()
    
    # Campo para BIPAR o rastreio (O cursor deve ficar focado aqui automaticamente)
    st.subheader("Bipagem")
    res_rastreio = st.text_input("👉 BIPE O CÓDIGO DE RASTREIO AQUI", key="rastreio_focus")

    # Botão de finalizar envio para o n8n
    if st.button("✅ FINALIZAR E ENVIAR AO N8N", use_container_width=True):
        if res_pedido and res_rastreio:
            payload = {
                "pedido": res_pedido,
                "comprador": res_nome,
                "fone": res_fone,
                "rastreio": res_rastreio,
                "status": "checkout_concluido"
            }
            # Aqui você inseriria orequests.post() para o seu n8n
            st.success(f"Pedido {res_pedido} processado com sucesso!")
            st.balloons()
            # Limpa para o próximo (opcional, depende do fluxo)
        else:
            st.error("Erro: Pedido e Rastreio são obrigatórios!")

# Dica para melhorar a leitura
st.info("💡 Dica: Centralize bem o topo da folha na câmera e garanta boa iluminação para uma leitura instantânea.")
