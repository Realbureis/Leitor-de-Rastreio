import gradio as gr
from ultralytics import YOLO
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np

# 1. Carregar o Modelo YOLO
model = YOLO('best.pt')

def scanner_jumbo(img):
    if img is None:
        return "Nenhuma imagem capturada", None
    
    # --- PARTE 1: CÓDIGO DE BARRAS ---
    barcodes = decode(img)
    rastreio = "Não encontrado"
    if barcodes:
        rastreio = barcodes[0].data.decode('utf-8')
    
    # --- PARTE 2: IA (DETECÇÃO DE CAMPOS) ---
    results = model(img)
    # Gera a imagem com os boxes desenhados
    annotated_img = results[0].plot()
    
    # --- PARTE 3: RESUMO ---
    labels_detectados = [model.names[int(box.cls[0])] for box in results[0].boxes]
    resumo = f"📦 Rastreio: {rastreio}\n"
    resumo += f"🔍 Campos identificados: {', '.join(set(labels_detectados))}"
    
    return resumo, annotated_img

# Interface do App
with gr.Blocks(title="Jumbo CDP Scanner") as demo:
    gr.Markdown("# 🚀 Scanner Inteligente Jumbo CDP")
    gr.Markdown("IA para pedidos + Leitor de Rastreio")
    
    with gr.Row():
        with gr.Column():
            # 'webcam' abre direto a câmera no celular
            input_img = gr.Image(source="webcam", type="pil", label="Capture a folha")
            btn = gr.Button("🔍 ESCANEAR PEDIDO", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Status do Processamento")
            output_img = gr.Image(label="Resultado da IA")

    btn.click(fn=scanner_jumbo, inputs=input_img, outputs=[output_text, output_img])

demo.launch()
