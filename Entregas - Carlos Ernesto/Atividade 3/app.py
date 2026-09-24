"""Interface local Gradio para a demonstração acadêmica do VigiaGado."""

from pathlib import Path
from datetime import datetime
import json
import os
from urllib.parse import urlparse

import gradio as gr

from vigia_core import (DEMO_VIDEOS, DEFAULT_DANGER_ZONE, ROOT, CONF_THRESHOLD, ZONE_OVERLAP_THRESHOLD,
                        preview_zone, process_video)


MODEL_CHOICES = {
    "YOLO11n — PyTorch (.pt)": "pt",
    "YOLO11n — ONNX": "onnx",
}
CSS = """
.gradio-container {max-width: 1160px !important; margin: auto;}
#hero {padding: 28px 32px; border-radius: 18px; background: linear-gradient(135deg, #123d31, #276b4b); color: #f5fff8;}
#hero h1, #hero p {color: #f5fff8 !important; margin: 0;}
#hero h1 {font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.1; margin: 9px 0 12px;}
#hero p {max-width: 720px; line-height: 1.55;}
#hero .eyebrow {color: #c8e9cf; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}
#overview {margin: 8px 0 16px;}
#overview .facts {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;}
#overview .fact {padding: 14px 18px; border: 1px solid #dce8de; border-radius: 12px; background: #f5faf6; color: #183e2e;}
#overview .fact strong {display: block; margin-bottom: 4px; font-size: 1.05rem;}
#overview .fact span {font-size: .9rem;}
#status-panel, #metrics-panel {padding: 16px 20px; border: 1px solid #d4e3d6; border-radius: 12px; background: #f8fbf8;}
#status-panel h3, #metrics-panel h3 {margin-top: 0;}
#run-button {margin: 10px 0 2px;}
@media (max-width: 700px) {#hero {padding: 22px;} #overview .facts {grid-template-columns: 1fr;}}
"""


def project_links():
    links = [
        "[Dataset SWG (javalis)](https://lila.science/datasets/swg-camera-traps)",
        "[Dataset WCS (onças)](https://lila.science/datasets/wcscameratraps)",
    ]
    github = os.getenv("VIGIAGADO_GITHUB_URL", "").strip()
    parsed = urlparse(github)
    if parsed.scheme == "https" and parsed.netloc.lower() in {"github.com", "www.github.com"}:
        links.insert(0, f"[Código no GitHub]({github})")
    return " · ".join(links)


def source_path(preset, upload):
    if upload:
        path = Path(upload)
        if path.suffix.lower() != ".mp4":
            raise ValueError("Envie um arquivo .mp4.")
        return path
    if not any(spec["path"].exists() for spec in DEMO_VIDEOS.values()):
        raise ValueError("Envie um vídeo MP4 para análise.")
    if preset not in DEMO_VIDEOS:
        raise ValueError("Selecione um vídeo de demonstração.")
    return DEMO_VIDEOS[preset]["path"]


def update_preset(preset):
    zone = DEMO_VIDEOS[preset]["danger_zone"]
    path = DEMO_VIDEOS[preset]["path"]
    preview = preview_zone(path, zone) if path.exists() else None
    return (*zone, preview, None)


def update_preview(preset, upload, x1, y1, x2, y2):
    try:
        return preview_zone(source_path(preset, upload), (x1, y1, x2, y2))
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc))


def format_events(events):
    if not events:
        return "Nenhum evento persistente registrado."
    labels = {
        "divergencia_visual": "Divergência na contagem",
        "presenca_na_zona": "Presença de bovinos na Danger Zone",
        "ameaca_na_zona": "Ameaça ou intruso na Danger Zone",
    }
    lines = []
    for event in events:
        action = "confirmada" if event["active"] else "encerrada"
        lines.append(f"- **{event['confirmed_s']:.2f} s** — {labels.get(event['type'], event['type'])} {action}.")
    return "\n".join(lines)


def format_result(result):
    reasons = result["status_reasons"]
    reason_text = "\n".join(f"- {item['message']}" for item in reasons)
    if not reason_text:
        reason_text = "Nenhum evento crítico confirmado."
    status = (f"### Status: {result['status']}\n"
              f"**Motivo:**\n{reason_text}\n\n"
              f"*O status resume o nível mais alto observado no vídeo; ao final: {result['status_at_end']}.*")
    expected = "desativada" if result["expected_count"] is None else str(result["expected_count"])
    metrics = (f"### Resultado da análise\n"
               f"**Modelo utilizado:** {result['model_label']}  \n"
               f"**Bovinos detectados (máximo em um frame):** {result['counts_max'].get('cow', 0)}  \n"
               f"**Bovinos esperados:** {expected}  \n"
               f"**Ameaças (máximo):** {result['threats_max']}  \n"
               f"**Intrusos (máximo):** {result['intruders_max']}")
    technical = {key: value for key, value in result.items() if key != "output"}
    return status, metrics, format_events(result["events"]), json.dumps(technical, ensure_ascii=False, indent=2)


def run_analysis(model_choice, preset, upload, x1, y1, x2, y2, confidence, max_seconds,
                 expected_count, progress=gr.Progress()):
    try:
        backend = MODEL_CHOICES.get(model_choice)
        if backend is None:
            raise ValueError("Selecione um dos modelos YOLO11n disponíveis.")
        source = source_path(preset, upload)
        progress(0.05, desc="Validando vídeo e configuração...")
        run_dir = ROOT / "resultados" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        progress(0.10, desc=f"Processando com {model_choice}...")
        result = process_video(source, run_dir, backend=backend,
                               roi=(x1, y1, x2, y2), confidence=confidence,
                               max_seconds=max_seconds or None, expected_count=expected_count)
        progress(1.0, desc="Análise concluída")
        return result["output"], *format_result(result)
    except Exception as exc:
        raise gr.Error(str(exc))


def build_app():
    available = [name for name, spec in DEMO_VIDEOS.items() if spec["path"].exists()]
    default = "gado5" if "gado5" in available else available[0] if available else None
    video_choices = [(f"Vídeo de demonstração — {name}.mp4", name) for name in available]
    with gr.Blocks(title="VigiaGado") as app:
        gr.Markdown("""<div class="eyebrow">Projeto acadêmico · Visão computacional</div>
<h1>VigiaGado</h1>
<p>Monitoramento visual de áreas rurais. Analise um vídeo, acompanhe os bovinos detectados e veja quando a presença na Danger Zone ou a divergência de contagem gera um alerta.</p>""", elem_id="hero")
        gr.Markdown("""<div class="facts">
<div class="fact"><strong>2 formatos</strong><span>YOLO11n PyTorch e ONNX</span></div>
<div class="fact"><strong>3 estados</strong><span>NORMAL · WARNING · DANGER</span></div>
<div class="fact"><strong>Upload MP4</strong><span>Envie um vídeo autorizado para análise</span></div>
</div>""", elem_id="overview")

        gr.Markdown("## 1. Escolha o vídeo e o modelo")
        gr.Markdown("Envie um vídeo MP4 autorizado. Para uma demonstração rápida, mantenha o limite de 12 segundos. Se houver um vídeo de exemplo local, o upload o substitui.")
        with gr.Row():
            model = gr.Radio(list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0], label="Modelo de inferência")
            preset = gr.Dropdown(video_choices, value=default, label="Vídeo de demonstração",
                                 visible=bool(video_choices))
        with gr.Row():
            upload = gr.File(label="Upload de vídeo (.mp4)", file_types=[".mp4"], type="filepath")
            gr.Textbox(value="Pastagem / Bovinos", label="Cenário demonstrado", interactive=False)

        gr.Markdown("## 2. Ajuste a Danger Zone")
        gr.Markdown(f"A região destacada delimita a área monitorada. A entrada é considerada quando o ponto inferior central da caixa está dentro da zona ou quando a sobreposição atinge pelo menos **{ZONE_OVERLAP_THRESHOLD:.0%} da área da caixa**.")
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.Image(label="Prévia da zona no vídeo selecionado", type="numpy")
            with gr.Column(scale=2):
                with gr.Row():
                    x1 = gr.Slider(0, 1, value=DEFAULT_DANGER_ZONE[0], step=0.01, label="X1 — esquerda")
                    x2 = gr.Slider(0, 1, value=DEFAULT_DANGER_ZONE[2], step=0.01, label="X2 — direita")
                with gr.Row():
                    y1 = gr.Slider(0, 1, value=DEFAULT_DANGER_ZONE[1], step=0.01, label="Y1 — topo")
                    y2 = gr.Slider(0, 1, value=DEFAULT_DANGER_ZONE[3], step=0.01, label="Y2 — base")
                gr.Markdown("Coordenadas normalizadas de 0 a 1. Mantenha **X1 < X2** e **Y1 < Y2**.")

        gr.Markdown("## 3. Execute a análise")
        with gr.Row():
            confidence = gr.Slider(0.05, 0.95, value=CONF_THRESHOLD, step=0.05, label="Confiança mínima")
            max_seconds = gr.Slider(1, 60, value=12, step=1, label="Trecho a processar (s)")
            expected_count = gr.Number(value=7, precision=0, label="Bovinos esperados (opcional)")
        gr.Markdown("Alertas exigem persistência temporal. A referência de bovinos ajuda a identificar divergência de contagem; não altera as detecções.")
        run = gr.Button("Analisar vídeo", variant="primary", size="lg", elem_id="run-button")

        gr.Markdown("## 4. Veja o resultado")
        with gr.Row():
            status = gr.Markdown("Execute uma análise para visualizar o status e o motivo.", elem_id="status-panel")
            metrics = gr.Markdown("O modelo utilizado e as contagens aparecerão aqui.", elem_id="metrics-panel")
        output = gr.Video(label="Vídeo processado com detecções e Danger Zone")
        gr.Markdown("### Eventos registrados")
        events = gr.Markdown("Nenhum processamento realizado.")
        with gr.Accordion("Detalhes técnicos (JSON)", open=False):
            technical = gr.Code(label="Resumo técnico", language="json")

        gr.Markdown("## Sobre o projeto")
        gr.Markdown("""**Como funciona:** detecção → contexto → Danger Zone → persistência → alerta. O sistema filtra caixas duplicadas no mesmo frame e resume o maior estado observado. As contagens são estimativas visuais; `WARNING` pode indicar bovinos na zona ou diferença persistente de contagem.

**Modelos disponíveis:** YOLO11n PyTorch usa `models/yolo11n.pt`. YOLO11n ONNX usa `models/yolo11n.onnx` e `models/yolo11n.onnx.data` por ONNX Runtime. A seleção acima muda o backend da mesma análise.

**Dataset rural:** 998 imagens anotadas de javali e onça foram preparadas separadamente. A demonstração desta página usa o cenário bovino; nenhum modelo rural foi treinado ou integrado.""")
        gr.Markdown(f"**Fontes e código:** {project_links()}")

        preset.change(update_preset, inputs=[preset], outputs=[x1, y1, x2, y2, preview, upload])
        for control in [upload, x1, y1, x2, y2]:
            control.change(update_preview, inputs=[preset, upload, x1, y1, x2, y2], outputs=preview)
        run.click(run_analysis,
                  inputs=[model, preset, upload, x1, y1, x2, y2, confidence, max_seconds, expected_count],
                  outputs=[output, status, metrics, events, technical], show_progress="full")
        if default:
            app.load(update_preset, inputs=[preset], outputs=[x1, y1, x2, y2, preview, upload])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                       inbrowser=False, css=CSS)
