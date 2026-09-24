"""VigiaGado MVP: regras extraídas e adaptadas do notebook YOLO11n fornecido."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import csv
import json
import os
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
_config_dir = ROOT / ".runtime" / "ultralytics"
_matplotlib_dir = ROOT / ".runtime" / "matplotlib"
_config_dir.mkdir(parents=True, exist_ok=True)
_matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_config_dir))
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_dir))

import cv2
import numpy as np
from ultralytics import YOLO


MODELS = ROOT / "models"
SAMPLES = ROOT / "samples"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70  # valor padrão do Ultralytics usado no baseline
ZONE_OVERLAP_THRESHOLD = 0.15  # interseção / área da bbox, não IoU
DEDUP_CONTAINMENT_THRESHOLD = 0.98
DEDUP_PARTIAL_CONTAINMENT_THRESHOLD = 0.95
DEDUP_PARTIAL_IOU_THRESHOLD = 0.60
DEDUP_MIN_AREA_RATIO = 0.40
DEDUP_MAX_CENTER_DISTANCE = 0.25  # fração da diagonal da maior caixa
IMAGE_SIZE = 960
PROCESS_FPS = 12
PERSISTENCE_FRAMES = 4  # 0,25 s com vídeo de 12 FPS, como no baseline.
DEFAULT_ENVIRONMENT = "cattle"
DEFAULT_DANGER_ZONE = (0.30, 0.70, 0.40, 0.94)
RURAL_CLASSES = {0: "cow", 1: "wild_boar", 2: "jaguar", 3: "wolf", 4: "person", 5: "dog"}
DEMO_VIDEOS = {
    "gado1": {"path": SAMPLES / "gado1.mp4", "environment": "cattle", "danger_zone": DEFAULT_DANGER_ZONE},
    "gado5": {"path": SAMPLES / "gado5.mp4", "environment": "cattle", "danger_zone": DEFAULT_DANGER_ZONE},
}
ENVIRONMENTS = {
    "cattle": {
        "expected": {"cow"}, "protected": {"cow"},
        "potential_threats": {"dog", "bear"}, "intruders": {"person"},
    },
    # Preparação apenas: não exposto na interface nem validado com modelo custom.
    # Neste contexto, dog fica OTHER até definir seu papel na propriedade.
    "rural_future": {
        "expected": {"cow"}, "protected": {"cow"},
        "potential_threats": {"wild_boar", "jaguar", "wolf"}, "intruders": {"person"},
    },
    # Estrutura para modelos futuros; o YOLO11n não detecta galinhas ou pragas específicas.
    "chicken_coop": {
        "expected": set(), "protected": set(),
        "potential_threats": {"dog", "bear"}, "intruders": {"person"},
    },
    "crop": {
        "expected": set(), "protected": set(),
        "potential_threats": set(), "intruders": {"person"},
    },
}


def rectangle_pixels(roi, frame_shape):
    """ROI normalizada (esquerda, topo, direita, base), relativa à imagem original."""
    if len(roi) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in roi):
        raise ValueError("ROI: informe quatro números entre 0 e 1.")
    left, top, right, bottom = roi
    if left >= right or top >= bottom:
        raise ValueError("ROI: esquerda < direita e topo < base.")
    h, w = frame_shape[:2]
    return (left * w, top * h, right * w, bottom * h)


def is_inside_zone(xyxy, rectangle):
    x1, _, x2, y2 = xyxy
    x, y = (x1 + x2) / 2, y2
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom


def bbox_zone_overlap_ratio(bbox, rectangle):
    """Fração da caixa coberta pela zona; contato sem área vale zero."""
    x1, y1, x2, y2 = bbox
    area = max(0, x2-x1) * max(0, y2-y1)
    if area == 0:
        return 0.0
    left, top, right, bottom = rectangle
    intersection = max(0, min(x2, right)-max(x1, left)) * max(0, min(y2, bottom)-max(y1, top))
    return intersection / area


def is_detection_in_zone(detection, rectangle):
    """Regra única para renderização, contagem, persistência e alertas."""
    bbox = detection["bbox"]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return False
    return (is_inside_zone(bbox, rectangle)
            or bbox_zone_overlap_ratio(bbox, rectangle) >= ZONE_OVERLAP_THRESHOLD)


def deduplicate_detections(detections):
    """Remove caixas quase contidas da mesma classe, apenas no frame atual.

    Mantém maior confiança. Não resolve oclusão nem identifica indivíduos.
    Limites conservadores evitam tratar sobreposição parcial como duplicidade.
    """
    kept = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        box = det["bbox"]
        area = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
        duplicate = False
        for other in kept:
            if det["class_name"] != other["class_name"]:
                continue
            other_box = other["bbox"]
            other_area = max(0, other_box[2]-other_box[0]) * max(0, other_box[3]-other_box[1])
            if min(area, other_area) <= 0:
                continue
            intersection = bbox_zone_overlap_ratio(box, other_box) * area
            iou = intersection / (area + other_area - intersection)
            containment = intersection / min(area, other_area)
            area_ratio = min(area, other_area) / max(area, other_area)
            bigger = box if area >= other_area else other_box
            diagonal = np.hypot(bigger[2]-bigger[0], bigger[3]-bigger[1])
            distance = np.hypot((box[0]+box[2]-other_box[0]-other_box[2])/2,
                                (box[1]+box[3]-other_box[1]-other_box[3])/2) / diagonal
            nested = (containment >= DEDUP_CONTAINMENT_THRESHOLD
                      or (containment >= DEDUP_PARTIAL_CONTAINMENT_THRESHOLD
                          and iou >= DEDUP_PARTIAL_IOU_THRESHOLD))
            if iou >= 0.85 or (nested
                              and area_ratio >= DEDUP_MIN_AREA_RATIO
                              and distance <= DEDUP_MAX_CENTER_DISTANCE):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def draw_zone(frame, detections, roi):
    """Preserva ROI e bottom-center do baseline; recebe detecções padronizadas."""
    view = frame.copy()
    rect = rectangle_pixels(roi, view.shape)
    flags = [is_detection_in_zone(d, rect) for d in detections]
    color = (75, 90, 245) if any(flags) else (45, 195, 245)
    layer = view.copy()
    cv2.rectangle(layer, tuple(map(int, rect[:2])), tuple(map(int, rect[2:])), color, -1)
    view = cv2.addWeighted(layer, 0.18, view, 0.82, 0)
    cv2.rectangle(view, tuple(map(int, rect[:2])), tuple(map(int, rect[2:])), color, 2)
    cv2.putText(view, "DANGER ZONE", (int(rect[0]), max(20, int(rect[1])-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for det, inside in zip(detections, flags):
        x1, y1, x2, y2 = map(int, det["bbox"])
        critical = det.get("category") in {"POTENTIAL_THREAT", "INTRUDER"}
        c = ((70, 90, 255) if critical else (0, 180, 255)) if inside else (70, 225, 120)
        cv2.rectangle(view, (x1, y1), (x2, y2), c, 2)
        cv2.putText(view, f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)
        cv2.circle(view, ((x1+x2)//2, y2), 5, c, -1)
    return view, sum(flags)


class TemporalState:
    """Confirma mudanças após persistência; uma transição gera um único evento."""
    def __init__(self, enter_s=0.25, clear_s=0.75):
        if enter_s < 0 or clear_s < 0:
            raise ValueError("Tempos de confirmação devem ser não negativos.")
        self.enter_s, self.clear_s = enter_s, clear_s
        self.active = False
        self.candidate_since = None

    def update(self, observed, time_s):
        observed = bool(observed)
        if observed == self.active:
            self.candidate_since = None
            return None
        if self.candidate_since is None:
            self.candidate_since = time_s
        delay = self.enter_s if observed else self.clear_s
        if time_s - self.candidate_since + 1e-9 < delay:
            return None
        change = {"active": observed, "onset_s": round(self.candidate_since, 4),
                  "confirmed_s": round(time_s, 4)}
        self.active = observed
        self.candidate_since = None
        return change


@lru_cache(maxsize=4)
def load_detector(backend="pt"):
    if backend not in {"pt", "onnx"}:
        raise ValueError("Modelo: escolha pt ou onnx.")
    path = MODELS / f"yolo11n.{backend}"
    if not path.is_file():
        raise FileNotFoundError(f"Modelo ausente: {path}. Execute setup_models.py.")
    model = YOLO(str(path), task="detect")
    names = model.names
    if not names or "cow" not in {str(v).lower() for v in names.values()}:
        raise ValueError("Modelo incompatível: classe cow não encontrada.")
    return model


def detect_objects(frame, detector, confidence=CONF_THRESHOLD, imgsz=IMAGE_SIZE):
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido.")
    result = detector.predict(frame, conf=confidence, iou=IOU_THRESHOLD, imgsz=imgsz,
                              device="cpu", verbose=False)[0]
    names = result.names
    detections = []
    for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(float, box)
        detections.append({
            "class_id": int(class_id), "class_name": str(names[int(class_id)]).lower(),
            "confidence": float(score), "bbox": [x1, y1, x2, y2],
            "center": [(x1+x2)/2, (y1+y2)/2],
            "bottom_center": [(x1+x2)/2, y2],
        })
    return detections


def validate_rural_classes(names):
    if names != RURAL_CLASSES:
        raise ValueError(f"Modelo rural incompatível: classes e IDs devem ser {RURAL_CLASSES}.")


def load_rural_detector(model_path):
    """Ponto de extensão para um best.pt real, ainda não integrado na UI.

    Requer ambiente Ultralytics compatível com o checkpoint futuro YOLO26n.
    Nunca baixa pesos nem substitui os arquivos do baseline.
    """
    path = Path(model_path).resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise FileNotFoundError("Informe um checkpoint rural .pt treinado e existente.")
    try:
        model = YOLO(str(path), task="detect")
    except Exception as exc:
        raise RuntimeError("Não foi possível carregar o modelo rural. Verifique o checkpoint e a versão do Ultralytics em ambiente separado.") from exc
    validate_rural_classes(model.names)
    return model


def classify(detection, environment=DEFAULT_ENVIRONMENT):
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Cenário desconhecido: {environment}")
    name = detection["class_name"]
    config = ENVIRONMENTS[environment]
    if name in config["expected"]:
        return "EXPECTED"
    if name in config["protected"]:
        return "PROTECTED"
    if name in config["potential_threats"]:
        return "POTENTIAL_THREAT"
    if name in config["intruders"]:
        return "INTRUDER"
    return "OTHER"


class AnalysisState:
    def __init__(self, fps, expected_count=None, environment=DEFAULT_ENVIRONMENT):
        self.fps = fps
        self.expected_count = expected_count
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Cenário desconhecido: {environment}")
        self.environment = environment
        self.zone = TemporalState((PERSISTENCE_FRAMES - 1) / fps, 0.75)
        self.threat = TemporalState((PERSISTENCE_FRAMES - 1) / fps, 0.75)
        self.divergence = TemporalState(2.0, 1.0)
        self.events = []

    def step(self, detections, roi, shape, time_s, frame_index):
        rect = rectangle_pixels(roi, shape)
        for det in detections:
            det["category"] = classify(det, self.environment)
            det["zone_overlap_ratio"] = bbox_zone_overlap_ratio(det["bbox"], rect)
            det["bottom_center_in_zone"] = is_inside_zone(det["bbox"], rect)
            det["in_zone"] = is_detection_in_zone(det, rect)
        counts = dict(Counter(d["class_name"] for d in detections))
        protected_in_zone = any(d["in_zone"] and d["category"] in {"EXPECTED", "PROTECTED"}
                                for d in detections)
        threats = [d for d in detections if d["category"] in {"POTENTIAL_THREAT", "INTRUDER"}]
        threat_in_zone = any(d["in_zone"] for d in threats)
        changes = [
            ("presenca_na_zona", self.zone.update(protected_in_zone, time_s)),
            ("ameaca_na_zona", self.threat.update(threat_in_zone, time_s)),
        ]
        if self.expected_count is not None:
            difference = counts.get("cow", 0) != self.expected_count
            changes.append(("divergencia_visual", self.divergence.update(difference, time_s)))
        new_events = []
        for kind, change in changes:
            if change:
                event = {"type": kind, "active": change["active"], "frame": frame_index,
                         "onset_s": change["onset_s"], "confirmed_s": change["confirmed_s"]}
                self.events.append(event)
                new_events.append(event)
        warning = bool(threats) or self.zone.active or self.divergence.active
        status = "DANGER" if self.threat.active else "WARNING" if warning else "NORMAL"
        return {"detections": detections, "counts": counts, "threats": len([d for d in threats if d["category"] == "POTENTIAL_THREAT"]),
                "intruders": len([d for d in threats if d["category"] == "INTRUDER"]),
                "zone_count": sum(d["in_zone"] for d in detections),
                "status": status, "alerts": new_events,
                "zone_active": self.zone.active, "divergence_active": self.divergence.active}


def process_frame(frame, detector, state, roi=DEFAULT_DANGER_ZONE,
                  confidence=CONF_THRESHOLD, imgsz=IMAGE_SIZE, time_s=0.0, frame_index=0):
    raw_detections = detect_objects(frame, detector, confidence, imgsz)
    detections = deduplicate_detections(raw_detections)
    result = state.step(detections, roi, frame.shape, time_s, frame_index)
    kept_ids = {id(det) for det in detections}
    result["raw_detection_count"] = len(raw_detections)  # após NMS do Ultralytics
    result["removed_detections"] = [d for d in raw_detections if id(d) not in kept_ids]
    rendered, _ = draw_zone(frame, detections, roi)
    header = np.full((84, rendered.shape[1], 3), (26, 33, 24), dtype=np.uint8)
    color = {"NORMAL": (95, 225, 130), "WARNING": (0, 195, 255), "DANGER": (60, 60, 255)}[result["status"]]
    cv2.putText(header, f"VIGIAGADO  |  {result['status']}", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    counts_text = ", ".join(f"{k}: {v}" for k, v in sorted(result["counts"].items())) or "nenhuma deteccao"
    cv2.putText(header, f"{time_s:.1f}s  |  {counts_text[:90]}  |  zona: {result['zone_count']}",
                (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2)
    result["frame"] = np.vstack((header, rendered))
    return result


def first_frame(source):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {source}")
    cap = cv2.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"Vídeo inválido ou sem frames: {source}")
    return frame


def preview_zone(source, roi):
    frame = first_frame(source)
    frame, _ = draw_zone(frame, [], roi)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("FFmpeg indisponível. Instale imageio-ffmpeg.") from exc


def normalize_video(source, destination, fps=PROCESS_FPS, width=1280):
    """Reutiliza a conversão CFR e o limite de largura do notebook original."""
    subprocess.run([_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-map", "0:v:0", "-an", "-vf", f"fps={fps},scale='min({width},iw)':-2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(destination)], check=True)


def _encode_h264(raw, destination):
    subprocess.run([_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True)


def summarize_reasons(records, events, expected_count):
    """Explica os estados observados usando os eventos já produzidos pelo core."""
    reasons = []
    active = {event["type"]: event for event in events if event["active"]}
    if "ameaca_na_zona" in active:
        event = active["ameaca_na_zona"]
        detections = records[event["frame"]]["detections"]
        names = sorted({d["class_name"] for d in detections
                        if d["in_zone"] and d["category"] in {"POTENTIAL_THREAT", "INTRUDER"}})
        kind = ", ".join(names) if names else "objeto relevante"
        reasons.append({"code": "THREAT_IN_ZONE", "time_s": event["confirmed_s"],
                        "message": f"Possível ameaça ou intruso ({kind}) confirmado na Danger Zone."})
    else:
        pending = next(((r, d) for r in records for d in r["detections"]
                        if d["in_zone"] and d["category"] in {"POTENTIAL_THREAT", "INTRUDER"}), None)
        if pending:
            record, detection = pending
            reasons.append({"code": "THREAT_PENDING", "time_s": record["time_s"],
                            "message": f"Possível ameaça ou intruso ({detection['class_name']}) na Danger Zone, sem persistência confirmada."})
    if "divergencia_visual" in active:
        event = active["divergencia_visual"]
        detected = records[event["frame"]]["counts"].get("cow", 0)
        reasons.append({"code": "COUNT_DIVERGENCE", "time_s": event["confirmed_s"],
                        "message": f"Divergência persistente na contagem de bovinos: esperados {expected_count}, detectados {detected} no momento da confirmação."})
    if "presenca_na_zona" in active:
        event = active["presenca_na_zona"]
        reasons.append({"code": "ZONE_EVENT", "time_s": event["confirmed_s"],
                        "message": "Presença de bovinos confirmada na Danger Zone."})
    for category, code, label in [("INTRUDER", "INTRUDER", "Possível intruso"),
                                  ("POTENTIAL_THREAT", "POTENTIAL_THREAT", "Possível ameaça")]:
        first = next(((r, d) for r in records for d in r["detections"]
                      if d["category"] == category and not d["in_zone"]), None)
        if first:
            record, detection = first
            reasons.append({"code": code, "time_s": record["time_s"],
                            "message": f"{label} ({detection['class_name']}) detectado fora da Danger Zone."})
    return reasons


def process_video(source, output_dir, backend="pt", environment=DEFAULT_ENVIRONMENT,
                  roi=DEFAULT_DANGER_ZONE, confidence=CONF_THRESHOLD,
                  expected_count=None, max_seconds=None, detector=None):
    source = Path(source).resolve()
    if environment != "cattle":
        raise ValueError("Nesta entrega, a análise de vídeo está validada somente para cattle.")
    if not 0 < confidence <= 1:
        raise ValueError("Confiança deve estar entre 0 e 1.")
    rectangle_pixels(roi, (720, 1280, 3))
    if expected_count is not None and (isinstance(expected_count, bool) or int(expected_count) != expected_count or expected_count < 0):
        raise ValueError("Contagem esperada deve ser um inteiro >= 0 ou vazia.")
    first_frame(source)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "VigiaGado_demo.mp4"
    if source == output:
        raise ValueError("A saída não pode substituir o vídeo original.")
    detector = detector or load_detector(backend)
    started = time.perf_counter()
    records = []
    with tempfile.TemporaryDirectory(prefix="vigiagado_") as temp:
        normalized = Path(temp) / "input.mp4"
        raw = Path(temp) / "render.mp4"
        normalize_video(source, normalized)
        cap = cv2.VideoCapture(str(normalized))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not 1 <= fps <= 120:
            cap.release()
            raise ValueError("Vídeo normalizado inválido.")
        state = AnalysisState(fps, expected_count)
        writer = None
        index = 0
        try:
            while True:
                if max_seconds is not None and index / fps >= max_seconds:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                result = process_frame(frame, detector, state, roi, confidence,
                                       time_s=index/fps, frame_index=index)
                rendered = result.pop("frame")
                if writer is None:
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (rendered.shape[1], rendered.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Falha ao iniciar codificador de vídeo.")
                writer.write(rendered)
                records.append({"frame": index, "time_s": round(index/fps, 4), **result})
                index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if not records:
            raise ValueError("Vídeo sem frames válidos.")
        _encode_h264(raw, output)
    with (output_dir / "contagens.csv").open("w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["frame", "time_s", "cow", "threats", "intruders", "zone_count", "status"])
        csv_writer.writerows([r["frame"], r["time_s"], r["counts"].get("cow", 0),
                              r["threats"], r["intruders"], r["zone_count"], r["status"]] for r in records)
    (output_dir / "eventos.json").write_text(json.dumps(state.events, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "detections.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    status_rank = {"NORMAL": 0, "WARNING": 1, "DANGER": 2}
    summary = {
        "video": source.name, "backend": backend, "model": str(MODELS / f"yolo11n.{backend}"),
        "model_label": "YOLO11n — ONNX" if backend == "onnx" else "YOLO11n — PyTorch (.pt)",
        "frames": len(records), "fps": fps, "duration_s": round(len(records)/fps, 3),
        "status": max((r["status"] for r in records), key=status_rank.get),
        "status_at_end": records[-1]["status"],
        "status_reasons": summarize_reasons(records, state.events, expected_count),
        "expected_count": expected_count,
        "inference_parameters": {"confidence": confidence, "nms_iou": IOU_THRESHOLD,
                                 "zone_overlap_threshold": ZONE_OVERLAP_THRESHOLD},
        "duplicates_removed": sum(len(r["removed_detections"]) for r in records),
        "counts_max": dict(Counter({k: max(r["counts"].get(k, 0) for r in records)
                                    for k in {name for r in records for name in r["counts"]}})),
        "threats_max": max(r["threats"] for r in records),
        "intruders_max": max(r["intruders"] for r in records),
        "events": state.events, "roi": list(roi), "elapsed_s": round(time.perf_counter()-started, 3),
        "output": str(output),
    }
    (output_dir / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
