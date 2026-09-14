# VigiaGado

O **VigiaGado** é um projeto acadêmico de visão computacional para monitoramento visual de bovinos.

O sistema transforma detecções em informações para apoio à decisão:

**vídeo → detecção → contagem/ROI → estados temporais → eventos → evidências**

## Versões do projeto

### 1. Baseline — YOLO11n

Arquivo: `01_VigiaGado_Baseline_YOLO11n.ipynb`

Primeira versão funcional do MVP, utilizando YOLO11n pré-treinado no COCO, sem fine-tuning específico para bovinos.

### 2. Versão final — YOLO26s

Arquivo: `02_VigiaGado_Final_YOLO26s.ipynb`

Versão especializada utilizando YOLO26s fine-tuned com um subconjunto público da classe **Cattle** do Open Images V7.

> A pasta `artifacts/` contém exclusivamente os artefatos da versão final YOLO26s.

## Dataset

- Open Images V7
- Classe: `Cattle`
- Obtenção via FiftyOne Dataset Zoo
- Train: 1200 imagens / 4919 boxes
- Validation: 134 imagens / 259 boxes
- Classe final: `0: bovino`

## Fine-tuning

- Modelo inicial: `yolo26s.pt`
- epochs: 40
- imgsz: 640
- patience: 10
- batch: -1
- seed: 42
- GPU: NVIDIA A100-SXM4-40GB

## Métricas do modelo final

| Métrica | Resultado |
|---|---:|
| Precision | 0.6824 |
| Recall | 0.5521 |
| mAP@50 | 0.5985 |
| mAP@50-95 | 0.4277 |

## Como executar

Para apresentação, mantenha:

```python
RETRAIN_MODEL = False
```

Disponibilize `artifacts/yolo26s_bovino_best.pt` e o vídeo de demonstração (gado5.mp4) e execute o notebook completo.

Para reproduzir o fine-tuning:

```python
RETRAIN_MODEL = True
```

## Estrutura do repositório

```text
VigiaGado/
├── 01_VigiaGado_Baseline_YOLO11n.ipynb
├── 02_VigiaGado_Final_YOLO26s.ipynb
├── README.md
├── gado5.mp4
└── artifacts/
    ├── yolo26s_bovino_best.pt
    ├── finetuned_metrics.json
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    ├── data.yaml
    ├── dataset_report.json
    └── ...
```

## Limitações

- contagem visual não representa o inventário completo da propriedade;
- oclusões, distância e iluminação podem afetar as detecções;
- Danger Zone é uma ROI 2D, não uma geofence física;
- não há tracking nem reconhecimento individual;
- o vídeo de demonstração não possui ground truth certificado para contagem.

## Fontes
https://www.youtube.com/watch?v=9ww7BPbNdc4&list=LL&index=6
