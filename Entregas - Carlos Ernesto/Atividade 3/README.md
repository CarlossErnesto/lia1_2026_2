# VigiaGado - ONNX

Projeto acadêmico de visão computacional para monitoramento rural utilizando **YOLO11n, OpenCV, ONNX Runtime e Gradio**.

A aplicação recebe um vídeo MP4, detecta objetos, realiza contagem visual de bovinos e monitora uma região configurável denominada **Danger Zone**.

## 🌐 Demonstração

**Aplicação online:**

https://vigiagado-485431963319.southamerica-east1.run.app

> O serviço utiliza scale-to-zero no Google Cloud Run. Após algum tempo sem uso, a primeira abertura pode levar cerca de 1–2 minutos.

**Vídeo para teste:**

https://drive.google.com/drive/folders/1adq267fgVoDYq27qzS6FUbvkJsDJ8ysM?usp=sharing

### Como testar

1. Baixe o vídeo de exemplo.
2. Abra a aplicação.
3. Escolha **YOLO11n PyTorch** ou **YOLO11n ONNX**.
4. Faça upload do MP4.
5. Ajuste a Danger Zone, se desejar.
6. Clique em **Analisar vídeo**.

---

## Funcionalidades

- Detecção de objetos com YOLO11n.
- Inferência com **PyTorch (.pt)**.
- Inferência real com **ONNX Runtime (.onnx)**.
- Processamento de vídeo.
- Contagem estimada de bovinos.
- Danger Zone configurável.
- Persistência temporal.
- Geração de eventos.
- Estados `NORMAL`, `WARNING` e `DANGER`.
- Vídeo de saída anotado.

### Danger Zone

Uma detecção é considerada dentro da zona quando:

- o **bottom-center** da bounding box está dentro dela; ou
- pelo menos **15% da área da bounding box** está sobreposta à zona.

---

## Dataset rural

Também foi preparado um dataset para expansão futura do VigiaGado:

- **499 imagens de wild_boar (javali)**;
- **499 imagens de jaguar (onça-pintada)**;
- **998 imagens no total**;
- **1.241 bounding boxes**;
- formato YOLO;
- split por `group_id` para reduzir data leakage.

Fontes utilizadas:

- [SWG Camera Traps](https://lila.science/datasets/swg-camera-traps)
- [WCS Camera Traps](https://lila.science/datasets/wcscameratraps)

**Importante:** esse dataset foi preparado e validado, mas **não foi utilizado para treinamento nesta entrega**. Portanto, os modelos atuais ainda não possuem detecção customizada de javali e onça-pintada. :contentReference[oaicite:2]{index=2}

---


## Modelos

| Opção | Backend |
|---|---|
| YOLO11n PyTorch | PyTorch / Ultralytics |
| YOLO11n ONNX | ONNX Runtime |

Arquivos incluídos:

```text
models/yolo11n.pt
models/yolo11n.onnx
models/yolo11n.onnx.data

---

## Próximos passos

O VigiaGado foi desenvolvido como um MVP e possui espaço para expansão.

A principal evolução planejada é utilizar o **dataset rural preparado nesta atividade** para realizar fine-tuning de um modelo YOLO voltado à identificação de fauna potencialmente ameaçadora ao ambiente rural.

O dataset atual já contém exemplos anotados de:

- `jaguar` — onça-pintada, como exemplo de predador;
- `wild_boar` — javali, como animal potencialmente invasor ou de risco.

A evolução prevista é:

```text
Dataset rural
    ↓
Treinamento / fine-tuning YOLO
    ↓
Modelo rural customizado
    ↓
Detecção de javali e onça-pintada
    ↓
Integração com a Danger Zone
    ↓
Alertas específicos de ameaça
