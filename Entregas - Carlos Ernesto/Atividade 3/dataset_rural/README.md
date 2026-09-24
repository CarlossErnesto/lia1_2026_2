# Dataset rural externo

Este repositório contém apenas `data.yaml` e `manifest.csv`. As 998 imagens e seus rótulos YOLO não estão aqui. A URL de hospedagem será publicada em `DATASET_URL_AQUI` no README principal.

O `manifest.csv` registra caminhos relativos à raiz `VigiaGado/`, fonte, licença declarada, classe, split, grupo, dimensões, hash SHA-256 e estado de revisão. Ao obter o dataset completo, extraia `images/` e `labels/` nesta pasta, preservando `train/`, `val/` e `test/`. Só então execute `python tools/validate_rural_dataset.py` na raiz do projeto.

O dataset não foi usado para treinar ou ajustar os modelos desta aplicação.
