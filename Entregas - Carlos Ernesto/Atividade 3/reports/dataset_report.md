# VigiaGado — dataset rural concluído

Data: 23/09/2026. Escopo desta etapa: aquisição e preparação de `wild_boar` (ID 1) e `jaguar` (ID 2), a partir dos dois CSVs de 500 candidatos que já estavam no checkpoint. Nenhum candidato novo foi selecionado. `wolf` foi adiado por causa da deadline, mantendo o ID 3 reservado. Não houve treinamento, fine-tuning, exportação ou comparação de modelos.

## Fontes e licença

| Classe | Fonte e categoria original | Licença declarada | Metadados usados |
|---|---|---|---|
| `wild_boar` | [SWG Camera Traps 2018–2020](https://lila.science/datasets/swg-camera-traps), `eurasian_wild_pig` | Community Data License Agreement, permissive variant | `dataset_staging/_metadata/wild_boar_annotations.json` (categoria 12) |
| `jaguar` | [WCS Camera Traps](https://lila.science/datasets/wcscameratraps), `panthera onca` | Community Data License Agreement, permissive variant | `dataset_staging/_metadata/jaguar_annotations.json` (categoria 24) |

As imagens foram baixadas individualmente pelas URLs públicas registradas em `dataset_rural/manifest.csv`. Os dois arquivos de anotações e seus ZIPs de origem foram preservados. As páginas oficiais da LILA descrevem as bounding boxes, os caminhos de download e as licenças acima. A fonte WCS alerta que parte dos rótulos de espécie é atribuída no nível da sequência; por isso, a amostra visual não substitui revisão manual integral.

## Resultado

| Classe | Candidatos | Imagens finais | Boxes | Grupos | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wild_boar` | 500 | 499 | 715 | 499 | 399 | 50 | 50 |
| `jaguar` | 500 | 499 | 526 | 438 | 399 | 51 | 49 |
| **Total** | **1.000** | **998** | **1.241** | **937** | **798** | **101** | **99** |

Os 998 downloads necessários foram concluídos. Houve dois descartes antes do download, um por classe, por bounding box claramente fora dos limites da imagem. Não houve falha de download final nem duplicata exata SHA-256. O arquivo `dataset_staging/_review/acquisition_log.csv` registra os 1.000 candidatos e seus resultados.

**Formato MPO:** 185 imagens da SWG chegaram como `.jpg`, mas eram identificadas como MPO pelo Pillow. A primeira imagem decodificada de cada arquivo foi convertida a JPEG convencional para compatibilidade; as dimensões originais foram mantidas. Os caminhos e IDs estão em `dataset_staging/_review/image_transforms.csv`. O SHA-256 no manifest se refere ao JPEG convertido nesses casos, não aos bytes remotos originais.

## Conversão e divisão

As caixas COCO `[x, y, width, height]` foram convertidas a YOLO normalizado a partir das dimensões do JSON da fonte. A pipeline rejeitou caixas degeneradas, não finitas e fora da imagem. Só foram aceitas imagens cujas anotações de bbox no JSON eram da categoria-alvo; nenhuma anotação de outra espécie foi descartada silenciosamente. Cada imagem tem um `.txt` correspondente. `dataset_rural/data.yaml` conserva o mapa planejado: `0 cow`, `1 wild_boar`, `2 jaguar`, `3 wolf`, `4 person`, `5 dog`.

O split foi atribuído por `group_id`, separadamente em cada classe, com semente fixa 42 e meta aproximada 80/10/10. A validação confirmou `train_groups ∩ val_groups = ∅`, `train_groups ∩ test_groups = ∅` e `val_groups ∩ test_groups = ∅`. O total é 798/101/99 imagens. O `group_id` WCS é derivado de sequências e não representa necessariamente identidade individual ou local; a ausência de grupo cruzado reduz vazamento por sequência, mas não elimina toda semelhança entre grupos de uma mesma câmera.

## Validação e auditoria visual

`python tools/validate_rural_staging.py` terminou com 0 erros e 0 avisos. `python tools/validate_rural_dataset.py` terminou com `passed: true`, 0 erros, 0 SHA-256 duplicados, 0 IDs originais duplicados e 0 grupos cruzados. A validação final abre e decodifica todas as imagens, confere dimensões, hashes, labels, IDs de classe, geometria das caixas, contagens e correspondência com o manifest. Resultado estruturado: `dataset_staging/_review/validation.json`.

Foram gerados contact sheets com caixas desenhadas para as duas classes e amostras separadas de train, val e test em `dataset_staging/_review/`. A inspeção visual das folhas gerais mostrou exemplos semanticamente compatíveis e caixas alinhadas nas amostras examinadas. Parte das imagens tem baixa iluminação, alvo pequeno, oclusão ou animal na borda, características da fonte de camera traps. O campo `reviewed=false` no manifest deixa explícito que **não houve revisão visual individual das 998 imagens**; as folhas são uma auditoria por amostra.

## Arquivos e reprodução

- Dataset final: `dataset_rural/images/{train,val,test}`, `dataset_rural/labels/{train,val,test}`, `dataset_rural/manifest.csv`, `dataset_rural/data.yaml`.
- Staging preservado: `dataset_staging/`, incluindo os dois CSVs de candidatos, imagens, labels, metadados, manifest e auditoria.
- Pipeline reproduzível: `python tools/complete_rural_dataset.py`; usa somente os CSVs existentes e reutiliza imagens já baixadas.
- Validação independente, sem download: `python tools/validate_rural_dataset.py`.
- Histórico do estágio anterior: `dataset_report_staging.md`.

Na etapa de preparação do dataset, o núcleo do MVP manteve os SHA-256 registrados em `core_sha256_checkpoint.json`. Posteriormente, `app.py` recebeu apenas alterações de apresentação autorizadas para a entrega acadêmica; a lógica permanece em `vigia_core.py`, que não foi modificada. Nenhum treinamento foi iniciado.
