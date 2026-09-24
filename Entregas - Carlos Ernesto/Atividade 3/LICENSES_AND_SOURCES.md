# Licenças e fontes

Este arquivo separa as situações dos componentes. **Nenhuma licença global foi atribuída ao projeto nesta entrega.** A presença de código ou dados neste pacote não substitui a análise dos termos de cada componente.

| Componente | Situação nesta entrega | Fonte/termos |
|---|---|---|
| Código próprio (`app.py`, `vigia_core.py`, scripts e documentação) | Titular e licença de publicação ainda não definidos neste checkpoint. É necessária uma decisão dos autores antes de conceder permissão geral de reutilização. | Código do projeto acadêmico VigiaGado. |
| Ultralytics e pesos YOLO11n (`.pt`, `.onnx`, `.onnx.data`) | O projeto depende de Ultralytics. A documentação oficial apresenta AGPL-3.0 e licença Enterprise; o uso e a distribuição devem respeitar a opção aplicável. Esta nota não concede licença aos pesos. | [Licenciamento Ultralytics](https://docs.ultralytics.com/help/contributing/). |
| Imagens/anotações SWG (`wild_boar`) | Fonte declara Community Data License Agreement, variante permissiva. Conferir condições e citação exigida antes de redistribuir o dataset completo. | [SWG Camera Traps na LILA](https://lila.science/datasets/swg-camera-traps). |
| Imagens/anotações WCS (`jaguar`) | Fonte declara Community Data License Agreement, variante permissiva. Conferir condições e citação exigida antes de redistribuir o dataset completo. | [WCS Camera Traps na LILA](https://lila.science/datasets/wcscameratraps). |
| Vídeo bovino usado na validação | Origem e permissão de redistribuição não confirmadas; **excluído** deste pacote e do deploy proposto. | Ver `reports/FINAL_VALIDATION_REPORT.md`. |
| Dependências Python | Cada biblioteca conserva seus próprios termos; `requirements.txt` identifica as versões/faixas utilizadas. | Metadados e licenças dos respectivos projetos. |

Os metadados de fonte e licença de cada imagem selecionada também constam em `dataset_rural/manifest.csv`. O dataset integral será distribuído separadamente somente após definir sua hospedagem e revisar os termos pertinentes. Para uso público do código/modelos Ultralytics, consulte a documentação oficial e tome uma decisão de licenciamento antes da publicação.
