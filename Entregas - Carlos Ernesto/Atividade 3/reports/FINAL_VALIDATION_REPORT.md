# VigiaGado — validação final do estado atual

Data: 23/09/2026. Esta etapa validou o checkpoint atual, sem treinamento, sem YOLO26, sem deploy e sem gerar pacote de publicação.

## 1. Status geral

**PASS WITH LIMITATIONS.** O MVP é funcional e reproduzível no ambiente Windows/Python 3.12 testado: a aplicação inicia, Gradio abre, os dois modelos reais processam vídeo, a Danger Zone produz eventos, o upload funciona e os cinco testes passam. O projeto está tecnicamente pronto para a **etapa de preparação** de um pacote acadêmico. A publicação externa ainda requer conferir os direitos do vídeo de demonstração e as obrigações de licença do Ultralytics, além de um teste no Render/Linux. Nenhum desses itens é uma falha observada na execução local.

## 2. Ambiente e comandos executados

- Windows, Python **3.12.14**, venv isolado em `work/ui-venv` fora do projeto; teste de interface servido em `http://127.0.0.1:7862/` com `PORT=7862`.
- `python -m pip install -r requirements.txt`: **PASS**, instalação concluída. `python -m pip check`: **PASS**, nenhuma dependência quebrada.
- `python -m unittest discover -s tests -v`: **PASS**, 5 testes, 0 falhas.
- `python setup_models.py`: **PASS** em cópia isolada do projeto, para não sobrescrever `models/comparacao_pt_onnx.json` do checkpoint.
- `python app.py`: **PASS** na mesma cópia, com `PORT=7862`; resposta HTTP 200, Gradio aberto em navegador.
- `process_video(..., backend='pt')` e `process_video(..., backend='onnx')`: **PASS** em trechos de 8 s do vídeo incluído.
- A interface Gradio executou PT e ONNX em trechos de 2 s; a API da própria interface recebeu upload MP4 e parâmetros alterados, processou e retornou vídeo/status/JSON.

As versões efetivamente instaladas foram Ultralytics 8.3.203, PyTorch 2.14.0, torchvision 0.29.0, OpenCV 4.11.0.86, ONNX 1.23.0, ONNX Runtime 1.30.0, onnxscript 0.7.2, Gradio 6.28.0 e imageio-ffmpeg 0.6.0. `render.yaml` instala PyTorch 2.5.1 e torchvision 0.20.1 para CPU; essa combinação no Linux não foi executada nesta validação.

## 3. Interface Gradio — PASS

A página abriu com textos legíveis em português e controles de modelo, vídeo de exemplo, upload MP4, ROI, confiança, duração e bovinos esperados. O seletor acionou os dois backends, e cada execução exibiu status, contagens, eventos e player com download do vídeo. O upload de um MP4 de 2 s pela API Gradio funcionou com ROI `(0,20; 0,65; 0,50; 0,95)`, confiança `0,30`, duração `2 s` e esperado `7`. Não houve exceção visível. A resposta retornou `WARNING` e vídeo processado.

## 4. PyTorch — PASS

`models/yolo11n.pt` foi encontrado, carregado e usado em inferência real. Na comparação do primeiro frame, houve 8 detecções. O processamento de 8 s produziu 96 frames, vídeo H.264 decodificável, CSV/JSON, 4 eventos, máximo de 1 bovino na zona e status `WARNING`. O vídeo de saída tem 1.953.981 bytes. O teste da interface exibiu `YOLO11n — PyTorch (.pt)`.

## 5. ONNX — PASS

`models/yolo11n.onnx` e o arquivo externo `models/yolo11n.onnx.data` foram encontrados. O ONNX Runtime carregou o grafo com `CPUExecutionProvider`; `setup_models.py` registrou explicitamente carregamento de `yolo11n.onnx` por ONNX Runtime. A comparação no primeiro frame produziu 8 detecções PT e 8 ONNX, com 8 pares e IoU médio das caixas de 0,9625. O processamento ONNX de 8 s produziu 96 frames, vídeo H.264 decodificável, CSV/JSON, 4 eventos, máximo de 1 bovino na zona e status `WARNING`. O vídeo tem 1.967.982 bytes. A interface exibiu `YOLO11n — ONNX` após a seleção. Não houve simulação ONNX por PT.

## 6. Vídeo, Danger Zone e estados — PASS WITH LIMITATIONS

O MP4 incluído foi lido, normalizado para 12 FPS, processado frame a frame, anotado com boxes e zona, codificado em H.264 e reaberto/decodificado. A contagem, deduplicação, persistência e os eventos aparecem nos resumos e arquivos exportados. A regra `bottom-center` **ou** interseção de pelo menos 15% da área da bbox permanece em `is_detection_in_zone`; os testes cobrem limites, overlap, persistência e vizinho ocluído. PT e ONNX produziram `WARNING` por bovinos na zona. O cenário demonstrado não contém ameaça real; `DANGER` foi coberto apenas por regras/fixtures anteriores, não por vídeo real desta validação. Contagem e presença são estimativas visuais, sem tracking.

## 7. Requirements — PASS WITH LIMITATIONS

As nove entradas de `requirements.txt` instalaram sem conflito. Gradio, Ultralytics, PyTorch, OpenCV, ONNX Runtime e imageio-ffmpeg foram exercitados; torchvision é dependência da pilha PyTorch/Ultralytics, e `onnx`/`onnxscript` apoiam a rotina opcional de exportação em `setup_models.py`. Não foi encontrada dependência faltante ou pacote manual necessário ao fluxo testado. As faixas de versão de PyTorch, torchvision, ONNX e onnxscript permitem atualizações futuras, então a reprodução exata do ambiente não é garantida sem lock; não houve motivo para alterar versões nesta etapa.

## 8. Paths e portabilidade — PASS WITH LIMITATIONS

Os caminhos de modelos, amostra, configuração de runtime e resultados derivam de `Path(__file__).resolve().parent`; `app.py` usa `0.0.0.0` e `PORT` (padrão 7860). O upload usa o caminho entregue pelo Gradio. A busca em `app.py`, `vigia_core.py`, `setup_models.py`, `requirements.txt`, `render.yaml` e README não encontrou referência operacional a `C:\Users`, `/content` ou diretórios temporários fixos. `gado1.mp4` é opcional; `gado5.mp4` está presente. O notebook Colab contém caminhos próprios do Colab, fora da execução da aplicação. A portabilidade foi verificada por código e execução Windows; não houve execução efetiva em Linux/Render.

## 9. Modelos necessários — PASS

| Arquivo em `models/` | Bytes | Uso |
|---|---:|---|
| `yolo11n.pt` | 5.613.764 | Inferência PyTorch |
| `yolo11n.onnx` | 11.266.428 | Grafo ONNX |
| `yolo11n.onnx.data` | 10.682.368 | Dados externos exigidos pelo grafo ONNX |

Os três arquivos são necessários para oferecer ambos os seletores. `models/comparacao_pt_onnx.json` é um relatório de comparação, não um peso de execução. Não foi encontrado outro checkpoint de modelo ativo ou redundante no projeto. Os hashes SHA-256 de `vigia_core.py`, `setup_models.py`, os três modelos e os dois notebooks continuam iguais ao registro em `core_sha256_checkpoint.json`; `app.py` difere por alterações de interface/deploy já autorizadas.

## 10. Metadados do dataset — PASS WITH LIMITATIONS

`dataset_rural/data.yaml`, `dataset_rural/manifest.csv` e `dataset_report.md` existem. O YAML registra `cow=0`, `wild_boar=1`, `jaguar=2`, `wolf=3` reservado, `person=4`, `dog=5`. O manifesto contém 998 linhas: 499 de javali e 499 de onça, com splits 798/101/99; os 998 pares de imagem/rótulo referenciados existem, resolvendo os caminhos a partir da raiz do projeto. `dataset_staging/_review/validation.json` registra `passed=true`, 1.241 boxes, zero erros, zero SHA-256 duplicados e zero vazamento de grupos. Esta etapa conferiu metadados e existência de arquivos; não repetiu a decodificação e o hash de todas as imagens. O manifesto marca `reviewed=false` para todas: não houve revisão visual individual. README, página e `dataset_report.md` afirmam corretamente que o modelo atual **não** foi treinado com esse dataset.

## 11. Testes e regressões — PASS

Os cinco testes em `tests/test_corrections.py` passaram, sem falhas. O relatório histórico `RELATORIO_CORRECOES.md` documenta 326 frames processados por backend; a amostra atual de 96 frames reproduziu o status `WARNING`, detecção de bovinos na zona e saída válida em ambos. A mudança recente de apresentação não alterou `vigia_core.py` nem os modelos, conforme hashes. A validação anterior de interface usava core simulado; nesta etapa os dois seletores e o upload foram exercitados com inferência real. Não apareceu regressão funcional.

## 12. Arquivos grandes e recomendação para publicação

| Caminho | Tamanho aprox. | Classificação | GitHub/deploy |
|---|---:|---|---|
| `dataset_staging/` | 1.523,33 MiB | Staging/metadados/auditoria | Excluir do deploy; hospedar separadamente |
| `dataset_rural/` | 1.198,16 MiB | Dataset final | Excluir do deploy; hospedar separadamente |
| `dataset_staging/_metadata/jaguar_annotations.json` | 220,89 MiB | Metadados de origem/staging | Inadequado para Git comum; excede o limite por arquivo do GitHub |
| `dataset_staging/_metadata/wild_boar_annotations.json` | 72,47 MiB | Metadados de origem/staging | Inadequado para repositório leve; gera aviso de tamanho |
| `validation_corrections/` | 38,47 MiB | Evidência histórica/backup de validação | Opcional, não necessário para execução |
| `demo_pt/` + `demo_onnx/` | 13,78 MiB | Demonstrações históricas | Redundantes para o deploy, preservar no checkpoint |
| `models/` | 26,29 MiB | Modelos e comparação | Incluir os três arquivos de modelo para PT/ONNX |
| `samples/gado5.mp4` | 7,92 MiB | Demonstração atual | Necessário para demonstração sem upload; conferir direito de redistribuição |

Há 4.152 arquivos no projeto atual. O `.gitignore` já exclui dataset, staging, demos históricos, validações e resultados para novos repositórios. A pasta atual não é um repositório Git; portanto não foi possível verificar índice ou histórico de versionamento. O GitHub bloqueia arquivos acima de 100 MiB em Git comum: [documentação oficial](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Nada foi apagado.

## 13. Problemas encontrados e correções realizadas

- **Corrigido:** o comentário inicial de `dataset_rural/data.yaml` dizia que o dataset estava vazio. Apenas esse comentário foi atualizado para refletir as imagens e os caminhos relativos; IDs de classes e caminhos não mudaram.
- **Não bloqueante, pendente:** `RELATORIO_CORRECOES.md` é histórico e ainda descreve o dataset como vazio. O relatório mais recente `dataset_report.md` e os arquivos reais são a referência para o estado atual. Nenhum relatório histórico foi reescrito.
- **Não bloqueante, pendente:** o README declara que a licença de redistribuição de `samples/gado5.mp4`, que contém marca d'água, não foi verificada. Antes de publicar o vídeo em GitHub/Render, confirme sua origem/licença ou use um substituto autorizado.
- **Não bloqueante, pendente:** verificar o regime de licença aplicável à publicação do código/modelos Ultralytics. A documentação oficial descreve as opções [AGPL-3.0 e Enterprise](https://docs.ultralytics.com/help/contributing/). Esta validação não determina conformidade jurídica de uma distribuição futura.
- **Não bloqueante, pendente:** executar build e teste funcional no Render/Linux com o plano de memória escolhido. O teste atual é Windows local; não comprova memória, disponibilidade ou persistência no serviço hospedado.

Nenhum bug de execução exigiu alteração no core, na interface, nos requisitos ou nos modelos. Não foi criado ZIP, repositório, deploy ou pacote final nesta etapa.

## 14. Checklist final

- [x] Aplicação inicia — **PASS**
- [x] Gradio abre — **PASS**
- [x] PT funciona — **PASS**
- [x] ONNX funciona — **PASS**
- [x] Seletor PT/ONNX funciona — **PASS**
- [x] Vídeo processa — **PASS**
- [x] Danger Zone funciona — **PASS**, com limitação de cenário real
- [x] Output é gerado — **PASS**
- [x] Requirements está completo — **PASS**, sem lock exato
- [x] Paths são portáveis — **PASS** no código, Linux não executado
- [x] Modelos necessários estão presentes — **PASS**
- [x] Dataset metadata está consistente — **PASS**, sem nova auditoria de todas as imagens
- [x] Testes passam — **PASS** (5/5)
- [x] Não houve treinamento — **PASS**
- [x] Não houve YOLO26 — **PASS**
- [x] Projeto está pronto para a **preparação** do empacotamento final — **PASS WITH LIMITATIONS**; publicação externa depende dos itens da seção 13
