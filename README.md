# Projeto final IA376J - Desafio SROIE Task 3 - Key Information Extraction from Scanned Receipts

# Introdução

Este repositório contém código para solucionar a `Task 3 - Key Information Extraction from Scanned Receipts` do `ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction`. A descrição completa do desafio e tarefas completas pode ser encontrada em [Tasks - ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction](https://rrc.cvc.uab.es/?ch=13&com=tasks). Esta tarefa busca extrair campos relevantes [“address”, “company”, “total”, “date”] de receipts escaneados. Nossa solução é baseada em extrair informações textuais das imagens utilizando o Google Vision OCR, e então alimentar um modelo T5 com esses dados para extrair os campos relevantes.

Todos os experimentos estão registrados no Neptune Logger, e podem ser consultados nesse link [link](https://ui.neptune.ai/marcospiau/final-project-ia376j-1/experiments?viewId=7b690bc2-f1ea-499d-81f2-30a5c0208c04). ALgumas bibliotecas importantes utilizadas no projeto são:
* pytorch-lightning: redução de boilerplate pytorch e configuração do loop de treino e eval
* Hugging Face 🤗 transformers: modelos T5
* gin-config: configuração dos experimentos
* Neptune: experiment tracking

# Como esse repositório está estruturado

 ```
 .
|-- LICENSE
|-- README.md
|-- notebooks <<<<< Jupyter Notebooks
|   |-- draft <<<<< notebooks de rascunho
|   |-- select_best_sroie_checkpoints_t5_ocr_baseline_initial_finetune.ipynb <<<<< notebook com seleção inicial dos melhores modelos 
|   `-- sroie_t5_ocr_baseline_prepare_competition_submission.ipynb <<<<< notebook com criação dos arquivos pra submissão na competição
|-- setup.py
`-- src
    |-- __init__.py
    |-- data <<< códigos que tratam os dados
    |   |-- __init__.py
    |   |-- google_vision_ocr_extraction.py <<<< código para extração de OCR usando o Google Vision
    |   |-- google_vision_ocr_parsing.py <<< parsing dos OCR gerados pelo Google Vision
    |   `-- sroie
    |       |-- __init__.py
    |       `-- t5_ocr_baseline.py <<<< Pytorch Dataset usado nos modelos
    |
    |-- evaluation <<<< códigos e scripts para avaliação dos modelos
    |   |-- __init__.py
    |   |-- save_experiment_predictions_t5_ocr_baseline.py <<<<< 
    |   |-- save_preds_t5_final_finetuning.sh
    |   |-- save_preds_t5_initial_finetuning.sh
    |   `-- sroie_eval_utils.py
    |-- metrics.py <<<< métricas utilizadas
    |-- models <<< códigos dos modelos
    |   |-- __init__.py
    |   |-- gin <<<< arquivos gin com configurações de todos experimentos realizados
    |   |   |-- README.md
    |   |   |-- best_t5_models_defaults.gin <<< gin default config treinamento dos modelos finais (treinados sobre todos dados rotulados com as melhores combinações de hiperparâmetros)
    |   |   |-- defaults.gin <<<< gin default config para experimentos iniciais de finetune
    |   |   |-- generate_t5_default_finetune_gin_configs.py <<<<< script que gera os arquivos gin config dos experimentos iniciais de finetune
    |   |   |-- t5_best_models_finetune <<<< gin configs dos modelos finais (treinados sobre todos dados rotulados com as melhores combinações de hiperparâmetros)
    |   |   |-- t5_default_finetune <<<<< arquivos gin config dos experimentos inciais de finetune
    |   |-- gin_configurables.py <<<< extensão de classes para configuráveis gin
    |   |-- gin_trainer_t5_ocr_baseline.py <<<< script principal de treinamento do modelo 
    |   |-- past_scripts <<<< scripts antigos
    |   |-- t5_ocr_baseline.py <<<< código módulo Pytorch Lightning do modelo
    |   `-- utils.py 
    `-- utils.py
 ```
