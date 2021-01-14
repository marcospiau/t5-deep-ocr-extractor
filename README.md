# Projeto final IA376J - Desafio SROIE Task 3

Todos os experimentos estão registrados no Neptune Logger, através desse [link](https://ui.neptune.ai/marcospiau/final-project-ia376j-1/experiments?viewId=7b690bc2-f1ea-499d-81f2-30a5c0208c04). Estão registrados os arquivos gin config (ver o repositório [gin-config](https://github.com/google/gin-config) para maiores informações.

Abaixo um resumo de como o projeto está estruturado. Algumas seções possuem um README mais detalhado:

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

