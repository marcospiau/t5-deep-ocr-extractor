import itertools

str_template = """
# necessary
include 'defaults.gin'

# Parameters used on T5 paper for finetuning
T5OCRBaseline.optimizer = 'adafactor'
T5OCRBaseline.learning_rate = 1e-3
T5OCRBaseline.generate_max_length = 512
# T5_FINETUNING_BATCH_SIZE = 128


# model size and task
task_train/operative_macro.x = 'extract_company'
t5_prefix = {t5_prefix}

# model
T5OCRBaseline.t5_model_prefix = %t5_prefix
T5OCRBaseline.t5_tokenizer_prefix = %t5_prefix
sroie_t5_baseline/get_default_preprocessing_functions.str_replace_newlines = {str_replace_newlines}

# dataset seq lengths
get_datasets_dict_from_task_functions_map.t5_prefix = %t5_prefix
get_datasets_dict_from_task_functions_map.max_source_length = 512
get_datasets_dict_from_task_functions_map.max_target_length = 64

# batch sizes
get_dataloaders_dict_from_datasets_dict.batch_size = {batch_size}

trainer.accumulate_grad_batches = {batch_accum}
trainer.min_epochs = {min_epochs}
trainer.max_epochs = {max_epochs}
early_stop.patience = {patience}

neptune_logger.experiment_name = {neptune_experiment_name}
"""

prefixes = ['t5-small', 't5-base']
str_replace_newlines = [' ', '|']
batch_size = [32]
batch_accum = [4]
min_epochs = [20]
max_epochs = [100]
patience = [50]

for format_values in itertools.product(
        prefixes,
        str_replace_newlines,
):
    format_keys = [
        'prefix', 'str_replace_newlines', 'min_epochs', 'max_epochs',
        'patience'
    ]
    format_dict = dict(zip(format_keys, format_values))
    print(format_dict)

print(str_template)
