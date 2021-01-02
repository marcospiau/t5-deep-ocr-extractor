import itertools
from src.models.utils import get_accumulate_grad_batches
from src.models.utils import get_model_size_from_prefix
from src.data.sroie import SROIE_FIELDS_TO_EXTRACT
import os

str_template = """#defaults
include 'defaults.gin'

# Parameters used on T5 paper for finetuning
T5OCRBaseline.optimizer = 'adafactor'
T5OCRBaseline.learning_rate = 1e-3
T5OCRBaseline.generate_max_length = 512

# model size and task
task_train/operative_macro.x = '{task}'
t5_prefix = '{t5_prefix}'

# model
T5OCRBaseline.t5_model_prefix = %t5_prefix
T5OCRBaseline.t5_tokenizer_prefix = %t5_prefix
sroie_t5_baseline/get_default_preprocessing_functions.str_replace_newlines = '{str_replace_newlines}'

# dataset seq lengths
get_datasets_dict_from_task_functions_map.t5_prefix = %t5_prefix
get_datasets_dict_from_task_functions_map.max_source_length = 512
get_datasets_dict_from_task_functions_map.max_target_length = 64

# batch sizes
get_dataloaders_dict_from_datasets_dict.batch_size = {batch_size}
trainer.accumulate_grad_batches = {batch_accum}

# epochs
trainer.min_epochs = {min_epochs}
trainer.max_epochs = {max_epochs}
early_stop.patience = {patience}

# neptune experiment name, same as .gin file
neptune_logger.experiment_name = '{neptune_experiment_name}'
"""

if __name__ == '__main__':
    gin_dump_dir = './t5_default_finetune/'
    os.makedirs(gin_dump_dir, exist_ok=True)
    prefixes = ['t5-small', 't5-base']
    tasks = ['extract_' + x for x in SROIE_FIELDS_TO_EXTRACT]
    tasks.append('all_tasks_concat')
    str_replace_newlines = [' ', '|']
    t5_finetuning_batch_size = 128
    batch_size_prefix_map = {'small': 32, 'base': 8}

    format_keys = ['task', 't5_prefix', 'str_replace_newlines']
    for format_values in itertools.product(tasks, prefixes,
                                           str_replace_newlines):
        format_dict = dict(zip(format_keys, format_values))
        model_size = get_model_size_from_prefix(format_dict['t5_prefix'])
        batch_size = batch_size_prefix_map[model_size]
        batch_accum = get_accumulate_grad_batches(t5_finetuning_batch_size,
                                                  batch_size)

        experiment_name = '{t5_prefix}_{task}_newlines_as_'.format(
                **format_dict)

        if format_dict['str_replace_newlines'] == ' ':
            experiment_name += 'spaces'
        elif format_dict['str_replace_newlines'] == '|':
            experiment_name += 'pipes'
        else:
            raise ValueError('Unexpected value `\\n` replacing')

        format_dict.update({
            'patience': 50,
            'min_epochs': 20,
            'max_epochs': 200,
            'batch_size': batch_size,
            'batch_accum': batch_accum,
            'neptune_experiment_name': experiment_name
        })
        gin_dump_file = os.path.join(gin_dump_dir, experiment_name + '.gin')
        with open(gin_dump_file, 'w') as f:
            f.write(str_template.format(**format_dict))
