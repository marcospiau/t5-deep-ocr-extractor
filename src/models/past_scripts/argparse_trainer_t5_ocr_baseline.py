from src.models.t5_ocr_baseline import (
        T5OCRBaseline,
        DEFAULT_TASK_FUNCTION_MAPS
        )
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pytorch_lightning  as pl
import multiprocessing as mp
from pytorch_lightning.loggers import NeptuneLogger
from src.utils import parse_yaml_file

# Default configs






def parse_args():
    pl.seed_everything(1234)

    # Necessary for model haprams initialization
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
            '--t5_model_prefix',
            type=str,
            default='t5-small',
            help='Prefix for T5 model initialization'
            )
    parser.add_argument(
            '--t5_tokenizer_prefix',
            type=str,
            default='t5-small',
            help="Prefix for T5Tokenizer initialization"
            )
    parser.add_argument(
            '--generate_max_length',
            type=int, default=512,
            help='Maximum length for token sequence generation'
            )
    parser.add_argument(
            '--optimizer',
            type=str,
            default='adafactor',
            choices=['adafactor', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # Data
    parser.add_argument('--train_dir', type=str, default=

    # Trainer specifics
    parser.add_argument(
            '--patience',
            type=int,
            default=None,
            help="Patience epochs (monitor on `val_f1`)"
            )

    parser.add_argument(
            '--task',
            type=str,
            default=None,
            required=True,
            help="Task used for model training and validation",
            choices=[
                'extract_company',
                'extract_date',
                'extrract_total',
                'extract_address',
                'all_tasks_concat'
                ]
            )
    parser.add_argument(
    # parser = T5OCRBaseline.add_model_specific_args(parser)

    args = parser.parse_args()

    # Datasets and dataloaders
    default_batch_sizes = {'t5-small': 32, 't5-base':8}
    train_loader_kwargs = {
        'num_workers': args.num_workers or mp.cpu_count()
        'shuffle': True,
        'pin_memory': args.pin_memory,
        'batch_size': args.batch_size or default_batch_sizes[args.t5_model_prefix]
    }
    eval_loader_kwargs = {**train_loader_kwargs, **{'shuffle': False}}

    train_datasets = get_datasets_dict_from_task_functions_map(
        train_keynames,
        DEFAULT_TASK_FUNCTION_MAPS,
        T5_PREFIX,
        MAX_SOURCE_LENGTH,
        MAX_TARGET_LENGTH
        )
    train_dataloaders = get_dataloaders_dict_from_datasets_dict(
        train_datasets,
        train_loader_kwargs
        )

    val_datasets = get_datasets_dict_from_task_functions_map(
        val_keynames,
        DEFAULT_TASK_FUNCTION_MAPS,
        T5_PREFIX,
        MAX_SOURCE_LENGTH,
        MAX_TARGET_LENGTH
        )
    val_dataloaders = get_dataloaders_dict_from_datasets_dict(
        val_datasets,
        eval_loader_kwargs
    )


if __name__ == '__main__':
    cli_main()
