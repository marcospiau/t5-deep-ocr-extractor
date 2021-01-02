import os
import gin
import sys
import pkg_resources
from src.models.t5_ocr_baseline import T5OCRBaseline
from absl import app, flags, logging
import multiprocessing as mp
import pytorch_lightning as pl
from src.data.sroie.t5_ocr_baseline import (
    T5BaselineDataset, get_datasets_dict_from_task_functions_map,
    get_default_preprocessing_functions)
from src.data.sroie import (SROIE_FIELDS_TO_EXTRACT, get_tasks_functions_maps,
                            get_all_keynames_from_dir,
                            get_dataloaders_dict_from_datasets_dict)
from src.models.utils import get_model_size_from_prefix
from functools import partial

flags.DEFINE_multi_string('gin_file', None,
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_enum('optimizer', 'adafactor', ['adafactor', 'adam'], 'Optimizer')

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('generate_max_length', 512,
                     'Maximum token length for T5 generation')
flags.DEFINE_integer(
    'batch_size', None,
    'Batch size. If unset, 32 and 8 will be used for `small` and `base`'
    'models, respectively')
flags.DEFINE_integer(
    'accumulate_grad_batches', None,
    'Number of batches trained before each optimizer update step.'
    'If unset, will be calculated to match t5 default batch size '
    'For finetuning (128)')
flags.DEFINE_integer('max_epochs', None, 'Max training epochs')
flags.DEFINE_integer('patience', 9999999, 'Patience epochs for training')
flags.DEFINE_string('str_replace_newlines', ' ',
                    'String that replaces `\n` on inputs from OCR text')
flags.DEFINE_string('t5_model_prefix', None,
                    'Any valid t5 model from HuggingFace')
flags.DEFINE_string(
    't5_tokenizer_prefix', None,
    'Any valid T5Tokenizer prefix from HuggingFace. If unset, '
    ' `t5_model_prefix` will be used')
flags.DEFINE_integer('max_source_length', 512,
                     'Maxumum sequence length on input')
flags.DEFINE_integer('max_target_length', 64,
                     'Maximum sequence length for labels')
flags.DEFINE_string(
    'train_basedir',
    '/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/train/',
    'Directory with train files')
flags.DEFINE_string(
    'val_basedir',
    '/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/test/',
    'Directory with validation files')
sroie_valid_tasks = ['extract_' + x for x in SROIE_FIELDS_TO_EXTRACT]
sroie_valid_tasks.append('all_tasks_concat')
flags.DEFINE_enum('task_train', None, sroie_valid_tasks,
                  'Task used on training')
flags.DEFINE_string('neptune_project_name', 'marcospiau/test-logs-until-ok',
                    'Neptune project name')
flags.DEFINE_string('checkpoints_basedir',
                    '/home/marcospiau/final_project_ia376j/checkpoints',
                    'Base directory for checkpoint saving')
flags.DEFINE_boolean(
    'debug', False,
    'Run in debug mode, checkpoints wil not be saved and Neptune logger will not be used'
)
flags.DEFINE_boolean(
    'upload_best_checkpoint', False,
    "If best model checkpoint should be uploaded to Neptune experiment")
FLAGS = flags.FLAGS


def main(_):
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/c0ea75dbe9e35a629ae2e3c964ef32adc0e997f3/t5/models/mesh_transformer_main.py#L149
    # Add search path for gin files stored in package.
    gin.add_config_file_search_path(
        pkg_resources.resource_filename(__name__, "gin"))
    gin.parse_config_files_and_bindings(FLAGS.gin_file,
                                        FLAGS.gin_param,
                                        finalize_config=True)
    pl.seed_everything(1234)

    task_functions_maps = get_tasks_functions_maps(
        partial(get_default_preprocessing_functions,
                str_replace_newlines=FLAGS.str_replace_newline))

    train_keynames = get_all_keynames_from_dir(FLAGS.train_basedir)
    val_keynames = get_all_keynames_from_dir(FLAGS.val_basedir)

    train_datasets = get_datasets_dict_from_task_functions_map(
        keynames=train_keynames,
        tasks_functions_maps=task_functions_maps,
        t5_prefix=FLAGS.t5_tokenizer_prefix,
        max_source_length=FLAGS.max_source_length)

    val_datasets = get_datasets_dict_from_task_functions_map(
        keynames=val_keynames,
        tasks_functions_maps=task_functions_maps,
        t5_prefix=FLAGS.t5_tokenizer_prefix,
        max_source_length=FLAGS.max_source_length)

    # Initializing model
    model = T5OCRBaseline(t5_model_prefix=FLAGS.t5_model_prefix,
                          t5_tokenizer_prefix=FLAGS.t5_tokenizer_prefix,
                          optimizer=FLAGS.optimizer,
                          learning_rate=FLAGS.learning_rate,
                          generate_max_length=FLAGS.generate_max_length)

    # Trainer
    if FLAGS.debug:
        logger = False
        trainer_callbacks = []
    else:
        logger = pl.loggers.NeptuneLogger(
            close_after_fit=False,
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            # project_name is set via gin file
            # params=None,
            tags=[FLAGS.t5_model_prefix, FLAGS.task_train, 't5_ocr_baseline'])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            prefix=
            f"experiment_id={logger.experiment.id}-task={task_train}-t5_model_prefix={model.t5_model_prefix.replace('-', '_')}",
            dirpath=os.path.join(FLAGS.checkpoint_basedir, "t5_ocr_baseline"),
            filename=
            "{step}-{epoch}-{val_precision:.6f}-{val_recall:.6f}-{val_f1:.6f}-{val_exact_match:.6f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            verbose=True)
        # Patience comes from gin
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f1', mode='max', patience=FLAGS.patience)
        trainer_callbacks = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(checkpoint_callback=not (FLAGS.debug),
                         log_gpu_memory=True,
                         profiler=FLAGS.debug,
                         logger=logger,
                         callbacks=trainer_callbacks,
                         progress_bar_refresh_rate=1,
                         log_every_n_steps=1)

    # Dataloaders
    train_loader_kwargs = {
        'num_workers': mp.cpu_count(),
        'shuffle': True if (trainer.overfit_batches == 0) else False,
        'pin_memory': True
    }

    if trainer.overfit_batches != 0:
        with gin.unlock_config():
            gin.bind_parameter(
                'get_dataloaders_dict_from_datasets_dict.batch_size', 1)

    eval_loader_kwargs = {**train_loader_kwargs, **{'shuffle': False}}

    train_dataloaders = get_dataloaders_dict_from_datasets_dict(
        datasets_dict=train_datasets, dataloader_kwargs=train_loader_kwargs)
    val_dataloaders = get_dataloaders_dict_from_datasets_dict(
        datasets_dict=val_datasets, dataloader_kwargs=eval_loader_kwargs)

    print(f'gin total config: {gin.config_str()}')
    print(f'gin operative config: {gin.operative_config_str()}')
    print(f"flags used: {FLAGS.flags_into_string()}")

    trainer.fit(model,
                train_dataloader=train_dataloaders[task_train],
                val_dataloaders=val_dataloaders[task_train])

    # Logging best metrics and saving best checkpoint on Neptune experiment
    if logger:
        trainer.logger.experiment.log_text(
            log_name='best_model_path',
            x=trainer.checkpoint_callback.best_model_path)
        trainer.logger.experiment.log_metric(
            'best_model_val_f1',
            trainer.checkpoint_callback.best_model_score.item())
        if FLAGS.upload_best_checkpoint:
            trainer.logger.experiment.log_artifact(
                trainer.checkpoint_callback.best_model_path)

        trainer.logger.experiment.stop()


if __name__ == '__main__':
    app.run(main)
