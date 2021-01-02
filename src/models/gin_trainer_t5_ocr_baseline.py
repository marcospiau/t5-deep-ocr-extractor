from absl import app, flags
from io import StringIO
from src.data.sroie import get_all_keynames_from_dir
from src.data.sroie import get_dataloaders_dict_from_datasets_dict
from src.data.sroie import get_tasks_functions_maps
from src.data.sroie.t5_ocr_baseline import \
        get_datasets_dict_from_task_functions_map
from src.models.gin_configurables import NeptuneLogger
from src.models.gin_configurables import Trainer
from src.models.gin_configurables import config_early_stopping_callback
from src.models.gin_configurables import config_model_checkpoint
from src.models.gin_configurables import operative_macro
from src.models.t5_ocr_baseline import T5OCRBaseline
import gin
import multiprocessing as mp
import os
import pkg_resources
import pytorch_lightning as pl
# https://github.com/google/gin-config/blob/3beb4056d232acd2381f74518dc399db4817bf92/docs/index.md
flags.DEFINE_multi_string('gin_file', None,
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Newline separated list of Gin parameter bindings.')
flags.DEFINE_boolean(
    'debug', False,
    'Run in debug mode, checkpoints wil not be saved and Neptune logger will '
    'not be used')
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
    with gin.config_scope('sroie_t5_baseline'):
        task_functions_maps = get_tasks_functions_maps()

    # Datasets
    with gin.config_scope('train_sroie'):
        train_keynames = get_all_keynames_from_dir()

    with gin.config_scope('validation_sroie'):
        val_keynames = get_all_keynames_from_dir()

    train_datasets = get_datasets_dict_from_task_functions_map(
        keynames=train_keynames, tasks_functions_maps=task_functions_maps)
    val_datasets = get_datasets_dict_from_task_functions_map(
        keynames=val_keynames, tasks_functions_maps=task_functions_maps)

    with gin.config_scope('task_train'):
        task_train = operative_macro()

    # Initializing model
    model = T5OCRBaseline()

    # Trainer
    if FLAGS.debug:
        logger = False
        trainer_callbacks = []
    else:
        logger = NeptuneLogger(
            close_after_fit=False,
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            # project_name is set via gin file
            # params=None,
            tags=[model.t5_model_prefix, task_train, 't5_ocr_baseline'])
        with gin.config_scope('sroie_t5_baseline'):
            checkpoint_callback = config_model_checkpoint(
                monitor="val_f1",
                dirpath=("/home/marcospiau/final_project_ia376j/checkpoints/"
                         f"{logger.project_name.replace('/', '_')}/"
                         "t5_ocr_baseline/"),
                prefix=(
                    f"experiment_id={logger.experiment.id}-task={task_train}-"
                    "t5_model_prefix="
                    f"{model.t5_model_prefix.replace('-', '_')}"),
                filename=("{step}-{epoch}-{val_precision:.6f}-{val_recall:.6f}"
                          "-{val_f1:.6f}-{val_exact_match:.6f}"),
                mode="max",
                save_top_k=1,
                verbose=True)
        early_stop_callback = config_early_stopping_callback()
        trainer_callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        checkpoint_callback=not (FLAGS.debug),
        log_gpu_memory=True,
        # profiler=FLAGS.debug,
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

    # Logging important artifacts and params
    if logger:
        to_upload = {
            'gin_operative_config.gin': gin.operative_config_str(),
            'gin_complete_config.gin': gin.config_str(),
            'abseil_flags.txt': FLAGS.flags_into_string()
        }
        for destination, content in to_upload.items():
            buffer = StringIO(initial_value=content)
            buffer.seek(0)
            logger.log_artifact(buffer, destination=destination)
        params_to_log = dict()
        params_to_log['str_replace_newlines'] = gin.query_parameter(
            'sroie_t5_baseline/get_default_preprocessing_functions.'
            'str_replace_newlines')
        params_to_log['task_train'] = task_train
        params_to_log['patience'] = early_stop_callback.patience
        params_to_log['max_epochs'] = trainer.max_epochs
        params_to_log['min_epochs'] = trainer.min_epochs
        params_to_log[
            'accumulate_grad_batches'] = trainer.accumulate_grad_batches

        for k, v in params_to_log.items():
            logger.experiment.set_property(k, v)

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
