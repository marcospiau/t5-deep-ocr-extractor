"""
Saves predictions from a T5OCRBaseline, loading configs from a Neptune
Experiment. The model is loaded using the best experiment checkpoint.
The same preprocessing from finetuning phase is applied, using the operative
gin config uploaded to Neptune.
"""

from absl import app, flags, logging
from src.data.sroie import get_all_keynames_from_dir
from src.data.sroie import get_dataloaders_dict_from_datasets_dict
from src.data.sroie import get_tasks_functions_maps
from src.data.sroie.t5_ocr_baseline import \
        get_datasets_dict_from_task_functions_map
from src.evaluation.sroie_eval_utils import extract_base_keyname
from src.evaluation.sroie_eval_utils import get_multiple_empty_predictions
from src.evaluation.sroie_eval_utils import save_predictions_in_dir
from src.models.t5_ocr_baseline import T5OCRBaseline
from tqdm import tqdm
import gin
import multiprocessing as mp
import neptune
import os
import tempfile


def predict_all_fields(model, dataloader_dict, keynames, device='cuda'):
    preds = get_multiple_empty_predictions(keynames)

    for task, dataloader in dataloader_dict.items():
        if task != 'all_tasks_concat':
            field = task.replace('extract_', '')
            for batch in tqdm(dataloader_dict[task],
                              desc=f"Extracting '{field}' field'"):
                batch['input_ids'] = batch['input_ids'].to(device)
                preds_values = model.predict(batch)
                preds_keynames = batch['keyname']
                for key_prediction, value_prediction in zip(
                        map(extract_base_keyname, preds_keynames),
                        preds_values):
                    preds[key_prediction][field] = value_prediction
    return preds


flags.DEFINE_string('neptune_project', 'marcospiau/final-project-ia376j-1',
                    'Neptune project')
flags.DEFINE_string('neptune_experiment_id', None, 'Neptune experiment id')
flags.DEFINE_string('input_path', None, 'Path with input data')
flags.DEFINE_string('output_path', None,
                    'Path where predictions will be saved')
flags.DEFINE_boolean('gpu', True, 'If a GPU should be used on predictions')
flags.mark_flags_as_required(
    ['input_path', 'output_path', 'neptune_experiment_id'])
FLAGS = flags.FLAGS


def main(_):
    device = 'cuda' if FLAGS.gpu else 'cpu'
    keynames = get_all_keynames_from_dir(FLAGS.input_path)

    logging.info('Retrieving experiment data')
    gin.parse_config_file(
        '/home/marcospiau/final_project_ia376j/src/models/gin/defaults.gin',
        skip_unknown=True)
    project = neptune.init(FLAGS.neptune_project)
    experiment = project.get_experiments(FLAGS.neptune_experiment_id)[0]
    experiment_channels = experiment.get_channels()
    with tempfile.TemporaryDirectory() as tmp_folder:
        experiment.download_artifact('gin_operative_config.gin', tmp_folder)
        gin.parse_config_file(os.path.join(tmp_folder,
                                           'gin_operative_config.gin'),
                              skip_unknown=True)

    logging.info('Loading model checkpoint')
    model = T5OCRBaseline.load_from_checkpoint(
        experiment_channels['best_model_path'].y)
    model.eval()
    model.freeze()
    model.to('cuda')

    logging.info('Preparing datasets and dataloaders')
    with gin.config_scope('sroie_t5_baseline'):
        task_functions_maps = get_tasks_functions_maps()
    datasets = get_datasets_dict_from_task_functions_map(
        keynames=keynames, tasks_functions_maps=task_functions_maps)
    loader_kwargs = {
        'num_workers': mp.cpu_count(),
        'shuffle': False,
        'pin_memory': True
    }
    dataloaders = get_dataloaders_dict_from_datasets_dict(
        datasets_dict=datasets, dataloader_kwargs=loader_kwargs)

    logging.info('Making predictions')
    preds = predict_all_fields(model, dataloaders, keynames, device=device)
    logging.info('Saving predictions')
    os.makedirs(FLAGS.output_path, exist_ok=True)
    save_predictions_in_dir(preds, FLAGS.output_path)


if __name__ == '__main__':
    app.run(main)
