from typing import List, Dict, Union, Callable
import glob
import json
import os
from torch.utils.data import Dataset, DataLoader
import gin

SROIE_FIELDS_TO_EXTRACT = ['address', 'date', 'total', 'company']


def load_labels(key_path: str) -> dict:
    """Loads labels, filling missing values with empty string.

    """
    required_keys = ['company', 'date', 'address', 'total']
    defaults = dict((k, '') for k in required_keys)

    text_data = json.load(open(f'{key_path}.txt', 'r'))
    # text_data = defaultdict(lambda: '', text_data)
    text_data = {**defaults, **text_data}
    return text_data


def load_full_ocr(key_path: str) -> dict:
    """Loads full output from Google Vision OCR.

    """
    text_data = json.load(open(f'{key_path}.json', 'r'))
    return text_data


@gin.configurable
def get_all_keynames_from_dir(base_dir: str) -> List[str]:
    """Gets all keynames (filenames without extensions) from dir.

    """
    # all_files = os.listdir(base_dir)
    all_files = glob.glob(base_dir + '/*')
    keynames = sorted(set([os.path.splitext(x)[0] for x in all_files]))
    return keynames


@gin.configurable
def get_dataloaders_dict_from_datasets_dict(
        datasets_dict: Dict[str, Dataset], batch_size: int,
        dataloader_kwargs: dict) -> Dict[str, DataLoader]:
    """Generates dataloaders dict from datasets dict.

    Args:
        datasets_dict (Dict[str, Dataset]): dict of datasets.
        batch_size (int): batch_size.
        dataloader_kwargs (dict): kwargs for dataloader initialization.

    Returns:
        Dict[str, DataLoader]: dict with dataloaders.
    """
    dataloaders = {
        k: DataLoader(v, batch_size=batch_size, **dataloader_kwargs)
        for k, v in datasets_dict.items()
    }
    return dataloaders


@gin.configurable
def get_tasks_functions_maps(
    base_function_generator_fn: Callable,
    fields_to_extract: Union[str, List[str]] = SROIE_FIELDS_TO_EXTRACT
) -> Dict[str, Dict[str, Callable]]:
    """Get maps for preprocessing functions for each task, assuming that the
        only difference is the field to be extracted.

    Args:
        base_function_generator_fn(Callable): function that generates functions
            for preprocessing inputs and labels. The only argument is the field
            to be extracted.
        fields_to_extract (List[str]): list of fields to be extracted.
            Defaults to the 4 fields of SROIE task 3.

    Returns:
        Dict[str, Dict[str, Callable]]: dict mapping tasks to preprocesing
            functions.

    """
    if isinstance(fields_to_extract, list):
        pass
    elif isinstance(fields_to_extract, str):
        fields_to_extract = [fields_to_extract]
    else:
        raise ValueError("fields_to_extract type must be `str` or `list`")

    task_function_maps = {
        'extract_' + k: base_function_generator_fn(k)
        for k in fields_to_extract
    }
    return task_function_maps
