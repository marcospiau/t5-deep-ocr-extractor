from IPython.display import Image as Image_py
from src.data.google_vision_ocr_parsing import extract_full_text_annotation
from src.data.sroie.sroie_common import load_full_ocr, load_labels
from torch.utils.data import Dataset, ConcatDataset
from transformers import T5Tokenizer
from typing import List, Dict, Union
from src.data.sroie import SROIE_FIELDS_TO_EXTRACT


class T5BaselineDataset(Dataset):
    """Dataset for text-to-text training.

    Args:
        keynames (List[str]): filenames without extension.
        max_source_length (int): maximum length for toke sequences on encoder.
        t5_tokenizer_prefix (str): tokenizer prefix. Defaults to `t5-small`.
            Defaults to None (maximum model length, 512 on T5).
        max_target_length (int): maximum length for toke sequences on decoder.
            Defaults to None (maximum model length, 512 on T5).
        format_labels_fn (callable): function that prepare labels for training.
        format_inputs_fn (callable): function that prepares inputs for
            training.
        ocr_outputs (Dict[str, str]): dict mapping key name to raw OCR text
            output.

    """
    def __init__(
        self,
        keynames: List[str],
        format_labels_fn: callable,
        format_inputs_fn: callable,
        t5_tokenizer_prefix: str = 't5-small',
        max_source_length: int = None,
        max_target_length: int = None,
    ):
        self.keynames = keynames
        self.t5_tokenizer_prefix = t5_tokenizer_prefix
        self.tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_prefix)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.format_labels_fn = format_labels_fn
        self.format_inputs_fn = format_inputs_fn

    def __len__(self):
        return len(self.keynames)

    def __getitem__(self, idx):
        raw_input_data = extract_full_text_annotation(
            load_full_ocr(self.keynames[idx]))
        formatted_input = self.format_inputs_fn(raw_input_data)

        raw_labels_data = load_labels(self.keynames[idx])
        formatted_labels = self.format_labels_fn(raw_labels_data)

        tokenized_inputs = self.tokenizer.encode_plus(
            formatted_input,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_source_length)

        tokenized_outputs = self.tokenizer.encode_plus(
            formatted_labels,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_target_length)

        # Labels with -100 are ignored on CrossEntropyLoss
        # https://github.com/huggingface/transformers/blob/b290195ac78275e048396eabcce396c4cee0975a/src/transformers/models/t5/modeling_t5.py#L1273
        labels = tokenized_outputs.input_ids.squeeze()
        labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)

        rets = {
            'raw_input_data': raw_input_data,
            'formatted_input': formatted_input,
            'raw_labels_data': raw_labels_data,
            'formatted_labels': formatted_labels,
            'keyname': self.keynames[idx],

            # tokenizer outputs
            'input_ids': tokenized_inputs.input_ids.squeeze(),
            'attention_mask': tokenized_inputs.attention_mask.squeeze(),
            'labels': labels
        }

        return rets

    def inspect_example(self, idx):
        """Inspects examples for debugging purposes.

        """
        image = Image_py(filename=f'{self.keynames[idx]}.jpg',
                         width=1200,
                         height=1200)
        display(image)

        example = self.__getitem__(idx)
        return example


def get_default_preprocessing_functions(field_to_extract):
    """Convenience function for generating preprocessing functions for inputs
        and labels when using T5 text-to-text.

    For inputs, returns `What is the <x>? <formatted_text>`, where <x> is the
    field to be extracted and <formatted_text> are the ocr outputs with `\n`
    replace by blank spaces.

    For the labels, just return the ground truth labels.
    """
    def format_labels_fn(x):
        return x[field_to_extract]

    def format_inputs_fn(x):
        return f'What is the {field_to_extract}? ' + x.replace('\n', ' ')

    return {'inputs': format_inputs_fn, 'labels': format_labels_fn}


DEFAULT_TASK_FUNCTION_MAPS = {
    k: get_default_preprocessing_functions(k)
    for k in SROIE_FIELDS_TO_EXTRACT
}


def get_datasets_dict_from_task_functions_map(
        keynames: List[str],
        tasks_functions_maps: [List[Dict[str, callable]]
                               ] = DEFAULT_TASK_FUNCTION_MAPS,
        t5_prefix: str = 't5-small',
        max_source_length: Union[int, None] = 512,
        max_target_length: Union[int, None] = 64) -> Dict[str, Dataset]:
    """Return dict with datasets for each individual task and one dataset with
        all tasks concatenated.

    Args:
        keynames (List[str]): keynames (paths without extension) for receipts
            on SROIE dataset.
        t5_prefix (str, optional): prefix identifying the chosen t5 model.
            Will be used to instantiate the tokenzier. Defaults to 't5-small'.
        max_source_length (Union[int, None], optional): Length of input token
            sequences. Size is enforced with truncation or padding. Defaults to
                64; None means size for model (512 for default
                HuggingFace's T5 configs.
        max_target_length (Union[int, None], optional): length of labels token
            sequences. Defaults to 64. The observations in `max_source_length`
            parameter are valid here too.

    Returns:
        Dict[str, Dataset]: dataset for each task.
    """
    # Dataset for each task
    datasets = {
        k: T5BaselineDataset(keynames=keynames,
                             format_labels_fn=v['labels'],
                             format_inputs_fn=v['inputs'],
                             t5_tokenizer_prefix=t5_prefix,
                             max_source_length=max_source_length,
                             max_target_length=max_target_length)
        for k, v in tasks_functions_maps.items()
    }
    # dataset with all tasks
    datasets['all_tasks_concat'] = ConcatDataset(datasets.values())
    return datasets
