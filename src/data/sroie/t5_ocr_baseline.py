from IPython.display import Image as Image_py
from src.data.google_vision_ocr_parsing import extract_full_text_annotation
from src.data.sroie import load_full_ocr, load_labels
from torch.utils.data import Dataset, ConcatDataset
from transformers import T5Tokenizer
from typing import List, Dict, Union
import gin


class T5BaselineDataset(Dataset):
    """Dataset for text-to-text training.

    Args:
        keynames (List[str]): filenames without extension.
        t5_tokenizer_prefix (str): tokenizer prefix.
        format_labels_fn (callable): function for labels preprocessing
        format_inputs_fn (callable): function for inputs preprocessing
        max_source_length (int): maximum length for toke sequences on encoder.
        max_target_length (int): maximum length for labels token sequences.
    """
    def __init__(
        self,
        keynames: List[str],
        format_labels_fn: callable,
        format_inputs_fn: callable,
        t5_tokenizer_prefix: str,
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


@gin.configurable(allowlist=['str_replace_newlines'])
def get_default_preprocessing_functions(
        field_to_extract: str,
        str_replace_newlines: str = None) -> Dict[str, callable]:
    """Convenience functions for generating  preprocessing functions for labels
        and inputs when using T5 text-to-text training.

    Args:
        field_to_extract (str): field_to_extract
        str_replace_newlines (str): string to replace `\n`
        characters. Usually spaces (` `) or pipes (`|`).

    Returns:
        Dict[str, callable]: dict with two keys:
            - `labels`: function that extracts the ground truth labels for the
                desired field.
            - `inputs`: Replaces `\n` on the ocr output text and prepends the
                question: `What is the <field_to_extract?`
    """
    def format_labels_fn(x):
        return x[field_to_extract]

    def format_inputs_fn(x):
        return f'What is the {field_to_extract}? ' + x.replace(
            '\n', str_replace_newlines)

    return {'inputs': format_inputs_fn, 'labels': format_labels_fn}

@gin.configurable(denylist=['keynames'])
def get_datasets_dict_from_task_functions_map(
        keynames: List[str],
        tasks_functions_maps: List[Dict[str, callable]],
        t5_prefix: str,
        max_source_length: int,
        max_target_length: int) -> Dict[str, Dataset]:
    """Return dict with datasets for each individual task and one dataset with
        all tasks concatenated.

    Args:
        keynames (List[str]): keynames (paths without extension) for receipts
            on SROIE dataset.
        t5_prefix (str, optional): prefix identifying the chosen t5 model.
            Will be used to instantiate the T5Tokenizer.
        max_source_length (int): Length of input token
            sequences. Size is enforced with truncation or padding. Defaults to
        max_target_length (int): length of labels token sequences.

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
