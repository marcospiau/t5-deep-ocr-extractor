from IPython.display import Image as Image_py
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from typing import List, Dict
import glob
import json
import os
from src.data.google_vision_ocr_parsing import extract_full_text_annotation


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


def get_all_keynames_from_dir(base_dir: str) -> List[str]:
    """Gets all keynames (filenames without extensions) from dir.

    """
    # all_files = os.listdir(base_dir)
    all_files = glob.glob(base_dir + '/*')
    keynames = sorted(set([os.path.splitext(x)[0] for x in all_files]))
    return keynames


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
