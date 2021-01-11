import os
from tqdm import tqdm
import json


def extract_base_keyname(keyname):
    return os.path.basename(keyname)


def get_single_empty_prediction(keyname):
    return dict.fromkeys(['company', 'date', 'address', 'total'], '')


def get_multiple_empty_predictions(keynames):
    base_keynames = list(map(extract_base_keyname, keynames))
    return {
        keyname: get_single_empty_prediction(keyname)
        for keyname in base_keynames
    }


def save_predictions_in_dir(predictions, basepath):
    for pred_keyname, pred_values in tqdm(
            predictions.items(), desc=f'Saving predictions in {basepath}'):
        preds_file = os.path.join(basepath, pred_keyname) + '.txt'
        with open(preds_file, 'w') as f:
            json.dump(pred_values, f, ensure_ascii=False, indent=4)
