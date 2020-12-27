from collections import OrderedDict
from fairseq.optim.adafactor import Adafactor
from src.metrics import compute_exact, compute_precision_recall_f1
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import numpy as np
import pytorch_lightning as pl
import random
import torch
from io import StringIO


class T5OCRBaseline(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            self.hparams.t5_prefix)

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.t5_tokenizer_prefix)

    def decode_token_ids(self, token_ids):
        """Decodes token_ids into text.
        """
        decoded_text = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return decoded_text

    def configure_optimizers(self):
        if self.hparams.optmizer == 'adam':
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.hparams.learning_rate,
                eps=1e-08)
        elif self.hparams.optmizer == 'adafactor':
            # https://discuss.huggingface.co/t/t5-finetuning-tips/684
            optimizer = Adafactor(
                [p for p in self.parameters() if p.requires_grad],
                scale_parameter=False,
                relative_step=False,
                lr=self.hparams.learning_rate)

        return optimizer

    @torch.no_grad()
    def _base_eval_step(self, batch):
        """Base function for eval/test steps.

        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        generated_tokens = self.t5.generate(
            input_ids=input_ids,
            # attention_mask = attention_mask,
            **self.hparams.t5_generate_kwargs)
        generated_text = self.decode_token_ids(generated_tokens)

        # Compute metrics
        precision_recall_f1 = list(
            map(lambda pair: compute_precision_recall_f1(*pair),
                zip(batch['formatted_labels'], generated_text)))
        precision = [x['precision'] for x in precision_recall_f1]
        recall = [x['recall'] for x in precision_recall_f1]
        f1 = [x['f1'] for x in precision_recall_f1]
        exact_match = list(
            map(lambda pair: compute_exact(*pair),
                zip(batch['formatted_labels'], generated_text)))

        rets = OrderedDict([
            ('keyname', batch['keyname']),
            ('formatted_input', batch['formatted_input']),
            ('formatted_labels', batch['formatted_labels']),
            ('generated_text', generated_text),
            ('precision', precision),
            ('recall', recall),
            ('f1', f1),
            ('exact_match', exact_match),
        ])
        return rets

    # Israel faz algo parecido
    def _concat_dict_by_key(self, x, key):
        return sum([x[key] for x in x], [])

    def _base_eval_epoch_end(self, outputs, prefix):
        # Reducing metrics over epoch by mean
        # Não usei o log do lightning porque não sei o que ele faz no final
        precision_epoch = np.mean(
            self._concat_dict_by_key(outputs, 'precision'))
        recall_epoch = np.mean(self._concat_dict_by_key(outputs, 'recall'))
        f1_epoch = np.mean(self._concat_dict_by_key(outputs, 'f1'))
        exact_match_epoch = np.mean(
            self._concat_dict_by_key(outputs, 'exact_match'))

        # Prints a random example to Neptune Logger
        random_outs = random.choice(outputs)
        idx_sample = random.randint(0,
                                    next(map(len, random_outs.values())) - 1)
        example = {k: v[idx_sample] for k, v in random_outs.items()}

        buffer = StringIO()
        print(100 * '-', file=buffer)
        print('Sample predictions epoch',
              f" {self.current_epoch} '{prefix}':",
              file=buffer)
        for k, v in example.items():
            if k in ['precision', 'recall', 'f1', 'exact_match', 'keyname']:
                print(f'{k}: {v}', file=buffer)
            else:
                print(f'{k}:\n    {v}', file=buffer)

        print(buffer.getvalue())
        buffer.seek(0)
        # if self.logger is not None:
        #     self.logger.experiment.log_artifact(
        #         buffer,
        #         f'example_{prefix}_predictions_epoch={self.current_epoch}.txt'
        #         )

        log_dict = {
            f"{prefix}_precision": precision_epoch,
            f"{prefix}_recall": recall_epoch,
            f"{prefix}_f1": f1_epoch,
            f"{prefix}_exact_match": exact_match_epoch
        }
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True  # TODO testing
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self.t5(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels,
                       return_dict=True).loss
        # self.log('loss',
        #          loss,
        #          prog_bar=False,
        #          on_epoch=False,
        #          on_step=True,
        #          logger=False) # Nao deixar esse logger=True porque da erro

        if self.logger is not None:
            self.logger.experiment.log_metric('train_loss_step', loss)

        # Acho que esse aqui vai dar problema, o global_step so aumentar com o grad_batch_accum
        # self.logger.experiment.log_metric('train_loss_manual_2', self.global_step, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self._base_eval_step(batch)
        return output

    def validation_epoch_end(self, outputs):
        self._base_eval_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        output = self._base_eval_step(batch)
        return output

    def test_epoch_end(self, outputs):
        self._base_eval_epoch_end(outputs, 'test')
        # return output
