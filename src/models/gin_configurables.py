from pathlib import Path
from typing import Optional, Union
import gin
import pytorch_lightning as pl

Trainer = gin.config.external_configurable(pl.Trainer, name='trainer')

NeptuneLogger = gin.config.external_configurable(pl.loggers.NeptuneLogger,
                                                 name='neptune_logger')


# Eearly Stopping callback must be pickeable
@gin.configurable('early_stop')
def config_early_stopping_callback(patience):
    return pl.callbacks.EarlyStopping(monitor='val_f1',
                                      mode='max',
                                      patience=patience)


# ModelCheckpoint callback must be pickeable
# Exatcly same signature from pl.callbacks.model_checkpoint.ModelCheckpoint
@gin.configurable('checkpoint')
def config_model_checkpoint(
    filepath: Optional[str] = None,
    monitor: Optional[str] = None,
    verbose: bool = False,
    save_last: Optional[bool] = None,
    save_top_k: Optional[int] = None,
    save_weights_only: bool = False,
    mode: str = "auto",
    period: int = 1,
    prefix: str = "",
    dirpath: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
) -> pl.callbacks.model_checkpoint.ModelCheckpoint:
    return pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=verbose,
        save_last=save_last,
        save_top_k=save_top_k,
        save_weights_only=save_weights_only,
        mode=mode,
        period=period,
        prefix=prefix,
        dirpath=dirpath,
        filename=filename)


@gin.configurable('operative_macro')
def operative_macro(x):
    """Hack to put macros on gin operative config.

    """
    return x
