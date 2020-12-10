import datetime, pytz
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import loggers

from PrecipitationForecastRadar.dataloader.regression_lightning_datamodule import PrecipRegressionDataModule
from PrecipitationForecastRadar.models.regression_lightning_unet_precip import UNet


def train_regression():
    batch_size = 8

    path_image_list = 'dataset/jma_okinawa_2015/file_list.csv'
    num_input_images = 12
    num_output_images = 1
    datetime_train_start = datetime.datetime(year=2015, month=1, day=1, tzinfo=pytz.utc)
    datetime_train_end = datetime.datetime(year=2015, month=1, day=31, tzinfo=pytz.utc)

    n_channels = num_input_images
    n_classes = num_output_images
    bilinear = True
    learning_rate = 0.001
    lr_patience = 5

    es_patience = 30

    epochs = 5
    gpus = 1

    dm = PrecipRegressionDataModule(
        batch_size=batch_size,
        path_img_list=path_image_list,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        datetime_train_start=datetime_train_start,
        datetime_train_end=datetime_train_end
    )
    net = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        learning_rate=learning_rate,
        lr_patience=lr_patience
    )

    dm.setup()
    # return
    default_save_path = "output/lightning/precip_regression_jma"
    if not os.path.exists(default_save_path):
        os.makedirs(default_save_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd() + "/" + default_save_path + "/" + net.__class__.__name__ + "/{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=net.__class__.__name__ + "_rain_threshhold_50_"
    )
    lr_logger = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=es_patience,
                                           # is effectively half (due to a bug in pytorch-lightning)
                                           )
    trainer = pl.Trainer(gpus=gpus,
                         weights_summary=None,
                         max_epochs=epochs,
                         weights_save_path=default_save_path,
                         logger=tb_logger,
                         # callbacks=[lr_logger, earlystopping_callback, checkpoint_callback],
                         callbacks=[lr_logger, earlystopping_callback],
                         val_check_interval=0.25,
                         overfit_batches=0.1)
    trainer.fit(net, dm)

    return

if __name__ == '__main__':
    train_regression()




