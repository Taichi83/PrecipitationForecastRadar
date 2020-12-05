import os
import torchsummary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import loggers
from PrecipitationForecastRadar.models.unet_precip_regression_lightning import UNet

def train_regression(args_lightning_model_parameters, epochs, gpus=1, es_patience=30):
    net = UNet(**args_lightning_model_parameters)

    torchsummary.summary(net, (12, 288, 288), device="cpu")
    # return
    default_save_path = "output/lightning/precip_regression"
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
                         callbacks=[lr_logger, earlystopping_callback],
                         val_check_interval=0.25,
                         overfit_batches=0.1)
    # resume_from_checkpoint=resume_from_checkpoint,
    trainer.fit(net)
    return


if __name__ == "__main__":
    # model parameters
    args_lightning_model_parameters = {
        'n_channels': 12,
        'n_classes': 1,
        'bilinear': True,
        # 'reduction_ratio': 16,
        # 'kernels_per_layer': 2,
        'dataset_dir': 'data/Dutch_radar/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5',
        'batch_size': 6,
        'learning_rate': 0.001,
        'lr_patience': 4,
        'use_oversampled_dataset': True
    }
    epochs = 1
    gpus =1
    es_patience = 30

    train_regression(args_lightning_model_parameters, epochs, gpus, es_patience)