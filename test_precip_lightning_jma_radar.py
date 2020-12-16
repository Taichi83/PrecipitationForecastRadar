import datetime, pytz
from PrecipitationForecastRadar.dataloader.regression_lightning_datamodule import PrecipRegressionDataModule
from PrecipitationForecastRadar.models.regression_lightning_unet_precip import UNet
import pytorch_lightning as pl

def test_regression():
    path_image_list = 'dataset/jma_okinawa_2015/file_list.csv'
    num_input_images = 12
    num_output_images = 1
    datetime_train_start = datetime.datetime(year=2015, month=2, day=1, tzinfo=pytz.utc)
    datetime_train_end = datetime.datetime(year=2015, month=2, day=7, tzinfo=pytz.utc)
    model = UNet
    model_folder = 'output/lightning/precip_regression_jma/UNet'
    model_file = 'UNet_rain_threshhold_50_-epoch=4-val_loss=88119.085938.ckpt'

    n_channels = num_input_images
    n_classes = num_output_images
    bilinear = True

    batch_size = 1

    dm = PrecipRegressionDataModule(
        batch_size=batch_size,
        path_img_list=path_image_list,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        datetime_test_start=datetime_train_start,
        datetime_test_end=datetime_train_end
    )
    dm.setup()

    gpus = 1
    epochs =1
    default_save_path = "output/lightning/precip_regression_jma"

    trainer = pl.Trainer(gpus=gpus,
                         weights_summary=None,
                         max_epochs=epochs,
                         weights_save_path=default_save_path,
                         # logger=tb_logger,
                         # callbacks=[lr_logger, earlystopping_callback, checkpoint_callback],
                         # callbacks=[lr_logger, earlystopping_callback],
                         val_check_interval=0.25,
                         overfit_batches=0.1)
    # trainer = pl.Trainer(gpus=gpus)
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    print(trainer.test(model, datamodule=dm))



    print()
    # model_loss = get_model_loss(model, test_dl, loss, denormalize=denormalize)



if __name__ == '__main__':
    test_regression()