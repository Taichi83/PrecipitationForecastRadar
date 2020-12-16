import pytorch_lightning as pl
from torch import nn, optim


class UNet_base(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.001, lr_patience: int = 5) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                              mode="min",
                                                              factor=0.1,
                                                              patience=self.lr_patience),
            'monitor': 'val_loss',  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        # reduction="mean" is average of every pixel, but I want average of image
        # return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / (y_true.size(-1) * y_true.size(-2))

        return nn.functional.mse_loss(y_pred, y_true, reduction="mean") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output['loss']

        loss_mean /= len(outputs)
        self.log("train_loss", loss_mean)

        # return {"log": {"train_loss": loss_mean},
        #         "progress_bar": {"train_loss": loss_mean}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        # val_loss += loss.item()
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["val_loss"]
        avg_loss /= len(outputs)
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs,
                "progress_bar": {"val_loss": avg_loss}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"test_loss": val_loss}

    def test_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["test_loss"]
        avg_loss /= len(outputs)
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs,
                "progress_bar": {"test_loss": avg_loss}}
