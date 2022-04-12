
import pytorch_lightning as pl

from dataset import DataModule
from models.basic.model import BasicModel


def main():
    """
    Main function
    """
    dm = DataModule('eng','fra')
    pl.seed_everything(12)
    trainer = pl.Trainer(gpus=1,max_epochs=10,
    # limit_train_batches=100,
    limit_val_batches=1
    )
    model = BasicModel(dm.input_lang, dm.output_lang)
    trainer.fit(model,datamodule=dm)
    # trainer.test(model,datamodule=dm)

if __name__ == "__main__":
    main()
