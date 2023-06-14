import json
import os
import random
import shutil

import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from annotate_generator import AudioAnnotateGenerator
from annotate_dataset import AnnotateData, collect_fn
from annotate_model import Annotater


class AnnotateDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.generate_data = config["generate_data"]
        self.num_workers = config["train"]["num_workers"]
        self.batch_size = config["train"]["batch_size"]
        self.num_data = config["dataset"]["num_data"]

        self.num_test = int(self.num_data * config["dataset"]["test"])
        self.num_valid = int(self.num_data * config["dataset"]["validation"])
        self.num_train = self.num_data - self.num_test - self.num_valid

        self.annotate_train, self.annotate_valid, self.annotate_test = None, None, None

    def prepare_data(self):
        if self.generate_data:
            generator = AudioAnnotateGenerator(self.config)
            generator.start_generates()

    def setup(self, stage=None):
        full_data = AnnotateData(self.config)
        self.annotate_train, self.annotate_valid, self.annotate_test, _ = random_split(
            full_data, [
                self.num_train, self.num_valid, self.num_test,
                len(full_data) - self.num_train - self.num_valid - self.num_test
            ]
        )

    def train_dataloader(self):
        return DataLoader(self.annotate_train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collect_fn)

    def val_dataloader(self):
        return DataLoader(self.annotate_valid, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collect_fn)

    def test_dataloader(self):
        return DataLoader(self.annotate_test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collect_fn)


class AnnotateModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.learning_rate = config["train"]["learning_rate"]
        self.annotater = Annotater(config)

        self.status_loss = nn.MSELoss()
        self.position_loss = nn.L1Loss()

    def forward(self, z1, z2, hidden):
        return self.annotater(z1, z2, hidden)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _step(self, batch):
        x_originals, x_changes, y_positions, y_states, batch_size, max_len = batch
        hidden = self.annotater.init_hidden(batch_size)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        y_hat_positions = []
        y_hats_states = []
        for i in range(max_len):
            y_hat_pos, y_hat_sta, (hn, c0) = self(x_originals[:, i], x_changes[:, i], (hn, c0))
            y_hat_positions.append(y_hat_pos)
            y_hats_states.append(y_hat_sta)

        y_hat_positions = torch.stack(y_hat_positions, dim=1)
        y_hats_states = torch.stack(y_hats_states, dim=1)

        pos_loss = self.position_loss(y_hat_positions, y_positions)
        sta_loss = self.status_loss(y_hats_states, y_states)

        return pos_loss, sta_loss

    def training_step(self, batch):
        pos_loss, sta_loss = self._step(batch)

        # self.log("train_pos_loss", pos_loss, prog_bar=True, batch_size=batch[4])
        self.log("train_sta_loss", sta_loss, prog_bar=True, batch_size=batch[4])

        # return (pos_loss + sta_loss) / 2
        return sta_loss

    def validation_step(self, batch, batch_idx):
        pos_loss, sta_loss = self._step(batch)

        # self.log("valid_pos_loss", pos_loss, prog_bar=True, batch_size=batch[4])
        self.log("valid_sta_loss", sta_loss, prog_bar=True, batch_size=batch[4])


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_float32_matmul_precision('medium')
    gpus = min(1, torch.cuda.device_count())
    with open("config.json", "r") as j_file:
        audio_spliter_config = json.load(j_file)

    data_module = AnnotateDataModule(audio_spliter_config)
    model = AnnotateModule(audio_spliter_config)

    if audio_spliter_config["use_gpu"]:
        trainer = pl.Trainer(max_epochs=audio_spliter_config["train"]["epochs"], devices=gpus)
    else:
        trainer = pl.Trainer(max_epochs=audio_spliter_config["train"]["epochs"])

    trainer.fit(model, data_module)
