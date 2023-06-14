import json
import os

import torch
from torch import nn
import torch.nn.functional as fn
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample


class AnnotateData(torch.utils.data.Dataset):
    def __init__(self, config):
        self.device = "cuda" if config["use_gpu"] else "cpu"

        self.annotate_config = os.path.join(config["path"]["output"], "config")
        self.annotate_original = os.path.join(config["path"]["output"], "audio_originals")
        self.annotate_wav = os.path.join(config["path"]["output"], "audios")

        self.generate_sample_rate = config["generate"]["sample_rate"]
        self.train_sample_rate = config["dataset"]["train_sample_rate"]

        self.seconds_per_item = config["dataset"]["seconds_per_item"]
        self.push_seconds = config["dataset"]["push_seconds"]
        self.n_feats = config["dataset"]["n_feats"]
        self.n_fft = config["dataset"]["n_fft"]

        self.annotate_files = [path[:-5] for path in os.listdir(os.path.join(config["path"]["output"], "config"))
                               if path.endswith(".json")]

    def _log_mel_spec(self, waveform):
        try:
            waveform = MelSpectrogram(sample_rate=self.train_sample_rate, n_mels=self.n_feats, n_fft=self.n_fft).to(self.device)(waveform)
            return torch.log(waveform + 1e-14)
        except:
            return None

    def __len__(self):
        return len(self.annotate_files)

    def __getitem__(self, index):
        file_name = self.annotate_files[index]

        original_path = os.path.join(self.annotate_original, file_name[:-3] + ".wav")
        wav_path = os.path.join(self.annotate_wav, file_name + ".wav")
        config_path = os.path.join(self.annotate_config, file_name + ".json")

        with open(config_path, "r", encoding="utf-8") as file:
            wav_config = json.load(file)

        original_wav, original_sr = torchaudio.load(original_path)
        changed_wav, changed_sr = torchaudio.load(wav_path)

        original_wav = original_wav.to(self.device)
        changed_wav = changed_wav.to(self.device)

        if self.generate_sample_rate != self.train_sample_rate:
            original_wav = Resample(self.generate_sample_rate, self.train_sample_rate).to(self.device)(original_wav)
            changed_wav = Resample(self.generate_sample_rate, self.train_sample_rate).to(self.device)(changed_wav)

        x_original = []
        x_change = []
        y_position = []
        y_state = []

        current_secs = 0

        if f"{len(wav_config['segments']) - 1}" not in wav_config["segments"]:
            wav_config["segments"][f"{len(wav_config['segments']) - 1}"] = wav_config["segments"][
                f"{len(wav_config['segments'])}"]
            del wav_config["segments"][f"{len(wav_config['segments']) - 1}"]
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(wav_config, file, indent=2)

        i = 0
        pass_zero = 0
        while i < len(wav_config["segments"]) - 1:
            end_secs = current_secs + self.seconds_per_item

            original_segment = original_wav[
                               :,
                               int((i - pass_zero) * wav_config["seconds_per_segment"] * self.train_sample_rate):
                               int((i + 1 - pass_zero) * wav_config["seconds_per_segment"] * self.train_sample_rate)
                               ]
            changed_segment = changed_wav[
                              :,
                              int(current_secs * self.train_sample_rate): int(end_secs * self.train_sample_rate)
                              ]

            if end_secs < wav_config["segments"][f"{i}"]["end"]:
                if i == 0:
                    y_position.append(torch.tensor([0.0, 0.0]))
                    y_state.append(torch.tensor([0.0]))
                else:
                    y_start_sec = (wav_config["segments"][f"{i}"]["start"] - current_secs)
                    y_position.append(torch.tensor([y_start_sec / self.seconds_per_item, 0.0]))
                    y_state.append(torch.tensor([self.seconds_per_item - y_start_sec]))
                i -= 1
                current_secs += self.push_seconds
            else:
                if i == 0:
                    i += 1
                    pass_zero = 1
                    continue
                y_start_sec = (wav_config["segments"][f"{i}"]["start"] - current_secs)
                y_end_sec = (wav_config["segments"][f"{i}"]["end"] - current_secs)
                y_position.append(torch.tensor([y_start_sec / self.seconds_per_item, y_end_sec / self.seconds_per_item]))
                y_state.append(torch.tensor([1.0]))

            x_original_check = self._log_mel_spec(original_segment)
            x_change_check = self._log_mel_spec(changed_segment)

            x_original.append(x_original_check.to("cpu") if x_original_check is not None else torch.zeros_like(x_original[-1]))
            x_change.append(x_change_check.to("cpu") if x_change_check is not None else torch.zeros_like(x_change[-1]))

            i += 1

        return x_original, x_change, y_position, y_state


def collect_fn(data):
    x_originals = []
    x_changes = []
    y_positions = []
    y_states = []

    for (x_original, x_change, y_position, y_state) in data:
        x_original_max = max([item.size(2) for item in x_original])
        x_change_max = max([item.size(2) for item in x_change])
        x_originals.append(torch.stack([fn.pad(item, (0, x_original_max - item.size(2))) for item in x_original]))
        x_changes.append(torch.stack([fn.pad(item, (0, x_change_max - item.size(2))) for item in x_change]))
        y_positions.append(torch.stack(y_position))
        y_states.append(torch.stack(y_state))

    max_len = max([len(item) for item in y_states])

    x_originals = torch.nn.utils.rnn.pad_sequence(x_originals, batch_first=True, padding_value=0)
    x_changes = torch.nn.utils.rnn.pad_sequence(x_changes, batch_first=True, padding_value=0)
    y_positions = torch.nn.utils.rnn.pad_sequence(y_positions, batch_first=True, padding_value=0)
    y_states = torch.nn.utils.rnn.pad_sequence(y_states, batch_first=True, padding_value=0)

    return x_originals, x_changes, y_positions, y_states, len(y_states), max_len
