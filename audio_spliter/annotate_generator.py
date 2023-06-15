import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
from torch.multiprocessing import Pool, set_start_method
from torchaudio.transforms import Resample, TimeStretch
import os
import subprocess

# AUDIO_TEST_ORIGINAL_ID = "-pHfPJGatgE"
# AUDIO_TEST_COVER_ID = "w-tYngyVXLM"


class AudioAnnotateGenerator:
    def __init__(self, config):
        set_start_method('spawn')

        self.device = "cuda" if config["use_gpu"] else "cpu"
        self.youtube_audio_path = config["path"]["youtube_audio"]
        self.generates_per_audio = config["generate"]["numbers_per_audio"]
        self.generate_sample_rate = config["generate"]["sample_rate"]
        self.generate_max_padding_seconds = config["generate"]["max_padding_seconds"]
        self.generate_speed_range = config["generate"]["speed_range"]
        self.noise_amplitude = config["generate"]["noise_amplitude"]
        self.seconds_per_segment = config["generate"]["seconds_per_segment"]

        self.speed_resample_num_workers = config["speed_resample_thread"]["num_workers"]
        self.speed_resample_batch_size = config["speed_resample_thread"]["batch_size"]

        self.output_wav = os.path.join(config["path"]["output"], "audios")
        self.output_config = os.path.join(config["path"]["output"], "config")
        self.output_original = os.path.join(config["path"]["output"], "audio_originals")

        self.max_len = config["dataset"]["num_data"]

    def _load_audio_paths(self):
        if not os.path.isdir(self.youtube_audio_path):
            raise Exception("youtube_audio_path is not a directory")
        if not os.path.isdir(self.output_wav):
            os.makedirs(self.output_wav)
        if not os.path.isdir(self.output_config):
            os.makedirs(self.output_config)
        if not os.path.isdir(self.output_original):
            os.makedirs(self.output_original)

        generated_paths = os.listdir(self.output_config)
        audio_paths = []

        for file_path in os.listdir(self.youtube_audio_path):
            if file_path.endswith(".wav"):
                video_id = file_path[:-4]
                for i in range(self.generates_per_audio):
                    if f"{video_id}_{str(i + 1).zfill(2)}.json" in generated_paths:
                        continue
                    audio_paths.append((video_id, i + 1))

        return audio_paths

    def start_generates(self):
        audio_paths = self._load_audio_paths()
        last_est = "nan"
        for video_id, generate_id in audio_paths:
            if len(os.listdir(self.output_config)) > self.max_len:
                print("Total generate done!")
                break
            print(f"Generating audios, {len(os.listdir(self.output_config))}/{self.max_len}, current: {video_id}_{str(generate_id).zfill(2)}, last_ext: {last_est}", end="\r")
            start_time = time.time()
            # print(f"Generating {video_id}_{str(generate_id).zfill(2)}...")
            waveform = self._load_audio(video_id)
            waveform, segment_config = self._generate_audio(waveform)

            config = {
                "seconds_per_segment": self.seconds_per_segment,
                "segments": segment_config
            }

            self.save_config(config, os.path.join(self.output_config, f"{video_id}_{str(generate_id).zfill(2)}.json"))
            self.save_audio(waveform, os.path.join(self.output_wav, f"{video_id}_{str(generate_id).zfill(2)}.wav"))

            last_est = "{:.2f} secs".format(time.time() - start_time)
            # print(f"Done Generate {video_id}_{str(generate_id).zfill(2)}, ext: {last_est}")

    def _load_audio(self, video_id):
        if os.path.exists(os.path.join(self.output_original, f"{video_id}.wav")):
            waveform, _ = torchaudio.load(os.path.join(self.output_original, f"{video_id}.wav"))
            waveform = waveform.to(self.device)
        else:
            waveform, sr = torchaudio.load(os.path.join(self.youtube_audio_path, f"{video_id}.wav"))
            waveform = waveform.to(self.device)
            waveform = self.trim_empty(waveform)
            waveform = Resample(44100, self.generate_sample_rate).to(self.device)(waveform)
            self.save_audio(waveform, os.path.join(self.output_original, f"{video_id}.wav"))

        return waveform

    def _generate_audio(self, waveform):
        start, end = self.get_padding(waveform)

        waveform_segments = self.split_audio(waveform, self.seconds_per_segment)
        waveform_segments = self.change_segments_speed(waveform_segments)
        segment_config = self.generate_segment_config(waveform_segments, start, end)
        waveform = self.combine_segments(waveform_segments)
        waveform = self.cat_padding(waveform, start, end)
        waveform = self.add_noise(waveform)

        return waveform, segment_config

    def save_audio(self, waveform, path):
        torchaudio.save(path, waveform.to("cpu"), self.generate_sample_rate)

    @staticmethod
    def save_config(segment_config, path):
        with open(path, "w") as file:
            json.dump(segment_config, file, indent=2)

    @staticmethod
    def trim_empty(waveform):
        energy = waveform.pow(2).sum(0)

        start_idx = (energy > 0).nonzero(as_tuple=False).min()
        end_idx = (energy > 0).nonzero(as_tuple=False).max()

        return waveform[:, start_idx:end_idx + 1]

    @staticmethod
    def mix_waveform(waveform):
        return torch.mean(waveform, dim=0, keepdim=True)

    def get_padding(self, waveform):
        start, end = int(torch.rand(1) * self.generate_max_padding_seconds * self.generate_sample_rate), \
            int(torch.rand(1) * self.generate_max_padding_seconds * self.generate_sample_rate)
        return start, end

    def cat_padding(self, waveform, start, end):
        start_frame, end_frame = torch.zeros(waveform.ndim, start).to(self.device), \
            torch.zeros(waveform.ndim, end).to(self.device)
        return torch.cat((start_frame, waveform, end_frame), dim=-1)

    def add_noise(self, waveform):
        waveform += (torch.randn_like(waveform) * self.noise_amplitude)
        return torch.clamp(waveform, min=-1.0, max=1.0)

    def split_audio(self, waveform, seconds_per_segment):
        segments = []
        for i in range(0, waveform.shape[-1], int(self.generate_sample_rate * seconds_per_segment)):
            segments.append(waveform[:, i: i + int(self.generate_sample_rate * seconds_per_segment)])
        return segments

    def change_speed(self, waveform, log=None):
        if log is not None:
            print(log)

        speed = (1 - self.generate_speed_range / 2) + torch.rand(1) * self.generate_speed_range
        resample = Resample(self.generate_sample_rate, self.generate_sample_rate // speed).to(self.device)
        return resample(waveform)

    def _change_segments_speed(self, waveform_segments):
        waveform_segments = [segment.to(self.device) for segment in waveform_segments]
        changed_segments = [self.change_speed(segment) for segment in waveform_segments]
        return [segment.to("cpu") for segment in changed_segments]

    def change_segments_speed(self, waveform_segments):
        waveform_segments = [segment.to("cpu") for segment in waveform_segments]
        batches = [waveform_segments[i: i+self.speed_resample_batch_size]
                   for i in range(0, len(waveform_segments), self.speed_resample_batch_size)]

        with Pool(self.speed_resample_num_workers) as pool:
            resampled_batches = pool.map(self._change_segments_speed, batches)

        resampled_segments = []
        for batch in resampled_batches:
            resampled_segments.extend([segment.to(self.device) for segment in batch])

        return resampled_segments

    def generate_segment_config(self, waveform_segments, start, end):
        segment_config = {
            "0": {
                "start": 0,
                "end": start / self.generate_sample_rate,
                "#": "empty start"
            }
        }
        current_frames = start
        for i, segment in enumerate(waveform_segments):
            segment_config[str(i + 1)] = {
                "start": current_frames / self.generate_sample_rate,
                "end": (current_frames + segment.shape[-1]) / self.generate_sample_rate,
            }
            current_frames += segment.shape[-1]

        segment_config[str(len(waveform_segments) + 1)] = {
            "start": current_frames / self.generate_sample_rate,
            "end": (current_frames + end) / self.generate_sample_rate,
            "#": "empty end"
        }

        return segment_config

    @staticmethod
    def combine_segments(waveform_segments):
        return torch.cat(waveform_segments, dim=-1)


if __name__ == "__main__":
    with open("config.json", "r") as j_file:
        audio_spliter_config = json.load(j_file)

    generator = AudioAnnotateGenerator(audio_spliter_config)
    generator.start_generates()
