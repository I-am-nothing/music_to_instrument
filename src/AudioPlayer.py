import threading
import time

import torch
import torchaudio
import sounddevice as sd
from torchaudio.transforms import Resample


class AudioPlayer:
    def __init__(self, original_path, cover_path, sample_rate=8000, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self.sample_rate = sample_rate
        self.original, original_sr = torchaudio.load(original_path)
        self.cover, cover_sr = torchaudio.load(cover_path)

        self.original = self.original.to(self.device)
        self.cover = self.cover.to(self.device)

        if original_sr != sample_rate:
            self.original = Resample(
                original_sr, sample_rate, dtype=self.original.dtype
            ).to(self.device)(self.original)
        if cover_sr != sample_rate:
            self.cover = Resample(
                cover_sr, sample_rate, dtype=self.cover.dtype
            ).to(self.device)(self.cover)

        self.is_original_playing = False
        self.is_cover_playing = False
        self.current_millis = None

    def get_play_millis(self):
        return int(round(time.time() * 1000)) - self.current_millis

    def play_original(self, start_sec, end_sec, volume=1.0):
        if self.is_original_playing or self.is_cover_playing:
            return
        self.is_original_playing = True
        self._play(self._get_play_original(start_sec, end_sec).T.contiguous() * volume)
        self.is_original_playing = False

    def play_cover(self, start_sec, end_sec, volume=1.0):
        if self.is_original_playing or self.is_cover_playing:
            return
        self.is_cover_playing = True
        self._play(self._get_play_cover(start_sec, end_sec).T.contiguous() * volume)
        self.is_cover_playing = False

    def play_both(self, original_start_sec, original_end_sec, cover_start_sec, cover_end_sec, original_vol=1.0,
                  cover_vol=1.0):
        if self.is_original_playing or self.is_cover_playing:
            return

        self.is_original_playing = True
        self.is_cover_playing = True

        original = self._get_play_original(original_start_sec, original_end_sec)
        cover = self._get_play_cover(cover_start_sec, cover_end_sec)

        if original.shape[1] != cover.shape[1]:
            factor = (cover_end_sec - cover_start_sec) / (original_end_sec - original_start_sec)
            original = Resample(
                self.sample_rate, int(self.sample_rate * factor), dtype=original.dtype
            ).to(self.device)(original)

        cover = (cover.T.contiguous() * cover_vol).to("cpu")
        original = (original.T.contiguous() * original_vol).to("cpu")
        original = torch.cat((original, torch.zeros(cover.shape[0] - original.shape[0], original.shape[1])), dim=0)

        self._play(cover + original)

        self.is_original_playing = False
        self.is_cover_playing = False

    def _get_play_original(self, start_sec, end_sec):
        start_frame = round(start_sec * self.sample_rate)
        end_frame = round(end_sec * self.sample_rate)
        return self.original[:, start_frame:end_frame]

    def _get_play_cover(self, start_sec, end_sec):
        start_frame = round(start_sec * self.sample_rate)
        end_frame = round(end_sec * self.sample_rate)
        return self.cover[:, start_frame:end_frame]

    def _play(self, wave):
        self.current_millis = int(round(time.time() * 1000))

        self.stream = sd.OutputStream(
            channels=self.original.shape[0], finished_callback=self.stop, samplerate=self.sample_rate
        )
        self.stream.start()
        self.stream.write(wave)
        self.stream.stop()

        self.current_millis = None

    def stop(self):
        self.stream.stop()

    def get_segment(self, original_start_sec, original_end_sec, cover_start_sec, cover_end_sec):
        return self._get_play_original(original_start_sec, original_end_sec).to("cpu"),\
            self._get_play_cover(cover_start_sec, cover_end_sec).to("cpu")
