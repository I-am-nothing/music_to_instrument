import torch
import torchaudio
from matplotlib import pyplot as plt
from IPython.display import Audio, display


def print_stats(waveform, waveform2, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max A:     {waveform.max().item():6.3f}")
    print(f" - Max B:     {waveform2.max().item():6.3f}")
    print(f" - Min A:     {waveform.min().item():6.3f}")
    print(f" - Min B:     {waveform2.min().item():6.3f}")
    print(f" - Mean A:    {waveform.mean().item():6.3f}")
    print(f" - Mean B:    {waveform2.mean().item():6.3f}")
    print(f" - Std Dev A: {waveform.std().item():6.3f}")
    print(f" - Std Dev B: {waveform2.std().item():6.3f}")
    print()


def plot_waveform(waveform, waveform2, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(2, 1)
    # figure, axes = plt.subplots(num_channels, 1)

    axes[0].plot(time_axis, waveform[0], linewidth=1)
    axes[0].grid(True)
    axes[0].set_ylabel(f'Waveform 1')

    axes[1].plot(time_axis, waveform2[0], linewidth=1)
    axes[1].grid(True)
    axes[1].set_ylabel(f'Waveform 2')

    # if num_channels == 1:
    #     axes = [axes]
    # for c in range(num_channels):
    #     axes[c].plot(time_axis, waveform[c], linewidth=1)
    #     axes[c].grid(True)
    #     if num_channels > 1:
    #         axes[c].set_ylabel(f'Channel {c + 1}')
    #     if xlim:
    #         axes[c].set_xlim(xlim)
    #     if ylim:
    #         axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_spectrogram(waveform, waveform2, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(2, 1)
    # figure, axes = plt.subplots(num_channels, 1)
    axes[0].specgram(waveform[0], Fs=sample_rate)
    axes[0].set_ylabel(f'Waveform 1')

    axes[1].specgram(waveform2[0], Fs=sample_rate)
    axes[1].set_ylabel(f'Waveform 2')

    # if num_channels == 1:
    #     axes = [axes]
    # for c in range(num_channels):
    #     axes[c].specgram(waveform[c], Fs=sample_rate)
    #     if num_channels > 1:
    #         axes[c].set_ylabel(f'Channel {c + 1}')
    #     if xlim:
    #         axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


file1 = "../data/audios1/originals/-pHfPJGatgE.wav"
file2 = "../data/audios1/Edward_Ong/w-tYngyVXLM.wav"

# waveform1, sr1 = torchaudio.load(file1, frame_offset=int(44100*0.55), num_frames=int(44100*2))
# waveform2, sr2 = torchaudio.load(file2, frame_offset=int(44100*4.9), num_frames=int(44100*2))

waveform1, sr1 = torchaudio.load(file1, frame_offset=int(44100*0.5), num_frames=int(44100*2))
waveform2, sr2 = torchaudio.load(file2, frame_offset=int(44100*4.5), num_frames=int(44100*2))

print(waveform1.shape)

# waveform2, sr2 = torchaudio.load(file2)

print_stats(waveform1, waveform2, sample_rate=sr1)
plot_waveform(waveform1, waveform2, sr1)
plot_spectrogram(waveform1, waveform2, sr1)
