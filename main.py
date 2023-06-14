# from pytube import YouTube
# import os
# import subprocess
# import librosa
# import soundfile as sf
#
# video_url = "https://www.youtube.com/watch?v=w-tYngyVXLM"
#
# yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
# out_file = yt.streams.filter(only_audio=True).first().download()
# #
# # base, ext = os.path.splitext(out_file)
# # new_file = base + '.wav'
# #
# # subprocess.call(['ffmpeg', '-i', out_file, new_file])
# # os.remove(out_file)
# #
# # audio, sr = librosa.load(new_file, sr=735)
# # # mfcc = librosa.feature.mfcc(y=y, sr=sr)
# # tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
# # print(tempo, beats)
# # sf.write(new_file, audio, int(sr), 'PCM_24')

# if __name__ == "__main__":
#     # config = Config()
#     # app = QtWidgets.QApplication(sys.argv)
#     # main_gui = MainGUI(config)
#     # main_gui.show()
#     # sys.exit(app.exec_())
#     # path = r"D:\music_to_instrument\data\audios\data\w-tYngyVXLM.wav"
#
#     # player = AudioPlayer(path, path, 8000, True)
#     # player.play_both(105, 120, 30, 50.11)
#
#     # YouTubeVideo.get_videos_from_channel("https://www.youtube.com/channel/UCy_QFkH1J7fawYImNXHLszA")
#     # YouTubeVideo.get_videos_from_channel("https://www.youtube.com/@chillseph/videos")
#     m_track_generator = mdb.load_all_multitracks()
#     for m_track in m_track_generator:
#         print(m_track.track_id)

import torch
import torchaudio
import scipy.signal as signal

original_song, original_sr = torchaudio.load(r"D:\music_to_instrument\data\audios\data\w-tYngyVXLM.wav")
cover_song, cover_sr = torchaudio.load(r"D:\music_to_instrument\data\audios\data\-pHfPJGatgE.wav")

_, _, spectrogram_original = signal.spectrogram(original_song, original_sr)
_, _, spectrogram_cover = signal.spectrogram(cover_song, cover_sr)

similarity = signal.correlate(spectrogram_original, spectrogram_cover, mode='same')
match_indices = torch.argmax(torch.tensor(similarity), dim=1)

segments = []
prev_index = match_indices[0]

for i in range(1, len(match_indices)):
    curr_index = match_indices[i]
    if not torch.eq(prev_index, curr_index):
        segments.append((prev_index, i))
        prev_index = curr_index

# Print the matched segments
for segment in segments:
    start_time = segment[0]   # Assuming you have the hop length
    end_time = segment[1]
    print("Matched segment: Start time:", start_time, "End time:", end_time)

