import json

from pytube import YouTube, Channel
import os
import subprocess

from pytube.exceptions import RegexMatchError


class YouTubeVideo:
    @staticmethod
    def get_videos_from_channel(channel_url):
        try:
            channel = Channel(channel_url)
            return channel.channel_name, [YouTubeVideo(video) for video in channel.videos]

        except RegexMatchError:
            raise ValueError("invalid channel url")

    @staticmethod
    def get_video_from_url(video_url):
        try:
            video = YouTube(video_url)
            return YouTubeVideo(video)

        except RegexMatchError:
            raise ValueError("invalid video url")

    def __init__(self, video):
        self.video = video

    @property
    def id(self):
        return self.video.video_id

    @property
    def title(self):
        return self.video.title

    @property
    def description(self):
        return self.video.description

    @property
    def author(self):
        return self.video.author

    @property
    def keywords(self):
        return self.video.keywords

    @property
    def views(self):
        return self.video.views

    @property
    def link(self):
        return self.video.watch_url

    def getMP3(self, audio_path, download_overwrite):
        mp4_path = os.path.join(audio_path, "__cache__", self.id + ".mp4")
        wav_path = os.path.join(audio_path, "data", self.id + ".wav")
        config_path = os.path.join(audio_path, "config", self.id + ".json")

        if not (os.path.exists(wav_path) and os.path.exists(config_path)) or download_overwrite:
            self.video.streams.filter(only_audio=True).first().download(
                output_path=os.path.join(audio_path, "__cache__"),
                filename=self.id + ".mp4",
                skip_existing=True
            )
            subprocess.call([
                'ffmpeg', '-y', '-i',
                mp4_path,
                wav_path
            ])
            os.remove(mp4_path)

            with open(config_path, "w", encoding="utf-8") as j_file:
                json.dump(self.to_dict(), j_file, indent=2, ensure_ascii=False)

        return wav_path, config_path

    def to_dict(self):
        return {
            "title": self.video.title,
            "description": self.video.description,
            "author": self.video.author,
            "keywords": self.video.keywords,
            "views": self.video.views,
            "link": self.video.watch_url
        }


    # def __init__(self, data_root):
    #     super().__init__()
    #     self.data_root = data_root
    #     self.config_lock = threading.Lock()
    #     self.config_writes = 0
    #     self.config = {}
    #     self.download_error = None
    #     self.download_id = None
    #
    #     self.init_data()
    #     self.reload_config()
    #
    # def init_data(self):
    #     if not os.path.isdir(self.data_root):
    #         os.makedirs(self.data_root)
    #     if not os.path.isdir(os.path.join(self.data_root, "originals")):
    #         os.mkdir(os.path.join(self.data_root, "originals"))
    #     if not os.path.isdir(os.path.join(self.data_root, "config_backups")):
    #         os.mkdir(os.path.join(self.data_root, "config_backups"))
    #     if not os.path.exists(os.path.join(self.data_root, "config.json")):
    #         self.config = {
    #             "audios1": {
    #                 # "_Video_ID": {
    #                 #     "name": "_Video_Name",
    #                 #     "description": "_Video_Description_",
    #                 # }
    #             },
    #             "instruments": [
    #                 "guitar",
    #             ],
    #             "audios_relation": {
    #                 # "_Original_ID_": {
    #                 #     "music_id": "_Video_ID_",
    #                 #     "cover": {
    #                 #         "guitar": [{
    #                 #             "channel_id": "_Channel_ID_",
    #                 #             "cover_id": "_Video_ID_",
    #                 #             "prompts": "_Prompts_"
    #                 #         }]
    #                 #     }
    #                 # }
    #             }
    #         }
    #         self.write_config()
    #
    # def reload_config(self):
    #     with open(os.path.join(self.data_root, "config.json"), "r", encoding='utf-8') as j_file:
    #         self.config = json.load(j_file)
    #
    # def write_config(self):
    #     with self.config_lock:
    #         with open(os.path.join(self.data_root, "config.json"), "w", encoding='utf-8') as j_file:
    #             json.dump(self.config, j_file, indent=2, ensure_ascii=False)
    #
    #     self.config_writes += 1
    #     if self.config_writes == 10:
    #         self.config_backup()
    #         self.config_writes = 0
    #
    # def config_backup(self):
    #     shutil.copy2(
    #         os.path.join(self.data_root, "config.json"),
    #         os.path.join(self.data_root, "config_backups", datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".json")
    #     )
    #
    # def get_originals(self):
    #     original_audios = os.listdir(os.path.join(self.data_root, "originals"))
    #     for index, data in enumerate(original_audios):
    #         base, config = self.get_audio_config(data)
    #         audio_config = config
    #         audio_config["audio_id"] = base
    #         audio_config["covers"] = self.config["audios_relation"][base] \
    #             if base in self.config["audios_relation"] else {}
    #         original_audios[index] = audio_config
    #
    #     return original_audios
    #
    # def get_audio_config(self, audio_id):
    #     base, _ = os.path.splitext(audio_id)
    #     return base, self.config["audios1"][base].copy()
    #
    # def get_instruments(self):
    #     return self.config["instruments"]
    #
    # def download(self, video_url, download_type, original_id=None):
    #     thread = threading.Thread(target=self._download, args=(video_url, download_type, original_id))
    #     thread.start()
    #     return thread
    #
    # def _download(self, video_url, download_type, original_id=None):
    #     try:
    #         output_path = ""
    #         yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
    #
    #         if download_type == DownloadType.ORIGINAL:
    #             output_path = os.path.join(self.data_root, "originals")
    #         else:
    #             output_path = os.path.join(self.data_root, yt.author.replace(" ", "_"))
    #
    #         yt.streams.filter(only_audio=True).first().download(
    #             output_path=output_path,
    #             filename=yt.video_id + ".mp3",
    #             skip_existing=True
    #         )
    #
    #         self.config["audios1"][yt.video_id] = {
    #             "name": yt.title,
    #             "description": yt.description,
    #             "author": yt.author,
    #             "keywords": yt.keywords,
    #             "views": yt.views,
    #             "link": yt.watch_url
    #         }
    #
    #         if download_type != DownloadType.ORIGINAL:
    #             if not original_id:
    #                 raise ValueError("missing value: original_id")
    #             if original_id not in self.config["audios_relation"]:
    #                 self.config["audios_relation"][original_id] = {}
    #             if download_type.value not in self.config["audios_relation"][original_id]:
    #                 self.config["audios_relation"][original_id][download_type.value] = {}
    #             self.config["audios_relation"][original_id][download_type.value][yt.video_id] = {
    #                 "prompt": None,
    #                 "negative_prompt": None
    #             }
    #
    #         self.write_config()
    #         self.download_error = None
    #         self.download_id = yt.video_id
    #
    #     except RegexMatchError:
    #         self.download_error = "url not found"
