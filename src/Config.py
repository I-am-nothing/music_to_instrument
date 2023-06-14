import os.path
import threading
import json


# make a class for config with threading lock
class Config:
    # function with read and write config.json
    def __init__(self, file_path="data/config.json"):
        self.lock = threading.Lock()
        self.file_path = file_path
        self._config = {}

    # read config.json
    def load_config(self):
        with self.lock:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as j_file:
                    self._config = json.load(j_file)
                    return self
            else:
                raise FileNotFoundError("config.json not found")

    # write config.json
    def write_config(self):
        with self.lock:
            with open(self.file_path, "w", encoding="utf-8") as j_file:
                json.dump(self._config, j_file, indent=2, ensure_ascii=False)

    # get instrument list
    @property
    def instruments(self):
        return self._config["instruments"]

    # get download_overwrite
    @property
    def download_overwrite(self):
        return self._config["download_overwrite"]

    # get youtube secrets
    @property
    def youtube_secrets(self):
        return self._config["youtube_secrets_file"]

    # get audio path
    @property
    def audio_path(self):
        if not os.path.exists(self._config["audio_path"]):
            os.makedirs(self._config["audio_path"])
        paths = [
            os.path.join(self._config["audio_path"], "__cache__"),
            os.path.join(self._config["audio_path"], "data"),
            os.path.join(self._config["audio_path"], "config")
        ]
        for path in paths:
            if not os.path.isdir(path):
                os.mkdir(path)
        return self._config["audio_path"]
