import threading

import torch
import torchaudio
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QMessageBox
from matplotlib.figure import Figure
from torch.distributed.elastic.agent.server import Worker

from gui.qt_widgets import ExtendedComboBox, PlotCanvas
from src import YouTubeVideo, AudioPlayer


class MainGUI(QtWidgets.QMainWindow):
    def __init__(self, config):
        QtWidgets.QMainWindow.__init__(self)

        self.config = config.load_config()

        self.ui = uic.loadUi("gui/main.ui", self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/assets/icon.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.plot_timer = QTimer()
        self.plot_timer.setInterval(50)
        self.plot_timer.timeout.connect(self._update_plot)

        self.current_index = 0

        self.audio_player = None

        combo_box1 = self.ui.findChild(QComboBox, "instrument_cbx")
        self.instrument_cbx = ExtendedComboBox(self)
        self.formLayout_2.replaceWidget(combo_box1, self.instrument_cbx)
        combo_box1.setParent(None)
        combo_box1.deleteLater()

        self.instrument_cbx.addItems([item["name"] for item in self.config.instruments])

        self.get_audios_btn.clicked.connect(self._get_audios_btn_clicked)
        self.play_all_btn.clicked.connect(self._play_all_btn_clicked)
        self.load_json_btn.clicked.connect(self._load_json_btn_clicked)

        self.canvas = PlotCanvas(self, width=5, height=4, dpi=100)
        self.verticalLayout.addWidget(self.canvas)

    def _load_json_btn_clicked(self):
        self._update_plot()

    def _update_plot(self):
        original_section, cover_section = self.audio_player.get_segment(self.current_index, self.current_index + 5, 15, 20)
        self.current_index += 0.05
        self.canvas.update_waveform(original_section[0], cover_section[0])

    def _play_all_btn_clicked(self):
        self.current_index = 10
        self.plot_timer.start()

    def _get_audios_btn_clicked(self):
        original_url = self.original_edt.text().strip()
        cover_url = self.cover_edt.text().strip()
        self.original_edt.setText("Fetching...")
        self.cover_edt.setText("Fetching...")

        # if not start with https
        if original_url[:4] != "http" or cover_url[:4] != "http":
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("url must to start with http or https")
            msg_box.exec_()
            self.original_edt.setText("")
            self.cover_edt.setText("")
            return

        threading.Thread(target=self._get_audios, args=[original_url, cover_url]).start()

    def _get_audios(self, original_url, cover_url):
        self.get_audios_btn.setEnabled(False)
        try:
            original_video = YouTubeVideo.get_video_from_url(original_url)
            cover_video = YouTubeVideo.get_video_from_url(cover_url)
        except ValueError:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("Invalid url")
            msg_box.exec_()
            self.original_edt.setText("")
            self.cover_edt.setText("")
            self.get_audios_btn.setEnabled(True)
            return

        original_wav_path, original_config_path = original_video.getMP3(self.config.audio_path, self.config.download_overwrite)
        cover_wav_path, cover_config_path = cover_video.getMP3(self.config.audio_path, self.config.download_overwrite)

        self.original_edt.setText("")
        self.original_txt.setText(original_video.title)
        self.cover_edt.setText("")
        self.cover_txt.setText(cover_video.title)

        self.get_audios_btn.setText("Loading audios...")

        self.audio_player = AudioPlayer(original_wav_path, cover_wav_path)

        self.get_audios_btn.setText("Get audios")
        self.get_audios_btn.setEnabled(True)


