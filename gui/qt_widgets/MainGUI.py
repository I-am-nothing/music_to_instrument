import threading

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QLineEdit, \
    QFrame, QMessageBox, QHBoxLayout, QTextEdit

from src import YouTubeMP3
from src.YouTubeMP3 import DownloadType
from gui.qt_widgets import ExtendedComboBox


class MainGUI(QWidget):
    def __init__(self, audio_data_root):
        super().__init__()
        self.audio_data_root = audio_data_root
        self.youtube_mp3_downloader = YouTubeMP3.YouTubeVideo(audio_data_root)
        self.originals = {}

        self.original_edt = QLineEdit(self)
        self.original_download_btn = QPushButton("Download", self)
        self.cover_edt = QLineEdit(self)
        self.originals_cbx = ExtendedComboBox(self)
        self.instrument_cbx = ExtendedComboBox(self)
        self.cover_download_btn = QPushButton("Download", self)

        self.log_txt = QTextEdit()

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("YouTube MP3 Downloader")

        self.original_download_btn.clicked.connect(self._original_download)

        # self.originals_cbx.lineEdit().editingFinished.connect(self._originals_cbx_finished)
        # self.instrument_cbx.lineEdit().editingFinished.connect(self._instrument_cbx_finished)
        self._instrument_cbx_finished()
        self._originals_cbx_finished()

        self.cover_download_btn.clicked.connect(self._cover_download)

        self.log_txt.setReadOnly(True)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(16)

        hbox = QHBoxLayout()

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Enter the url of original song:"))
        vbox.addWidget(self.original_edt)
        vbox.addWidget(self.original_download_btn)
        vbox.addWidget(line)
        vbox.addWidget(QLabel("Enter the url of cover song:"))
        vbox.addWidget(self.cover_edt)
        vbox.addWidget(QLabel("Choose the original song:"))
        vbox.addWidget(self.originals_cbx)
        vbox.addWidget(QLabel("Choose the instrument:"))
        vbox.addWidget(self.instrument_cbx)
        vbox.addWidget(self.cover_download_btn)

        widget = QWidget()
        widget.setFixedWidth(400)
        widget.setLayout(vbox)

        hbox.addWidget(widget)
        hbox.addWidget(self.log_txt)

        self.setLayout(hbox)
        self.setContentsMargins(0, 8, 8, 8)
        self.setFixedWidth(700)
        self.show()

    def _originals_cbx_finished(self, download_id=None):
        index = self.originals_cbx.currentIndex()

        self.originals = self.youtube_mp3_downloader.get_originals()

        self.originals_cbx.clear()
        self.originals_cbx.addItems([
            f"({len(item['covers'][self.instrument_cbx.currentText()]) if self.instrument_cbx.currentText() in item['covers'] else 0}) {item['name'][:50]}..."
            for item in self.originals
        ])

        if download_id is not None:
            self.originals_cbx.setCurrentIndex([item['audio_id'] for item in self.originals].index(download_id))
        elif index != -1:
            self.originals_cbx.setCurrentIndex(index)

    def _instrument_cbx_finished(self):
        index = self.instrument_cbx.currentIndex()
        self.instrument_cbx.clear()
        self.instrument_cbx.addItems(self.youtube_mp3_downloader.get_instruments())
        if index != -1:
            self.instrument_cbx.setCurrentIndex(index)

    def _original_download(self):
        if self.original_edt.text().lstrip()[:4] != "http":
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("url must to start with http or https")
            msg_box.exec_()
            self.original_edt.setText("")
            return

        threading.Thread(target=self._audio_download, args=(self.original_edt.text(), DownloadType.ORIGINAL)).start()
        self.original_edt.setText("")

    def _cover_download(self):
        if self.cover_edt.text().lstrip()[:4] != "http":
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("url must to start with http or https")
            msg_box.exec_()
            self.cover_edt.setText("")
            return

        if self.instrument_cbx.currentText() == DownloadType.COVER_GUITAR.value:
            original_id = self.originals[self.originals_cbx.currentIndex()]["audio_id"]
            threading.Thread(target=self._audio_download, args=(self.cover_edt.text(), DownloadType.COVER_GUITAR, original_id)).start()
        self.cover_edt.setText("")

    def _audio_download(self, audio_url, download_type, original_id=None):
        self.log_txt.append(f"starting download {audio_url.strip()[:50]}...\n")
        self._scroll_to_bottom()

        self.youtube_mp3_downloader.download(audio_url.strip(), download_type, original_id).join()

        if self.youtube_mp3_downloader.download_error:
            self.log_txt.append(f"download failed, {self.youtube_mp3_downloader.download_error}\n")
        else:
            self.log_txt.append(f"download {audio_url.strip()[:50]}... successful\n")
        self._scroll_to_bottom()

        self._originals_cbx_finished(self.youtube_mp3_downloader.download_id if original_id is None else None)

    def _scroll_to_bottom(self):
        self.log_txt.verticalScrollBar().setValue(self.log_txt.verticalScrollBar().maximum())