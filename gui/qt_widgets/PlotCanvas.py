import matplotlib
import torchaudio

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=1000):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor="black")
        super(PlotCanvas, self).__init__(self.figure)
        self.axes_1 = self.figure.add_subplot(2, 1, 1)
        self.axes_2 = self.figure.add_subplot(2, 1, 2)
        self.axes_1.axis("off")
        self.axes_2.axis("off")
        self.axes_1.set_ylim(-0.5, 0.5)
        self.axes_2.set_ylim(-0.5, 0.5)
        self.axes_1.set_frame_on(False)
        self.axes_2.set_frame_on(False)

        # self.axes_2.set_xlim([0, 10000])

        self.figure.tight_layout(pad=0)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update_waveform(self, original_wav, cover_wav):
        self.axes_1.clear()
        self.axes_1.plot(original_wav, color=(0, 1, 0.29))
        self.axes_2.clear()
        self.axes_2.plot(cover_wav, color=(0, 1, 0.29))
        self.draw()
