# coding=utf-8


from neural_windows.ui import Ui_MainWindow
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys, os
from quickpaint import eval_mul_dims


class neural_style_transfer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(neural_style_transfer, self).__init__(parent)
        self.setupUi(self)
        #
        QObject.connect(self.shanshui, SIGNAL("clicked()"), self.fshanshui)
        QObject.connect(self.youhua, SIGNAL("clicked()"), self.fyouhua)
        QObject.connect(self.shuimohua, SIGNAL("clicked()"), self.fshuimohua)
        QObject.connect(self.contentButton, SIGNAL("clicked()"), self.show_contDialog)
        # QObject.connect(self.contentSlider, SIGNAL("valueChanged(int)"), self.get_content_slider)
        # QObject.connect(self.styleSlider, SIGNAL("valueChanged(int)"), self.get_style_slider)
        QObject.connect(self.runButton, SIGNAL("clicked()"), self.transfer)

        self.ckpt = ""
        self.style_weight = 0.
        self.content_weight = 1e-3
        self.content_img = None
        self.out_path = './out'
        self.outputs = []

    def fshanshui(self):
        self.ckpt = 'transfer/checkpoint/rain_princess.ckpt'

    def fyouhua(self):
        self.ckpt = 'transfer/checkpoint/scream.ckpt'

    def fshuimohua(self):
        self.ckpt = 'transfer/checkpoint/wreck.ckpt'

    def get_style_slider(self):
        value = self.contentSlider.value()
        self.style_weight = value * 0.1

    def get_content_slider(self):
        value = self.contentSlider.value()
        self.content_weight = value * 0.1

    #
    def show_contDialog(self):
        filename = QFileDialog.getOpenFileName(self, 'open file', './')
        assert filename.split('.')[-1] in ['jpg', 'png']
        self.content_img = str(filename)
        self.content_label.setPixmap(QPixmap(filename))

    # style transfer
    def transfer(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        device = '/cpu:0'
        out_path = self.out_path + '/out' + self.content_img.split('/')[-1]
        self.outputs = eval_mul_dims(self.content_img, out_path, self.ckpt, device=device)
        self.content_label.setPixmap(QPixmap(out_path))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    neural_st = neural_style_transfer()
    neural_st.show()
    app.exec_()