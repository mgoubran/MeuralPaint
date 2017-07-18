#! /usr/bin/env python
# Maged Goubran @ 2017, mgoubran@stanford.edu 

# coding: utf-8 

import sys
import glob
import os
import math
from PyQt4 import QtGui, QtCore
import quickpaint as qp

class meural_gui(QtGui.QWidget):
   def __init__(self, parent = None):
      super(meural_gui, self).__init__(parent)
		
      #layout = QVBoxLayout()
      layout = QtGui.QGridLayout()

      self.setLayout(layout)
      self.setWindowTitle("Meural Paint")
      self.resize(1000,500)

      # background color
      # palette = QPalette()
      # palette.setColor(QPalette.Background, Qt.black)
      # self.setPalette(palette)

      #font = QFont("Monaco", 14, QtCore.QFont.Bold)

      self.btn = QtGui.QPushButton("Select input image")
      self.btn.clicked.connect(self.getfile)
		
      layout.addWidget(self.btn,0,0)
      
      self.inlabel = QtGui.QLabel("No file chosen")
      #self.inlabel.setFont(font)
      self.outlabel = QtGui.QLabel("")
      self.filelabel = QtGui.QLabel("")
		
      #layout.addWidget(self.label)
      layout.addWidget(self.filelabel,0,1)
      layout.addWidget(self.inlabel,0,1)
      layout.addWidget(self.outlabel,0,2)
	
      # loop over styles
      styles = glob.glob('styles/*.jpg')

      ncols = 4
      nrows = int(math.ceil(len(styles) / float(ncols)))

      s = -1
      for r in range(nrows):
         r += 1
         for c in range(ncols):
            s += 1
            if s <= len(styles): 
               self.clickbutton = self.clickableimage(styles[s-1],layout)
               layout.addWidget(self.clickbutton, r, c)


   def clickableimage(self, modelpath, layout):

      self.button = QtGui.QPushButton('', self)
      self.button.clicked.connect(lambda : self.handleButton(str(self.filelabel.text()), modelpath, layout))
      self.button.setIcon(QtGui.QIcon(modelpath))
      self.button.setIconSize(QtCore.QSize(170,170))

      return self.button


   def getfile(self):
      
      filename = QtGui.QFileDialog.getOpenFileName(self, 'Open image file', 
         '.',"Image files (*.jpg *.jpeg *.png)")

      self.filelabel.setText(filename)

      pixmap = QtGui.QPixmap(filename)
      scaledpixmap = pixmap.scaled(500,500, QtCore.Qt.KeepAspectRatio)
      self.inlabel.setPixmap(scaledpixmap)

		
   def handleButton(self, filename, modelpath, layout):
      
      print('input file: %s' % filename)

      modelname = os.path.basename(modelpath).split('.')[0]
      model = glob.glob('pre-trained_models/%s*' % modelname)[0]

      print('chosen model: %s' % modelname)

      in_name = os.path.splitext(os.path.basename(filename))
      out_name = str(in_name[0] + '_' + modelname + in_name[1])

      qp.eval_mul_dims(in_path=[filename],out_path=[out_name], model_path=model, device='/gpu:0', 
         batch_size=4, model_arch='pre-trained_models/model.meta', mask=0, blend=0)

      outpixmap = QtGui.QPixmap(out_name)
      scaledoutpixmap = outpixmap.scaled(500,500, QtCore.Qt.KeepAspectRatio)
      self.outlabel.setPixmap(scaledoutpixmap)

      self.savebtn = QtGui.QPushButton("Save output image")
      self.savebtn.clicked.connect(lambda: self.savefile(out_name))

      layout.addWidget(self.savebtn,0,3)


   def savefile(self, outfile):

      savename = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
      os.rename(outfile, savename)


def main():

   app = QtGui.QApplication(sys.argv)
   ex = meural_gui()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
