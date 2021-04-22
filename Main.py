from PyQt5 import QtCore, QtGui, QtWidgets
from Detection import Ui_Detection


class Ui_Main(object):
    def img(self):
        self.Dialog = QtWidgets.QDialog()
        self.ui = Ui_Detection()
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()
    def vid(self):
        self.Dialog = QtWidgets.QDialog()
        self.ui = Ui_Detection()
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1100, 770)
        Dialog.setStyleSheet("QDialog{background-image: url(../DetectionCounting/img/main.jpg);}")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(410, 250, 381, 411))
        self.frame.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(100, 100, 100, 77), stop:1 rgba(255, 255, 255, 255));")
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(2)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(80, 40, 211, 51))
        self.label_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.label_2.setStyleSheet("font: 75 23pt \"Tahoma\";\n"
"color: rgb(0, 0, 0);\n"
"background-color: rgb(220, 219, 217);")
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setLineWidth(2)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(80, 170, 231, 61))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("font: 75 14pt \"Arial\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(176, 70, 99);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 270, 231, 61))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("font: 75 14pt \"Arial\";\n"
"background-color: rgb(176, 70, 99);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(100, 10, 931, 81))
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.label.setStyleSheet("font: 75 28pt \"Tahoma\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:1 rgba(0, 85, 0, 150));\n"
"color: rgb(255, 255, 255);")
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(5)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.img)
        self.pushButton_2.clicked.connect(self.vid)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "Detection"))
        self.pushButton.setText(_translate("Dialog", "Detection with\n"
"Image"))
        self.pushButton_2.setText(_translate("Dialog", "Detection with\n"
"Video"))
        self.label.setText(_translate("Dialog", "Real-time Human Detection & Counting"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Main()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
