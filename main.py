import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QRadioButton, QComboBox, QSlider, \
    QPushButton, QLineEdit
from PyQt5.QtCore import Qt
from inspect import getmembers
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize


class Main(QWidget):

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.cb = QComboBox()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-100, 100)
        self.sliderValue = QLabel('0')
        self.slider.valueChanged.connect(self.updateLabel)
        self.sliderValue.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.etap = 0
        self.stop = "Ilość iteracji"
        self.wybor = "Ilość iteracji"
        self.funkcja = "brak"
        self.poczatek = 0

        self.layout = QVBoxLayout()

        self.label = QLabel("Wybierz warinek stopu")
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.layout.addWidget(self.label)

        self.b1 = QRadioButton("Ilość iteracji")
        self.b1.setChecked(True)
        self.b1.toggled.connect(lambda: self.radioButtonZmiana(self.b1))
        self.layout.addWidget(self.b1)

        self.b2 = QRadioButton("Brak postępu")
        self.b2.toggled.connect(lambda: self.radioButtonZmiana(self.b2))

        self.layout.addWidget(self.b2)
        self.setLayout(self.layout)

        self.next = QPushButton('OK')
        self.next.clicked.connect(self.zmianaEtapu)
        self.layout.addWidget(self.next)

        self.lineEdit = QLineEdit()

    def radioButtonZmiana(self, b):

        if b.isChecked():
            self.wybor = b.text()

    def zmianaEtapu(self):

        if self.etap == 0:
            self.etap += 1
            self.stop = self.wybor
            self.label.setText("Wybierz punkt startowy")
            self.layout.removeWidget(self.b1)
            self.layout.removeWidget(self.b2)
            self.layout.removeWidget(self.next)
            self.b1.deleteLater()
            self.b2.deleteLater()
            self.layout.addWidget(self.slider)
            self.layout.addWidget(self.sliderValue)
            self.layout.addWidget(self.next)

        elif self.etap == 1:
            self.etap += 1
            self.poczatek = self.slider.value()
            self.layout.removeWidget(self.slider)
            self.layout.removeWidget(self.sliderValue)
            self.slider.deleteLater()
            self.sliderValue.deleteLater()
            self.label.setText("Wybierz funkcję testową")
            self.layout.addWidget(self.lineEdit)
            self.layout.addWidget(self.next)

        elif self.etap == 2:
            self.next.setDisabled(True)
            self.funkcja = self.lineEdit.text()
            self.layout.removeWidget(self.lineEdit)
            self.lineEdit.deleteLater()
            self.label.setText("Podaj parametry")
            print(self.funkcja)
            print(self.poczatek)
            print(self.stop)

    def updateLabel(self, value):
        self.sliderValue.setText(str(value))

    def updateLabelFloat(self, value):
        self.sliderValue.setText(str(value / 100.0))

    def calculate(function, name_dict):
        math_name_dict = dict(getmembers(math))

        # Ax = np.array([1, 2, 3])
        # Ay = np.array([2, 4, 6])
        # ret = np.array(list(map(lambda x, y: eval_math_fn(fun, {'x': x, 'y': y}), Ax, Ay)))

        return eval(function, {**name_dict, **math_name_dict})


def main():
    app = QApplication(sys.argv)
    ex = Main()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
