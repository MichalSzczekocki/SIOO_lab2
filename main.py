import sys

import numpy
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
        self.iteracje = 0  # wybór

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
            # print(self.funkcja)
            # print(self.poczatek)
            # print(self.stop)
            self.pełzak()

    def updateLabel(self, value):
        self.sliderValue.setText(str(value))

    def updateLabelFloat(self, value):
        self.sliderValue.setText(str(value / 100.0))

    def calculate(self, name_dict):
        math_name_dict = dict(getmembers(math))
        # Ax = np.array([1, 2, 3])
        # Ay = np.array([2, 4, 6])
        # ret = np.array(list(map(lambda x, y: calculate(fun, {'x': x, 'y': y}), Ax, Ay)))

        # fun = self.funkcja
        fun = "x - y + 2 * x ** 2 +2 * x * y + y ** 2"  #####################
        return eval(fun, {**name_dict, **math_name_dict})

    def pełzak(self):
        epsilon = 0.1
        iter = 100

        self.poczatek = 4  ###################
        alfa = 1.0 # odbicie > 0
        beta = 0.5 # kontrakcja 0 < X <1
        gamma = 2.0 # ekspansja > 1
        sigma = 0.5 # reduckcja

        x_array = np.array([self.poczatek, self.poczatek + 1, self.poczatek])
        y_array = np.array([self.poczatek, self.poczatek, self.poczatek + 1])

        ret = np.array(list(
            map(lambda x, y: self.calculate({'x': x, 'y': y}), x_array, y_array)))
        # print(ret)

        # TODO while zbierznosc > epislon || iteracje > self.iteracje

        zbierznosc = 100
        while zbierznosc > epsilon:  # and self.iteracje < iter:

            # ustalenie min i max

            f_Xh = ret[0]
            f_Xl = ret[0]
            index_max = 0
            index_min = 0

            for i in range(len(ret)):
                if ret[i] < f_Xl:
                    f_Xl = ret[i]
                    index_min = i
                if ret[i] > f_Xh:
                    f_Xh = ret[i]
                    index_max = i

            Xh = np.array([[x_array[index_max]], [y_array[index_max]]])
            Xl = np.array([[x_array[index_min]], [y_array[index_min]]])

            # Xo - środek cięzkości
            Xo = 0.5 * numpy.array([[float(x_array[0]) + float(y_array[0])],
                                    [float(x_array[len(x_array) - 1]) + float(
                                        y_array[len(y_array) - 1])]])  # len(y_array)-1

            f_Xo = float(np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), Xo[0], Xo[1]))))
            # Xr - odbice
            Xr = (1 + alfa) * Xo - alfa * Xh

            f_Xr = float(np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), Xr[0], Xr[1]))))
            # Xe - ekspansja
            if f_Xr < f_Xl:
                Xe = Xo + gamma * (Xr - Xo)
                f_Xe = float(np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), Xe[0], Xe[1]))))

                if f_Xe < f_Xh:  # xh <
                    x_array[index_max] = float(Xe[0])
                    y_array[index_max] = float(Xe[1])
                    ret[index_max] = f_Xe

                else:
                    x_array[index_max] = float(Xr[0])
                    y_array[index_max] = float(Xr[1])
                    ret[index_max] = f_Xr

                zbierznosc = 0
                for i in ret:
                    zbierznosc += (i - f_Xo) ** 2
                zbierznosc /= len(ret)
                zbierznosc = zbierznosc ** 0.5
                print(zbierznosc)
            if not f_Xr >= f_Xh:
                x_array[index_max] = float(Xr[0])
                y_array[index_max] = float(Xr[1])
                ret[index_max] = f_Xr

            # Xc kontrakcja

            Xc = beta * Xh + (1 - beta) * Xo

            f_Xc = float(np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), Xc[0], Xc[1]))))

            if f_Xc > f_Xh:
              cosik = 0
              #redukcja

              if f_Xr < f_Xo and f_Xr < f_Xl:
                  x_array[index_max] = float(Xr[0])
                  y_array[index_max] = float(Xr[1])
                  ret[index_max] = f_Xr
                  return Xr

            else:
                x_array[index_max] = float(Xc[0])
                y_array[index_max] = float(Xc[1])
                ret[index_max] = f_Xc

def main():
    app = QApplication(sys.argv)
    ex = Main()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
