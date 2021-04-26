import sys

import numpy
import scipy.optimize
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
        self.iteracje = 100  # wybór
        self.epsilon = 0  # wybór
        self.odbicie = 0  # wybór
        self.kontrakcja = 0  # wybór
        self.ekspansja = 0  # wybór
        self.floatLabel = 0

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

        if self.etap == 0:  # zapis stop wybór start
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

        elif self.etap == 1:  # zapis start wybór epsilon/iteracje
            self.etap += 1
            self.poczatek = self.slider.value()
            if self.stop == "Ilość iteracji":
                self.label.setText("Wybierz ilość iteracji")
                self.slider.setRange(1, 50)
                self.slider.setValue(25)
            else:
                self.label.setText("Wybierz dokładność")
                self.slider.setRange(0, 1000)
                self.slider.setValue(500)
                self.sliderValue.setText('0.5')
                self.slider.valueChanged.connect(self.updateLabelFloat)
                self.floatLabel = 1000.0

        elif self.etap == 2:  # zapis epsilon/iteracje wybor odbicie
            self.label.setText("Wybierz wartość współczynnika odbicia")
            self.etap += 1
            if self.stop == "Ilość iteracji":
                self.iteracje = self.slider.value()
            else:
                self.epsilon = self.slider.value() / self.floatLabel
            self.slider.valueChanged.connect(self.updateLabel)
            self.slider.setRange(1, 10)
            self.slider.setValue(1)

        elif self.etap == 3:  # zapis odbicie wybor kontr
            self.label.setText("Wybierz wartość współczynnika kontrakcji")
            self.etap += 1
            self.slider.valueChanged.connect(self.updateLabelFloat)
            self.odbicie = self.slider.value()
            self.floatLabel = 100.0
            self.slider.setRange(1, 99)
            self.slider.setValue(50)
            self.sliderValue.setText('0.5')
        elif self.etap == 4:  # zapis kontr wybor eksp
            self.label.setText("Wybierz wartość współczynnika ekspansji")
            self.etap += 1
            self.kontrakcja = self.slider.value() / self.floatLabel
            self.floatLabel = 10.0

            self.slider.setRange(1, 10)
            self.slider.setValue(1)
            self.sliderValue.setText('1')
            self.slider.valueChanged.connect(self.updateLabel)

        elif self.etap == 5:  # zapis eksp wybór funkcji
            self.etap += 1
            self.ekspansja = self.slider.value()
            self.layout.removeWidget(self.slider)
            self.layout.removeWidget(self.sliderValue)
            self.slider.deleteLater()
            self.sliderValue.deleteLater()
            self.label.setText("Wpisz funkcję testową")
            self.layout.addWidget(self.lineEdit)
            self.layout.addWidget(self.next)
        elif self.etap == 6:
            self.funkcja = self.lineEdit.text()
            self.hide()

            self.pełzak()

    def updateLabel(self, value):
        self.sliderValue.setText(str(value))

    def updateLabelFloat(self, value):
        self.sliderValue.setText(str(value / self.floatLabel))

    def calculate(self, name_dict):
        math_name_dict = dict(getmembers(np))
        # Ax = np.array([1, 2, 3])
        # Ay = np.array([2, 4, 6])
        # ret = np.array(list(map(lambda x, y: calculate(fun, {'x': x, 'y': y}), Ax, Ay)))

        # fun = self.funkcja #TODO odkomentować
        # fun = "x - y + 2 * x ** 2 +2 * x * y + y ** 2"  #####################
        fun = "sin((x ** 2 + y ** 2) ** 0.5 )"  #####################
        # fun = "(x ** 2 + y ** 2) ** 0.5"  #####################
        return eval(fun, {**name_dict, **math_name_dict})

    def draw_graph(self, a):
        x = np.linspace(-6, 6, 30)
        # x = np.arange(-6, 6, 0.5)
        y = np.linspace(-6, 6, 30)
        # y = np.arange(-6, 6, 0.5)

        X, Y = np.meshgrid(x, y)

        Z = np.array(list(
            map(lambda x, y: self.calculate({'x': x, 'y': y}), X, Y)))

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('surface')

        # plt.plot(a.x[0], a.x[1], a.fun)
        # ax.plot_surface(,)

        plt.show()

    def fun(self, *args):
        return np.sin((args[0][0] ** 2 + args[0][1] ** 2) ** 0.5)

    def wbudowana(self):
        t = [-6, -6]
        x = scipy.optimize.minimize(self.fun, t, method='Nelder-Mead', options={'disp': True})
        print(x.x[0])
        print(x.x[1])
        print(x.fun)
        self.draw_graph(x)
        return

    def pełzak(self):

        epsilon = self.epsilon  # 0.001
        iter = 0

        # self.poczatek = -6.0  ###################
        alfa = self.odbicie  # 1.0  # odbicie > 0
        beta = self.kontrakcja  # 0.5  # kontrakcja 0 < X <1
        gamma = self.ekspansja  # 2.0  # ekspansja > 1

        x_array = np.array([self.poczatek, self.poczatek + 1, self.poczatek])
        y_array = np.array([self.poczatek, self.poczatek, self.poczatek + 1])

        ret = np.array(list(
            map(lambda x, y: self.calculate({'x': x, 'y': y}), x_array, y_array)))
        # print(ret)

        # TODO while zbierznosc > epislon || iteracje > self.iteracje

        zbierznosc = 100
        while zbierznosc > epsilon and self.iteracje > iter:
            # while self.iteracje > iter:

            # ustalenie min i max
            iter += 1

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
            sux = 0
            suy = 0
            for i in range(len(x_array)):
                if i != index_max:
                    sux += x_array[i]
                    suy += y_array[i]
            suma = np.array([[sux], [suy]])
            Xo = suma / float((len(x_array) - 1))

            """
            Xo = 0.5 * numpy.array([[float(x_array[0]) + float(y_array[0])],
                                    [float(x_array[len(x_array) - 1]) + float(
                                        y_array[len(y_array) - 1])]])  # len(y_array)-1
            """
            f_Xo = float(np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), Xo[0], Xo[1]))))
            # Xr - odbice
            Xr = (1. + alfa) * Xo - alfa * Xh

            f_Xr = float(np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), Xr[0], Xr[1]))))
            # Xe - ekspansja
            if f_Xr < f_Xl:
                Xe = Xo + gamma * (Xr - Xo)
                f_Xe = float(np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), Xe[0], Xe[1]))))

                # if f_Xe >= f_Xl:  # xh <
                if f_Xe < f_Xh:
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
                zbierznosc /= float(len(ret))
                zbierznosc = zbierznosc ** 0.5
                # print(zbierznosc)

                if zbierznosc < epsilon:
                    print(Xl)
                    print(f_Xl)
                    return

                continue
            # TODO stop
            for i in ret:
                if i != f_Xh and f_Xr < i:
                    if f_Xr < f_Xh:
                        x_array[index_max] = float(Xr[0])
                        y_array[index_max] = float(Xr[1])
                        ret[index_max] = f_Xr
            """
            if f_Xr < f_Xh:  # TODO P8
                x_array[index_max] = float(Xr[0])
                y_array[index_max] = float(Xr[1])
                ret[index_max] = f_Xr
            """
            # Xc kontrakcja

            Xc = beta * Xh + (1 - beta) * Xo

            f_Xc = float(np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), Xc[0], Xc[1]))))

            if f_Xc >= f_Xh:
                for i in range(len(x_array)):
                    temp = np.array([x_array[i], y_array[i]])
                    temp = (temp + Xl) / 2.
                    x_array[i] = temp[0]
                    y_array[i] = temp[1]
                # redukcja

                for i in ret:
                    if i != f_Xh and f_Xr < i:
                        x_array[index_max] = float(Xr[0])
                        y_array[index_max] = float(Xr[1])
                        ret[index_max] = f_Xr

                # TODO stop

                zbierznosc = 0
                for i in ret:
                    zbierznosc += (i - f_Xo) ** 2
                zbierznosc /= float(len(ret))
                zbierznosc = zbierznosc ** 0.5
                # print(zbierznosc)

                if zbierznosc < epsilon:
                    print(Xl)
                    print(f_Xl)
                    return

                """
                if f_Xr < f_Xo and f_Xr < f_Xl:  # TODO 10
                    x_array[index_max] = float(Xr[0])
                    y_array[index_max] = float(Xr[1])
                    ret[index_max] = f_Xr
                    print("elo")
                    return Xr
                """
            else:
                x_array[index_max] = float(Xc[0])
                y_array[index_max] = float(Xc[1])
                ret[index_max] = f_Xc
            print(iter)
            print(Xl)
            print(f_Xl)
        print("finito")


def main():
    app = QApplication(sys.argv)
    ex = Main()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
