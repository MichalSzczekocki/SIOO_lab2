import sys
from random import randint

import scipy.optimize
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QRadioButton, QComboBox, QSlider, \
    QPushButton, QLineEdit
from PyQt5.QtCore import Qt
from inspect import getmembers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def record_step(points, value):
    result = []
    """
    for i in range(len(value)):
        temp = []
        for j in points:
            temp.append(j[i])
        temp.append(value[i])
        result.append(temp)
    """
    for i in points:
        result.append(i.copy())
    result.append(value.copy())

    # prev.append(result)
    return result


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
        self.epsilon = 0.0001  # wybór
        self.odbicie = 0  # wybór
        self.kontrakcja = 0  # wybór
        self.ekspansja = 0  # wybór
        self.floatLabel = 0
        self.wymiar = 0
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
                self.slider.setRange(50, 150)
                self.slider.setValue(100)
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

        elif self.etap == 5:  # zapis eksp wybor wymiar
            self.label.setText("Wybierz ilość zmiennych")
            self.etap += 1
            self.ekspansja = self.slider.value()

            self.slider.setRange(2, 4)
            self.slider.setValue(2)
            self.sliderValue.setText('2')
            self.slider.valueChanged.connect(self.updateLabel)

        elif self.etap == 6:  # zapis wymiar wybór funkcji
            self.etap += 1
            self.wymiar = self.slider.value()
            self.layout.removeWidget(self.slider)
            self.layout.removeWidget(self.sliderValue)
            self.slider.deleteLater()
            self.sliderValue.deleteLater()
            self.label.setText("Wpisz funkcję testową")
            self.layout.addWidget(self.lineEdit)
            self.layout.addWidget(self.next)

        elif self.etap == 7:
            self.funkcja = self.lineEdit.text()
            self.hide()

            self.pełzak()

    def updateLabel(self, value):
        self.sliderValue.setText(str(value))

    def updateLabelFloat(self, value):
        self.sliderValue.setText(str(value / self.floatLabel))

    def calculate(self, name_dict):
        math_name_dict = dict(getmembers(np))

        # fun = self.funkcja #TODO odkomentować
        # fun = "x - y + 2 * x ** 2 +2 * x * y + y ** 2"  #####################
        if self.wymiar == 2:
            #fun = "sin((x ** 2 + y ** 2) ** 0.5 )"  #####################
            fun = "(1 - x)**2 + 100*(y - x**2)**2"  #####################
        elif self.wymiar == 3:
            fun = "sin((x ** 2 + y ** 2) ** 0.5 ) + cos(z)"  #####################
        else:
            fun = "sin((x ** 2 + y ** 2) ** 0.5 ) + cos(z) - cos(t)"  #####################

        return eval(fun, {**name_dict, **math_name_dict})

    def draw_graph(self, a, x, y):
        if x < 0:
            xx = int(x - 3)
        else:
            xx = int(x + 3)
        if y < 0:
            yy = int(y - 3)
        else:
            yy = int(y + 3)

        fig = plt.figure()

        x = np.linspace(xx, xx + 10, 30)
        # x = np.arange(-6, 6, 0.5)
        y = np.linspace(yy, yy + 10, 30)
        # y = np.arange(-6, 6, 0.5)

        X, Y = np.meshgrid(x, y)

        Z = np.array(list(
            map(lambda x, y: self.calculate({'x': x, 'y': y}), X, Y)))
        """
        plt.contour(X, Y, Z, levels=list(np.arange(0, 10, 0.5)))
        plt.gca().set_aspect("equal")
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))

        for i in a:
            i[0] = np.append(i[0], i[0][0])
            i[1] = np.append(i[1], i[1][0])
            print(i[0])
            print(i[1])
            plt.plot(i[0], i[1], color='red')
            

        """
        # ax = fig.add_subplot(projection='3d')
        # ax = plt.axes(projection='3d')

        ax = Axes3D(fig)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', zorder=1)
        ax.set_title('surface')

        for i in a:
            # ax = fig.add_subplot(projection='3d')
            # ax.set_xlim3d(-10, 10)
            # ax.set_ylim3d(-10, 10)
            # ax.set_zlim3d(-1, 1)
            # ax = plt.axes(projection='3d')
            i[0] = np.append(i[0], i[0][0])
            i[1] = np.append(i[1], i[1][0])
            i[2] = np.append(i[2], i[2][0])
            ax.plot3D(i[0], i[1], i[2], zorder=10)

        plt.show()

    def fun(self, *args):
        if self.wymiar == 2:
            #return np.sin((args[0][0] ** 2 + args[0][1] ** 2) ** 0.5)
            return (1 - args[0][0])**2 + 100*(args[0][1] - args[0][0]**2)**2
        elif self.wymiar == 3:
            return np.sin((args[0][0] ** 2 + args[0][1] ** 2) ** 0.5) + np.cos(args[0][2])
        else:
            return np.sin((args[0][0] ** 2 + args[0][1] ** 2) ** 0.5) + np.cos(args[0][2]) - np.cos(args[0][3])

    def wbudowana(self):
        t = [self.poczatek] * self.wymiar

        print("Wbudowana")
        x = scipy.optimize.minimize(self.fun, t, method='Nelder-Mead', options={'disp': True})
        for i in range(self.wymiar):
            print(x.x[i])
        # print(x.x[0])
        # print(x.x[1])
        print(x.fun)
        # self.draw_graph(x)

    def pełzak(self):

        self.wbudowana()
        print("Własna")

        self.poczatek = float(self.poczatek)
        steps = []

        epsilon = self.epsilon  # 0.001
        iter = 0

        # self.poczatek = -6.0  ###################
        alfa = self.odbicie  # 1.0  # odbicie > 0
        beta = self.kontrakcja  # 0.5  # kontrakcja 0 < X <1
        gamma = self.ekspansja  # 2.0  # ekspansja > 1

        punkty = []

        for i in range(self.wymiar):
            temp = np.array([])
            for j in range(self.wymiar + 1):
                temp = np.append(temp, float(self.poczatek + randint(-1, 1)))
            punkty.append(temp)

        # x_array = np.array([self.poczatek, self.poczatek + 1, self.poczatek])
        # y_array = np.array([self.poczatek, self.poczatek, self.poczatek + 1])  # D

        if self.wymiar == 2:
            ret = np.array(list(
                map(lambda x, y: self.calculate({'x': x, 'y': y}), punkty[0], punkty[1])))
        elif self.wymiar == 3:
            ret = np.array(list(
                map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), punkty[0], punkty[1], punkty[2])))
        else:
            ret = np.array(list(
                map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), punkty[0], punkty[1],
                    punkty[2], punkty[3])))

        # steps = record_step(steps, punkty, ret)
        # print(steps[len(steps) - 1])

        zbierznosc = 100
        #while zbierznosc > epsilon and self.iteracje > iter:
        while self.iteracje > iter:
            # while self.iteracje > iter:

            record = record_step(punkty, ret)
            # record_step(steps, punkty, ret)
            steps.append(record)
            print(record)

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

            Xh = np.array([])
            Xl = np.array([])

            for i in punkty:
                Xh = np.append(Xh, i[index_max])
                Xl = np.append(Xl, i[index_min])

            # Xh = np.array([[x_array[index_max]], [y_array[index_max]]])  # D
            # Xl = np.array([[x_array[index_min]], [y_array[index_min]]])  # D

            # Xo - środek cięzkości
            # sux = 0
            # suy = 0

            sumy = np.zeros(self.wymiar)
            temp = 0
            for i in range(self.wymiar):
                for j in range(self.wymiar + 1):
                    if j != index_max:
                        sumy[temp] += punkty[i][j]
                temp += 1

            Xo = sumy / float(self.wymiar)

            """
            Xo = 0.5 * numpy.array([[float(x_array[0]) + float(y_array[0])],
                                    [float(x_array[len(x_array) - 1]) + float(
                                        y_array[len(y_array) - 1])]])  # len(y_array)-1
            """

            if self.wymiar == 2:

                f_Xo = np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), np.array([Xo[0]]), np.array([Xo[1]]))))
            elif self.wymiar == 3:
                f_Xo = np.array(list(
                    map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), np.array([Xo[0]]), np.array([Xo[1]]),
                        np.array([Xo[2]]))))
            else:
                f_Xo = np.array(list(
                    map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), np.array([Xo[0]]),
                        np.array([Xo[1]]),
                        np.array([Xo[2]]), np.array([Xo[3]]))))

            # Xr - odbice
            Xr = (1. + alfa) * Xo - alfa * Xh

            if self.wymiar == 2:

                f_Xr = np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), np.array([Xr[0]]), np.array([Xr[1]]))))
            elif self.wymiar == 3:
                f_Xr = np.array(list(
                    map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), np.array([Xr[0]]), np.array([Xr[1]]),
                        np.array([Xr[2]]))))
            else:
                f_Xr = np.array(list(
                    map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), np.array([Xr[0]]),
                        np.array([Xr[1]]),
                        np.array([Xr[2]]), np.array([Xr[3]]))))

            # Xe - ekspansja
            if f_Xr < f_Xl:
                Xe = Xo + gamma * (Xr - Xo)

                if self.wymiar == 2:

                    f_Xe = np.array(list(
                        map(lambda x, y: self.calculate({'x': x, 'y': y}), np.array([Xe[0]]), np.array([Xe[1]]))))
                elif self.wymiar == 3:
                    f_Xe = np.array(list(
                        map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), np.array([Xe[0]]),
                            np.array([Xe[1]]),
                            np.array([Xe[2]]))))
                else:
                    f_Xe = np.array(list(
                        map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), np.array([Xe[0]]),
                            np.array([Xe[1]]),
                            np.array([Xe[2]]), np.array([Xe[3]]))))

                # if f_Xe >= f_Xl:  # xh <
                if f_Xe < f_Xh:

                    temp = 0
                    for i in punkty:
                        i[index_max] = Xe[temp]
                        temp += 1
                    # x_array[index_max] = float(Xe[0])
                    # y_array[index_max] = float(Xe[1])  # D
                    ret[index_max] = f_Xe

                    # steps = record_step(steps, punkty, ret)
                    # print(steps[len(steps) - 1])

                else:
                    temp = 0
                    for i in punkty:
                        i[index_max] = Xr[temp]
                        temp += 1
                    # x_array[index_max] = float(Xr[0])
                    # y_array[index_max] = float(Xr[1])  # D
                    ret[index_max] = f_Xr

                    # steps = record_step(steps, punkty, ret)
                    # print(steps[len(steps) - 1])

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

                        temp = 0
                        for j in punkty:
                            j[index_max] = Xr[temp]
                            temp += 1

                        ret[index_max] = f_Xr

                        # steps = record_step(steps, punkty, ret)
                        # print(steps[len(steps) - 1])

            """
            if f_Xr < f_Xh:  # TODO P8
                x_array[index_max] = float(Xr[0])
                y_array[index_max] = float(Xr[1])
                ret[index_max] = f_Xr
            """
            # Xc kontrakcja

            Xc = beta * Xh + (1 - beta) * Xo

            if self.wymiar == 2:

                f_Xc = np.array(list(
                    map(lambda x, y: self.calculate({'x': x, 'y': y}), np.array([Xc[0]]), np.array([Xc[1]]))))
            elif self.wymiar == 3:
                f_Xc = np.array(list(
                    map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), np.array([Xc[0]]),
                        np.array([Xc[1]]),
                        np.array([Xc[2]]))))
            else:
                f_Xc = np.array(list(
                    map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), np.array([Xc[0]]),
                        np.array([Xc[1]]),
                        np.array([Xc[2]]), np.array([Xc[3]]))))

            if f_Xc >= f_Xh:

                # redukcja

                for i in range(self.wymiar):
                    temp = np.array([])
                    for j in punkty:
                        temp = np.append(temp, j[i])
                    temp = (temp + Xl) / float(self.wymiar)

                    k = 0
                    for j in punkty:
                        j[i] = temp[k]
                        k += 1

                """
                for i in range(len(ret)):
                    temp = np.array([])
                    for j in punkty:
                        temp = np.append(temp, j[i])  # ([[float(x_array[i])], [float(y_array[i])]])
                    temp = (temp + Xl) / float(self.wymiar)
                    for k in range(len(ret)):
                        for n in punkty:
                            n[k] = temp[k]
                """

                if self.wymiar == 2:
                    ret = np.array(list(
                        map(lambda x, y: self.calculate({'x': x, 'y': y}), punkty[0], punkty[1])))
                elif self.wymiar == 3:
                    ret = np.array(list(
                        map(lambda x, y, z: self.calculate({'x': x, 'y': y, 'z': z}), punkty[0], punkty[1], punkty[2])))
                else:
                    ret = np.array(list(
                        map(lambda x, y, z, t: self.calculate({'x': x, 'y': y, 'z': z, 't': t}), punkty[0], punkty[1],
                            punkty[2], punkty[3])))
                # steps = record_step(steps, punkty, ret)
                # print(steps[len(steps) - 1])

                for i in ret:
                    if i != f_Xh and f_Xr < i:

                        temp = 0
                        for j in punkty:
                            j[index_max] = Xr[temp]
                            temp += 1

                        ret[index_max] = f_Xr

                        # steps = record_step(steps, punkty, ret)
                        # print(steps[len(steps) - 1])

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

                temp = 0
                for i in punkty:
                    i[index_max] = Xc[temp]
                    temp += 1
                ret[index_max] = f_Xr

                # steps = record_step(steps, punkty, ret)
                # print(steps[len(steps) - 1])

            # print(iter)
        print(Xl)
        print(f_Xl)
        print("finito")
        if self.wymiar == 2:
            self.draw_graph(steps, Xl[0], Xl[1])


def main():
    app = QApplication(sys.argv)
    ex = Main()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
