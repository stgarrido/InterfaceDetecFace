"""
    INTERFAZ PARA LA AUTENTIFICACIOND DE USUARIO
    MEDIANTE RECONOCIMIENTO FACIAL

    @autor: Stalin
"""

import cv2
import sys
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

# Valores default para una deteccion de rostro cualquiera
scale_factor = 1.2
min_neighbors = 5
min_size = (130, 130)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
        cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
        cv2.CASCADE_SCALE_IMAGE

######################################  VENTANA DE INSTRUCCIONES REGISTRO ######################################

class Informaciones(QMainWindow):
    def __init__(self):
        super(Informaciones, self).__init__()
        loadUi('interfaz/info.ui', self)
        self.setWindowIcon(QIcon('interfaz/escudo.png'))
        self.setStyleSheet('QMainWindow{background-image: url(interfaz/black.jpg)}')
        self.setWindowTitle('Informaciones')
        self.pushButton.clicked.connect(self.cerrar)
        self.show()

    def cerrar(self):
        self.close()

######################################  VENTANA DE REGISTRO DE USUARIO ######################################

class Registro(QMainWindow):
    def __init__(self):
        super(Registro, self).__init__()
        loadUi('interfaz/registro.ui', self)
        self.setWindowIcon(QIcon('interfaz/camara.jpg'))
        self.setStyleSheet('QMainWindow{background-image: url(interfaz/blue.jpg)}')
        self.setWindowTitle('Registro')
        # CLASIFICADOR
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    # Carga el clasificador detector
        # VARIABLES AUXILIARES
        self.cam = False
        self.cont = 0
        # INTERFAZ
        self.show()
        # CONEXIONES CON BOTONES
        
        self.informacion.clicked.connect(self.informaciones)
        self.salir.clicked.connect(self.salida)

    def informaciones(self):
        self.info = Informaciones()

    def salida(self):
        if self.cam:
            self.cam = False
            self.camara.release()
        self.close()


########################################## PROGRAMA INICIAL ##########################################

class ProgramaInicial(QMainWindow):
    def __init__(self):
        super(ProgramaInicial, self).__init__()
        loadUi('interfaz/interfaz.ui', self)
        self.setWindowIcon(QIcon('interfaz/escudo.png'))
        self.setStyleSheet('QMainWindow{background-image: url(interfaz/blue.jpg)}')
        # CLASIFICADOR
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    # Carga el clasificador detector
        # PARA INICIAR EL PROGRAMA ENTRENADO
        
        # VARIABLES AUXILIARES
        self.cam = False
        self.cont_int = 0
        self.cont_ini = 0
        # CONEXIONES CON BOTONES
        self.encender.clicked.connect(self.grabar)
        self.detener.clicked.connect(self.quitar)
        self.registrarse.clicked.connect(self.registrar)
        self.salir.clicked.connect(self.salida)
        self.lomito()
        self.label_2.setPixmap(QPixmap('interfaz/logo_eln.png'))

    def lomito(self):
        self.figura = cv2.imread('interfaz/figura.jpg')
        self.figura = cv2.resize(self.figura, (640, 480), interpolation = cv2.INTER_CUBIC)
        self.figura = QImage(self.figura, self.figura.shape[1], self.figura.shape[0], QImage.Format_RGB888)
        self.figura = self.figura.rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.figura))

    # FUNCIONES DE BOTONES
    # Inicia el programa
    def grabar(self):
        if self.cam:
            self.advertencia1()
        else:
            self.cont_init = 0
            self.cam = True
            self.camara = cv2.VideoCapture(0)
            # Contador que muestra cada 20ms un frame de la camara
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.mostrar)
            self.timer.start(20)
    
    #Enciende la camara
    def mostrar(self):
        ret, frame = self.camara.read()
        if ret == False:
            return True
        # Convierte la imagen de la camara en blanco y negro
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Obtiene la posicion de la cara en la imagen
        pos_cara = self.detector.detectMultiScale(grayscale, 
                                                  scaleFactor = scale_factor, 
                                                  minNeighbors = min_neighbors, 
                                                  minSize = min_size, 
                                                  flags = flags)
        # Dibujo de rectangulo en las caras detectadas
        for (x,y,w,h) in pos_cara:
            self.caras = grayscale[y: y + h, x: x + w]  # Recorta la posicion de la cara en el frame gris
            self.caras = cv2.equalizeHist(self.caras)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 2)

        frame = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        frame = frame.rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(frame))

    # Apaga la carama
    def quitar(self):
        if self.cam:
            self.cam = False
            self.camara.release()
            self.lomito()
        else:
            self.advertencia2()

    # Registra un nuevo usuario
    def registrar(self):
        self.ventana2 = Registro()
        if self.cam:
            self.cam = False
            self.camara.release()
            self.lomito()

    # Cierra el programa en modo seguro (apaga la camara)
    def salida(self):
        salida = QMessageBox.question(self, 'Advertencia', '¿Desea cerrar el programa?\nTodo el proceso se perderá',
                                            QMessageBox.Yes, QMessageBox.No)
        if salida == QMessageBox.Yes:
            if self.cam:
                self.cam = False
                self.camara.release()
            self.close()

    # CUADROS DE ADVERTENCIA
    def advertencia1(self):
        QMessageBox.information(self, 'Información', 'El programa ya está identificando',
                                QMessageBox.Ok)
    
    def advertencia2(self):
        QMessageBox.information(self, 'Error', 'La cámara ya está apagada',
                                QMessageBox.Ok)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    identificacion = ProgramaInicial()
    identificacion.setWindowTitle('Interfaz')
    identificacion.show()
    sys.exit(app.exec_())