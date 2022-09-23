"""
    INTERFAZ PARA LA AUTENTIFICACIOND DE USUARIO
    MEDIANTE RECONOCIMIENTO FACIAL

    @autor: Stalin
"""
'''
import cv2
import sys
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QInputDialog

class ProgramaInicial(QMainWindow):
    def __init__(self):
        super(ProgramaInicial, self).__init__()
        loadUi('interfaz.ui', self)
        self.setWindowIcon(QIcon('escudo.png'))     # Agrega el icono en la esquina superior izquierda de la interfaz
        self.setStyleSheet('QMainWindow{background-image: url(blue.jpg)}')      # Agrega la imagen de fondo a la interfaz
        # CLASIFICADOR
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    # Crea el clasificador
        # VARIABLES AUXILIARES
        self.cam = False
        self.cont_int = 0
        self.cont_ini = 0
        # CONEXIONES CON BOTONES
        #self.encender.clicked.connect(self.grabar)          # Conexion boton Encender con funcion grabar
        #self.detener.clicked.connect(self.quitar)           # Conexion boton Detener con funcion quitar
        #self.registrarse.clicked.connect(self.registrar)    # Conexion boton Registrarse con funcion registrar
        #self.actualizar.clicked.connect(self.actual)        # Conexion boton Actualizar datos con funcion actual
        #self.guardar.clicked.connect(self.guardar)          # Conexion boton Guardar datos con funcion guardar
        self.salir.clicked.connect(self.salida)              # Conexion boton Salir con funcion salida

    def salida(self):
        salida = QMessageBox.question(self, 'Advertencia', '¿Desea cerrar el programa?\nTodo el proceso no guardado se perderá',
                                      QMessageBox.Yes, QMessageBox.No)
        if salida == QMessageBox.Yes:
            if self.cam:
                self.cam = False
                self.camara.release()
            self.close()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    identificacion = ProgramaInicial()
    identificacion.setWindowTitle('Interfaz')
    identificacion.show()
    sys.exit(app.exec_())

'''
# CAMARA SIN NECESIDAD DE INTERFAZ
import cv2
import os
import numpy as np

capture = cv2.VideoCapture(0)

# Carga el modelo entrenado que detecta rostros cualquiera
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Carga el detector de caras

cont = 0    # Contador para agregar al nombre de la persona

# Crea la carpeta para almacenar las fotos
dir_proyecto = os.path.dirname(os.path.abspath(__file__))   # Directorio .py
dir_imagenes = os.path.join(dir_proyecto, 'Fotos')          # Agrega la carpeta Fotos
carp_usu = dir_imagenes + '\\Stalin'                        # Agrega la carpeta del usuario especifico
if not os.path.exists(carp_usu):                            # Crea la carpeta Fotos
    os.makedirs(carp_usu)

# RECOLECTA LAS FOTOS DE UN USUARIO
while(True):
    ret, frame = capture.read()

    # Valores default para una deteccion de rostro cualquiera
    scale_factor = 1.2
    min_neighbors = 5
    min_size = (130, 130)
    biggest_only = True
    flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE

    # Convierte la imagen de la camara en blanco y negro
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtiene la posicion de la cara en la imagen
    pos_cara = detector.detectMultiScale(grayscale, 
                                         scaleFactor = scale_factor, 
                                         minNeighbors = min_neighbors, 
                                         minSize = min_size, 
                                         flags = flags)

    # Si detecta un rostro cualquiera dibuja un rectangulo en su posicion
    for (x,y,i,j) in pos_cara:
        # Crea el marco para la cara
        i_rm = int(0.2 * i / 2)             #Crea un pequeño espacio mas grande para el rostro
        cv2.rectangle(frame, (x,y), (x+i,y+j), (255,0,0),2)
        cv2.putText(frame, 'Registrando a ', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (250,0,0), cv2.LINE_4)

        # Guarda el rostro en una imagen .png
        cara = grayscale[y: y + j, x + i_rm: x + i - i_rm]  # Recorta la cara
        cara = cv2.equalizeHist(cara)                       # Normaliza la cara
        #cara = frame[y: y + j, x + i_rm: x + i - i_rm]  # Recorta la cara
        #cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)   # La pasa a escala de grises (Innecesario si la imagen completa esta en escala de grises)
        #cara = cv2.equalizeHist(cara)                   # Normaliza la imagen
        
        # Redimensiona la imagen a una resolucion de 130x130
        if cara.shape < (130,130):                      
            cara_recortada = cv2.resize(cara, (130,130), interpolation = cv2.INTER_AREA)    # Agranda la imagen
        else:
            cara_recortada = cv2.resize(cara, (130,130), interpolation = cv2.INTER_CUBIC)   # Achica la imagen

        # Almacena el rostro en un directorio especifico
        nombre_cara = 'Stalin' + str(cont) + '.png'             # Se asigna el nombre de la imagen
        cv2.imwrite(carp_usu+'\\'+nombre_cara, cara_recortada)  # Guarda la imagen
        cont += 1

    cv2.imshow('Camara', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Almacena los datos en un diccionario
fotos = []          # Almacena las fotos 
nombres = []        # Almacena el numero entero que corresponde al nombre
nombres_dicc = {}   # Almacena la relacion Numerp-Nombre para la posterior identificacion
personas = [persona for persona in os.listdir('Fotos/')]

for i, persona in enumerate(personas):
    nombres_dicc[i] = persona
    for imagen in os.listdir('Fotos/' + persona):
        fotos.append(cv2.imread('Fotos/' + persona + '/' + imagen, 0))
        nombres.append(i)
    nombres = np.array(nombres)

# Entrena el modelo con los datos almacenados
modelo_lpbh = cv2.face.LBPHFaceRecognizer_create()  # Crea el modelo con lpbh
modelo_lpbh.train(fotos, nombres)

# DETECCION CON EL MODELO ENTRENADO ANTERIORMENTE
while(True):
    ret, frame = capture.read()

    # Valores default para una deteccion de rostro cualquiera
    scale_factor = 1.2
    min_neighbors = 5
    min_size = (130, 130)
    biggest_only = True
    flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE

    # Convierte la imagen de la camara en blanco y negro
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtiene la posicion de la cara en la imagen
    pos_cara = detector.detectMultiScale(grayscale, 
                                         scaleFactor = scale_factor, 
                                         minNeighbors = min_neighbors, 
                                         minSize = min_size, 
                                         flags = flags)
    
    for (x,y,w,h) in pos_cara:
        w_rm = int(0.2 * w/2)
        cara = grayscale[y: y + h, w + w_rm: x + w - w_rm]  # Recorta la posicion de la cara en el frame gris
        #cara = frame[y: y + h, w + w_rm: x + w - w_rm]      # Recorta la posicion de la cara en el frame
        #cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)       # La pasa a escala de grises
        cara = cv2.equalizeHist(cara)                       # Normaliza la imagen
        print(type(cara))
        if type(cara) == np.ndarray:
            if cara.shape < (130, 130):                         # Reescala la cara a 130x130
                cara = cv2.resize(cara, (130, 130), interpolation = cv2.INTER_AREA)
            elif cara.shape > (130, 130):
                cara = cv2.resize(cara, (130, 130), interpolation = cv2.INTER_CUBIC)

            result = modelo_lpbh.predict(cara)  # Busca la foto mas parecida en el modelo entrado
                                            # result es un arreglo de 2 valores: La etiqueta y el valor de distancia
            for i, face in enumerate(cara):
                if len(fotos) == 0:
                    cv2.putText(frame, 'Desconocido', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (0,0,250), cv2.LINE_4)
                    cv2.rectangle(frame, (x+w_rm, y), (x+w-w_rm, y+h), (0,0,250), 2)
                else:
                    if result[1] < 90:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        cv2.putText(frame, nombres_dicc[result[0]], (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (250,0,0), cv2.LINE_4)
                    else:
                        cv2.putText(frame, 'Desconocido', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (0,0,250), cv2.LINE_4)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        #'''
        '''
        for i, faces in enumerate(cara):
            if len(fotos) == 0:
                cv2.putText(frame, 'Desconocido', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (0,0,250), cv2.LINE_4)
                cv2.rectangle(frame, (x+w_rm, y), (x+w-w_rm, y+h), (0,0,250), 2)
            else:
                modelo_lpbh.predict(cara)
        '''

    cv2.imshow('Camara', frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

capture.release()
cv2.destroyAllWindows()


'''
def mostrar(self):
        ret, frame = self.camara.read()
        #VALORES ESTANDAR PARA EL RECONOCIMIENTO FACIAL DE CUALQUIER PERSONA, ESTO DETECTA UNA CARA CUALQUIERA
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (130,130)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE
        pos_cara = self.detector.detectMultiScale(frame,
                                             scaleFactor = scale_factor,
                                             minNeighbors = min_neighbors,
                                             minSize = min_size, flags = flags)
        for(x,y,w,h) in pos_cara:
            w_rm = int(0.2 * w / 2)
            self.caras = self.cortar_caras(frame, pos_cara)  #Recorta la cara
            self.caras = self.normalizar()         #Normaliza la foto de la cara
            self.caras = self.reescalar()          #La reescala para compararla con la imagen registrada

            for i, cara in enumerate(self.caras):
                if len(self.fotos) == 0:    #SI EL MODELO NO ESTA ENTRENADO, TODAS LAS PERSONAS DETECTADAS SON DESCONOCIDOS Y MANDA UN CORREO AL USUARIO
                    cv2.putText(frame, 'Desconocido', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (0,0,250), cv2.LINE_4)
                    cv2.rectangle(frame, (x+w_rm, y), (x+w-w_rm, y+h), (0,0,250), 2)
                    if not (len(pos_cara) == self.cont_ini):
                        self.aviso(frame)
                        self.cont_ini = len(pos_cara)
                else:
                    self.distancia = cv2.face.MinDistancePredictCollector() #CREA UNA VARIABLE PARA EL CALCULO DE LA PRESICION
                    self.modelo_lpbh.predict(cara, self.distancia)      #PREDICE LA CARA INGRESADA CON LA VARIABLE DISTANCIA Y DETERMINA QUE TAN CERCA ESTA EN LA "NUBE" DE FOTOS
                    self.certeza_lpbh = self.distancia.getDist()        #DETERMINA LA DISTANCIA DE QUE TAN CERCA ESTA LA FOTO CON LA "NUBE" ALMACENADA
                    self.prediccion_lpbh = self.distancia.getLabel()    #ENTREGA LA ETIQUETA DE LA PREDICCION
                    print "Certeza LPBH: " + str(self.certeza_lpbh) + "Nombre: " + self.nombres_dic[self.prediccion_lpbh]   #Imprime la distancia a la predccion realizada
                    if self.certeza_lpbh < 75:      #sI ES UNA DISTANCIA PEQUEÑA, RECONOCE EL USUARIO Y REMARCA EN MANTALLA UN CUADRO CON SU ETIQUETA
                        cv2.putText(frame, self.nombres_dic[self.prediccion_lpbh], (pos_cara[i][0], pos_cara[i][1]-5), cv2.FONT_ITALIC, 1, (0,250,0), cv2.LINE_4)
                        cv2.rectangle(frame, (x+w_rm, y), (x+w-w_rm, y+h), (250,0,0), 2)
                    else:       #EN CASO CONTRARIO, MARCA A LA PERSONA COMO ALGUIEN NO IDENTIFICADO
                        cv2.putText(frame, 'Desconocido', (pos_cara[0][0], pos_cara[0][1]-5), cv2.FONT_ITALIC, 1, (0,0,250), cv2.LINE_4)
                        cv2.rectangle(frame, (x+w_rm, y), (x+w-w_rm, y+h), (250,0,0), 2)
                        if not (len(pos_cara) == self.cont_int):
                            self.aviso(frame)
                            self.cont_int = len(pos_cara)
        if self.cam:        #SI LA CAMARA ESTA ENCENDIDA, SE ACTUALIZA EL FRAME DE LA CAMARA A LA SALIDA EN UNA LABEL, INVIERTIENDO LA MATRIZ DE COLORES PORQUE ESE ORDEN NECESITA EL LABEL
            frame = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            frame = frame.rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(frame))


    def cortar_caras(self, frame, pos_cara):      #De la imagen capturada, recorta la posicion donde esta la cara identificada
        caras = []
        for(x,y,w,h) in pos_cara:
            w_rm = int(0.2 * w / 2)
            caras.append(frame[y: y + h, x + w_rm: x + w - w_rm])
        return caras

    def normalizar(self):              #Normaliza los colores de la imagen para luego realizar la comparacion
        imagen_norm = []
        for imagen in self.caras:
            if len(imagen.shape) == 3:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen_norm.append(cv2.equalizeHist(imagen))
        return imagen_norm

    def reescalar(self):       #Deja la imagen de la cara en dimensiones de 130x130 para comparar con las registradas
        frame_norm = []
        for imagen in self.caras:
            if imagen.shape < (130,130):
                imagen_norm = cv2.resize(imagen, (130,130), interpolation = cv2.INTER_AREA)
            else:
                imagen_norm = cv2.resize(imagen, (130,130), interpolation = cv2.INTER_CUBIC)
            frame_norm.append(imagen_norm)
        return frame_norm
'''