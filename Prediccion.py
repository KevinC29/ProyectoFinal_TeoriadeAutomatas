import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = '/Users/danixasolano/Desktop/KEVYN/Proyecto/ModeloVocales.h5'
peso = '/Users/danixasolano/Download/KEVYN/Proyecto/pesosVocales.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = '/Users/danixasolano/Desktop/Proyecto/Fotos/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

#Lectura de la camara
cap = cv2.VideoCapture(0)

#Creacion de objeto para la deteccion y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

#Dibujo de las manos
dibujo = mp.solutions.drawing_utils

while(1):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []
    
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:  
            for id, lm in enumerate(mano.landmark):  
                alto, ancho, c = frame.shape  
                corx, cory = int(lm.x * ancho), int(lm.y * alto)  
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                punto_i1 = posiciones[3]
                punto_i2 = posiciones[17]
                punto_i3 = posiciones[10]
                punto_i4 = posiciones[0]
                punto_i5 = posiciones[9]
                x1, y1 = (punto_i5[1] - 100), (punto_i5[2] - 100)
                ancho, alto = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation = cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg) #Convertir la imagen a una matriz
                x = np.expand_dims(x, axis=0) #Se agrega un nuevo eje
                vector = cnn.predict(x) #Va ser un arreglo de 2 dimensiones
                resultado = vector[0]
                respuesta = np.argmax(resultado) #Entrega el indice del valor más alto 0 | 1 
                if respuesta == 0:
                    print(resultado)
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1,y1-5),1,1.3,(0,255,0),1,cv2.LINE_AA)
                elif respuesta == 1:
                    print(resultado)
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,255),3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1,y1-5),1,1.3,(0,0,255),1,cv2.LINE_AA)
                elif respuesta == 2:
                    print(resultado)
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[2]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                elif respuesta == 3:
                    print(resultado)
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,255),3)
                    cv2.putText(frame, '{}'.format(dire_img[3]), (x1,y1-5),1,1.3,(255,0,255),1,cv2.LINE_AA)
                elif respuesta == 4:
                    print(resultado)
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,255),3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1,y1-5),1,1.3,(0,255,255),1,cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'LETRA_DESCONOCIDA', (x1,y1-5), 1, 1.3, (0,255,255),1,cv2.LINE_AA)
    cv2.imshow("Proyecto Final", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()