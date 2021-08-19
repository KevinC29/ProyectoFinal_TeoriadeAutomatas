import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model
import spacy
from spacy import displacy

modelo = '/Users/danixasolano/Downloads/KEVYN/Proyecto/ModeloLetras.h5'
peso = '/Users/danixasolano/Downloads/KEVYN/Proyecto/pesosLetras.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = '/Users/danixasolano/Downloads/KEVYN/Proyecto/Fotos/Validacion'
dire_img = os.listdir(direccion)
aux = 0
for i in dire_img:
    print(aux , "=" , i)
    aux = aux + 1

#Lectura de la camara
cap = cv2.VideoCapture(0)

#Creacion de objeto para la deteccion y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

#Dibujo de las manos
dibujo = mp.solutions.drawing_utils

def obtener_letra_espacio(cont_letra, letra, numero):
    if  numero == 1:
        if cont_letra == 1:
            return letra.upper()
        else:
            return letra
    elif numero == 2:
        return letra + " "
    else:
        print("Escribio un numero o valor incorrecto")

def obtener_frase():
    print("Desea guardar la letra: 1 Si - 0 No")
    valor = int(input())
    if valor == 1:
        print("Desea agregar un espacio: 1 Si - 0 No")
        valor2 = int(input())
        if valor2 == 1:
            return 1
        elif valor2 == 0:
            return 2
        else:
            print("Valor erroneo")
    else:
        print("\n")

def tokenizar(frase):
    tokens = frase.split()
    print(tokens)
    return tokens

def analisis_sintáctico(frase):
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(frase)
    return [token.text for token in doc]

def analisis_semantico(frase):
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(frase)
    return displacy.render(doc, style="ent", jupyter="TRUE")



while(1):
    cont_letras = 0
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []
    frase = ""
    letra = ""
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
                x1, y1 = (punto_i5[1] - 200), (punto_i5[2] - 200)
                ancho, alto = (x1 + 100), (y1 + 300)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation = cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg) #Convertir la imagen a una matriz
                x = np.expand_dims(x, axis=0) #Se agrega un nuevo eje
                vector = cnn.predict(x) #Va ser un arreglo de 2 dimensiones
                resultado = vector[0]
                respuesta = np.argmax(resultado) #Entrega el indice del valor más alto 0 | 1 
                if respuesta == 0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[21]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "a"
                elif respuesta == 1:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[7]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "b"
                elif respuesta == 2:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "c"
                elif respuesta == 3:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[2]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "d"
                elif respuesta == 4:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[6]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "e"
                elif respuesta == 5:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[22]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "f"
                elif respuesta == 6:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[18]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "g"
                elif respuesta == 7:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[24]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "h"
                elif respuesta == 8:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[16]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "i"
                elif respuesta == 9:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[8]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "k"
                elif respuesta == 10:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[9]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "l"
                elif respuesta == 11:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "m"
                elif respuesta == 12:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[17]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "n"
                elif respuesta == 13:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[23]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "o"
                elif respuesta == 14:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[11]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "p"
                elif respuesta == 15:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[4]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "q"
                elif respuesta == 16:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[14]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "r"
                elif respuesta == 17:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[20]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "s"
                elif respuesta == 18:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[19]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "t"
                elif respuesta == 19:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[15]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "u"
                elif respuesta == 20:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[3]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "v"
                elif respuesta == 21:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[12]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "w"
                elif respuesta == 22:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[5]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "x"
                elif respuesta == 23:
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (255,0,0),3)
                    cv2.putText(frame, '{}'.format(dire_img[10]), (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)
                    letra = "y"
                else:
                    cv2.putText(frame, 'LETRA_DESCONOCIDA', (x1,y1-5), 1, 1.3, (0,255,255),1,cv2.LINE_AA)
    
    cv2.imshow("Proyecto Final", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()