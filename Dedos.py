import cv2
import mediapipe as mp
import os

# Se crea la carpeta donde se almacenara el entrenameinto
nombre = 'Letra_A'  # Letras del abecedario
direccion = '/Users/danixasolano/Download/KEVYN/Proyecto/Fotos/Entrenamiento'  # Entrenamiento 
#direccion = '/Users/danixasolano/Download/KEVYN/Proyecto/Fotos/Validacion' # Validacion
carpeta = direccion + '/' + nombre


if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

# Contador para el nombre de las fotos
cont = 0

# Leemos la camara
cap = cv2.VideoCapture(0)

# Creacion de un objeto que va almacenar la deteccion y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands() # Detector de manos

# Primer parametro, FALSE para que no haga la deteccion 24/7
# Solo hara deteccion cuando hay una confianza alta
# Segundo parametro, numero maximo de manos 2
# Tercer parametro, confianza minima de deteccion 50%
# Cuarto parametro, confianza minima de seguimiento 50%

# Metodo para dibujar las manos
dibujo = mp.solutions.drawing_utils  # Con este metodo se dibujan 21 puntos criticos de la mano

while (1):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista se almacena las coordenadas de los puntos
    # print(resultado.multi?hand?landmarks) su se quiere visualizar la deteccion

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:  # Busqueda de la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):  # Se obtiene la informacion de cada mano encontrada por el ID
                alto, ancho, c = frame.shape  # Extraemos el ancho y el alto de los fotogramas para multiplicarlos por la proporcion
                corx, cory = int(lm.x * ancho), int(lm.y * alto)  # Extraemos la ubicacion de cada punto que pertenece a la mano en coodenadas
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                punto_i1 = posiciones[4]
                punto_i2 = posiciones[20]
                punto_i3 = posiciones[12]
                punto_i4 = posiciones[0]
                punto_i5 = posiciones[9]
                x1, y1 = (punto_i5[1] - 100), (punto_i5[2] - 100) #Se obtiene el punto inicial y las longitudes
                ancho, alto = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC) #Redimensionamos las fotos
            cv2.imwrite(carpeta + "/Dedos_{}.jpg".format(cont), dedos_reg)
            cont = cont + 1

    cv2.imshow("Proyecto Final", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break
cap.release()
cv2.destroyAllWindows()
