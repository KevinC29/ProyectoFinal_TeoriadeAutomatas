# Creacion de Modelo y Entrenamiento
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

datos_entrenamiento = '/Users/danixasolano/Download/KEVYN/Proyecto/Fotos/Entrenamiento'
datos_validacion = '/Users/danixasolano/Download/KEVYN/Proyecto/Fotos/Validacion'

#Parametros
iteraciones = 20 #Numero de iteraciones par ajustar nuestro modelo
altura, longitud = 200,200 #Tamaño de las imagenes de entrenamiento
batch_size = 1 #Numero de imagenes que se va enviar
pasos = 300/1 #Numero de veces que se va a procesar la informacion en cada iteracion
pasos_validacion = 300/1 #Luego de cada iteracion se valida lo anterior
filtroconv1 = 32 
filtroconv2 = 64   #Numero de filtros que se va aplicar a cada convolucion, para captar más informacion
filtroconv3 = 128 
tam_filtro1 = (4,4)
tam_filtro2 = (3,3)   #Tamaños de los filtros 1,2,3
tam_filtro3 = (2,2)
tam_pool = (2,2) #Tamaño del filtro en max pooling
clases = 5 #5 Vocales 
lr = 0.0005 #ajustes de la red neuronal para acercarse a una solucion optima

#Pre-Procesamiento de las imagenes
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255, #Pasar los pixeles de 0 a 255 | 0 a 1
    shear_range = 0.3, #Generar las imagenes inclinadas para el entrenamiento
    zoom_range = 0.3, #Genera imagenes con zoom para el entrenamiento
    horizontal_flip = True #Invierte las imagenes para el entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno =  preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento, #Toma las fotos que ya se encuentran previamente alacenadas
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical' #Clasificacion categorica = por clases
)

imagen_validacion =  preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

#Creacion de la red neuronal convolucional (CNN)
cnn = Sequential() #Red neuronal secuencial
#Agregar filtros para la imagen profunda pequeña
cnn.add(Convolution2D(filtroconv1, tam_filtro1, padding = 'same', input_shape = (altura,longitud,3), activation = 'relu')) #Se agrega la primera capa
cnn.add(MaxPooling2D(pool_size = tam_pool)) #Sirve para la abstraccion de caracteristicas
cnn.add(Convolution2D(filtroconv2, tam_filtro2, padding = 'same', activation = 'relu')) #Se agrega la segunda capa
cnn.add(MaxPooling2D(pool_size = tam_pool))
cnn.add(Convolution2D(filtroconv3, tam_filtro3, padding = 'same', activation = 'relu')) #Se agrega la tercera capa
cnn.add(MaxPooling2D(pool_size = tam_pool))

#Convertir imagen profunda a una plana para tener una dimension con toda la informacion obtenida
cnn.add(Flatten()) #Aplanar la imagen
cnn.add(Dense(640,activation='relu')) #Asignamos 640 neuronas
cnn.add(Dropout(0.5)) # Se agrega el 50% de las neuronas en la funcion anterior para no sobreajustar la red
cnn.add(Dense(clases, activation='softmax')) #Ultima capa en la que muestra la probabilidad de que sea alguna de las posiciones de las manos

#Agregar parametros para optimizar el modelo
optimizar = Adam(learning_rate = lr)
cnn.compile(loss= 'categorical_crossentropy', optimizer = optimizar, metrics = ['accuracy'])

#Entrenamiento de la red
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs = iteraciones, validation_data = imagen_validacion, validation_steps = pasos_validacion)

#Guardar el modelo
cnn.save('ModeloVocales.h5')
cnn.save_weights('pesosVocales.h5')