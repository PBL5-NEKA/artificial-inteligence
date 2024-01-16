import tensorflow as tf
import cv2
from scripts import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing.image import img_to_array

#---------------Direcciones de modelos---------------
ModeloCNN2 = 'models/01FinalDatasetModel.h5'
ModelHaarCascade = 'models/haarcascade_frontalface_alt2.xml'
#---------------Leemos la red neuronal---------------
CNN2 = tf.keras.models.load_model(ModeloCNN2)
pesoscnn2 = CNN2.get_weights()
CNN2.set_weights(pesoscnn2)

#---------------Realizamos la VideoCaptura---------------
cap = cv2.VideoCapture(0)

#---------------Empieza nuestra While true---------------
while (cap.isOpened()):
    #Lectura de nuestra VideoCaptura
    ret, frame = cap.read()

    #Redimensionamos la imagen
    gray = cv2.resize(frame, (256,256), interpolation=cv2.INTER_CUBIC)

    # Normalizamos la imagen
    face_cascade = preprocessing.load_face_cascade(ModelHaarCascade)
    img, _ = preprocessing.apply_haar_cascade_on_image(gray, face_cascade)
    img = cv2.resize(img, (128, 128))
    img = np.array(img).astype(float) / 255.0

    #Convertimos la imagen en matriz
    img = np.expand_dims(img, axis=0)

    #Realizamos la prediccion
    prediccion = CNN2.predict(img)
    prediccion = prediccion[0][0]

    #Realizamos la clasificacion
    umbral = 0.65
    if prediccion >= umbral:
        prediccion2 = 1
    else:
        prediccion2 = 0

    #print(prediccion2)
    # Muestra la predicción en la ventana de la cámara
    #cv2.putText(frame, f"Predicción: {round(prediccion,3)}|{prediccion2}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Predicción: {prediccion2}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    cv2.imshow('Captura', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir del bucle
        break

cap.release()
cv2.destroyAllWindows()
