## Importar os. CUDA_VISIBLE_DEVICES con "1" es para usar la CPU si no se tiene GPU
## se usa paquetes de Python: openCV y DeepFace
## Hay que descargar https://github.com/kipr/opencv/tree/master/data/haarcascades


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
from deepface import DeepFace

# Para el reconocimiento facial frontal
rostroCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)  #opcional a la webcam se puede poner el archivo de una imagen.
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede abrir la Webcam")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection = False)  # fuerzo deteccion para que no se apague

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = rostroCascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in rostros:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    emocion=''
    if result['dominant_emotion']=='neutral':
        emocion='neutral'
    elif result['dominant_emotion']=='angry':
        emocion='enojado'    
    elif result['dominant_emotion']=='disgust':
        emocion='desagrado'
    elif result['dominant_emotion']=='fear':
        emocion='temor'
    elif result['dominant_emotion']=='happy':
        emocion='alegria'
    elif result['dominant_emotion']=='sad':
        emocion='tristeza'
    elif result['dominant_emotion']=='surprise':
        emocion='sorpresa'
    
    cv2.putText(frame,
                emocion,
                (50, 50),
                font, 2,
                (0,255,255),
                2,
                cv2.LINE_4)
    cv2.imshow("original video", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):  #Presionar la  tecla 'q' para salir
        break

cap.release()
cv2.destroyAllWindows()