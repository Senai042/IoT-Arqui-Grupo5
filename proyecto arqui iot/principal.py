import cv2
import numpy as np

# Cargar el clasificador de detección de humanos SSD
human_net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 
                                     'MobileNetSSD_deploy.caffemodel')

# Cargar el video
cap = cv2.VideoCapture('grabacion personas.mp4')
cap.set(cv2.CAP_PROP_FPS, 60)
detections = {}
person_id = 0
cantidadMaxima = 0
cantidaActual = 0
cantidadTotal = 0
cont = 0

while True:
    # Leer un frame del video
    ret, frame = cap.read()

    if not ret:
        break

    # Obtener las dimensiones del frame
    (h, w) = frame.shape[:2]

    # Calcular el ancho y alto del área central
    center_width = int(w * 0.75)
    center_height = int(h * 0.75)

    # Calcular los límites del área central
    center_x_start = int((w - center_width) / 2)
    center_x_end = center_x_start + center_width
    center_y_start = int((h - center_height) / 2)
    center_y_end = center_y_start + center_height

    # Obtener el área central del frame
    center_frame = frame[center_y_start:center_y_end, center_x_start:center_x_end]

    # Convertir a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(center_frame, (800, 800)), 0.007843, (300, 300), 127.5)

    # Pasar la imagen a través de la red y obtener las detecciones
    human_net.setInput(blob)
    detections = human_net.forward()

    # Diccionario para almacenar las detecciones actuales
    detections_dict = {}

    # Dibujar los cuadros de detección en el área central
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Si la confianza es mayor a 0.5, se considera una detección válida
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([center_width, center_height, center_width, center_height])
            (x, y, x2, y2) = box.astype("int")

            # Ajustar las coordenadas a las del frame completo
            x += center_x_start
            y += center_y_start
            x2 += center_x_start
            y2 += center_y_start

            match_id = None

            for id, detection in detections_dict.items():
                if (x < detection[2] and x2 > detection[0] and
                        y < detection[3] and y2 > detection[1]):
                    match_id = id
                    break

            # Asignar identificador a la detección actual
            if match_id is not None:
                detections_dict[match_id] = (x, y, x2, y2)

                # Calcular las coordenadas para el cuadro superior
                top_y = y - int((y2 - y) / 2)
                height = int((y2 - y) / 2)

                cv2.rectangle(frame, (x, top_y), (x2, y2), (0, 255, 0), 2)
            else:
                person_id += 1
                detections_dict[person_id] = (x, y, x2, y2)

                # Ajustar el tamaño del cuadro de detección
                padding = int((y2 - y) / 4)
                x -= padding
                y -= padding
                x2 += padding
                y2 += padding

                # Asegurarse de que las coordenadas no estén fuera de los límites de la imagen
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Calcular las coordenadas para el cuadro superior
                top_y = y - int((y2 - y) / 2)
                height = int((y2 - y) / 2)

                cv2.rectangle(frame, (x, top_y), (x2, y2), (0, 0, 255), 2)

    # Mostrar la cantidad de personas detectadas
    if cont == 0:
        cantidadMaxima = 0
        if cantidadMaxima < len(detections_dict):
            cantidadMaxima = len(detections_dict)
            cont = cantidadMaxima

    print("cantidad maxima: {}".format(cantidadMaxima))
    cantidaActual = len(detections_dict)
    print("cantidad Actual: {}".format(cantidaActual))

    if cantidaActual < cantidadMaxima:
        if cantidaActual < cont:
            cantidadTotal += 1
            cont -= 1

    print("cantidad total: {}".format(cantidadTotal))
    cv2.putText(frame, f"Personas detectadas: {len(detections_dict)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.putText(frame, f"Total: {cantidadTotal}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    # Mostrar la imagen
    cv2.imshow('frame', frame)

    # Esperar por la tecla 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
