import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

def image_detection(path_x):
    image = cv2.imread(path_x)

    model = models.get('yolo_nas_s', num_classes=1, checkpoint_path='C:/Users/Oliveira/PycharmProjects/Yolo/dia5Crack.pth')  # Gado

    classNames = ['Rachadura']

    # Realizar a detecção
    result = model.predict(image, conf=0.32)
    bbox_xyxys = result.prediction.bboxes_xyxy
    confidences = result.prediction.confidence
    labels = result.prediction.labels

    # Desenhar as detecções na imagem
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ajustar o tamanho vertical da bounding box
        vertical_padding = 20  # Ajuste conforme necessário
        y1 += vertical_padding
        y2 -= vertical_padding

        classname = int(cls)
        class_name = classNames[classname]
        conf = math.ceil((confidence * 100)) / 100
        label = f'{class_name} {conf}'  # Adicionando espaço entre o nome da classe e a confiança
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # Obtendo o tamanho da caixa de texto
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.rectangle(image, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), (0, 128, 0), -1)  # Desenhando um retângulo para a caixa de texto
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Adicionando texto à imagem

    # Redimensionar a imagem para uma altura fixa
    max_height = 400  # Altura máxima desejada
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = max_height
        image = cv2.resize(image, (new_width, new_height))

    # Mostrar a imagem em uma janela
    cv2.imshow("Detecção de Objetos", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_detection('C:/Users/Oliveira/PycharmProjects/Yolo/1foto - Copia.jpg')
