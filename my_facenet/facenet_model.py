import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from ultralytics import YOLO
import cv2

def getFacePos(img:str):
    """
    :param img:
    :return:返回人脸图像四个坐标点
    """

    model = YOLO(r"D:\Desktop\face_ai\myYolo\runs\detect\train6\weights\best.pt")
    results = model(img)
    boxes = results[0].boxes
    x0, y0, x1, y1 = map(int,boxes.xyxy[0].tolist())
    return x0 ,y0, x1, y1

def facenet(img:str,pos:tuple):
    """
    将人脸区域通过facenet转换成512维向量，返回出来
    :param img: 图片路径
    :param pos: 人脸区域位置
    :return: 512维向量，以numpy数组返回
    """
    image = cv2.imread(img)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = pos

    face_rgb = image_rgb[y0:y1, x0:x1]
    cv2.imshow("face_rgb",face_rgb)
    cv2.waitKey(0)
    face_pil = Image.fromarray(face_rgb)

    face_resize = face_pil.resize((160, 160))

    face_tensor = (torch.tensor(np.array(face_resize)).permute(2, 0, 1) / 255.0).unsqueeze(0)

    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    embedding = resnet(face_tensor)

    return embedding


if __name__ == "__main__":
    img = r"D:\Desktop\face_ai\dataset\images\test\img.png"
    pos = getFacePos(img)

    vec = facenet(img,pos)

    print(vec)
