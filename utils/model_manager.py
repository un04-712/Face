"""
模型管理器 - 单例模式
YOLO 和 FaceNet 模型全局只加载一次，消除重复加载带来的内存浪费和启动延迟
"""
import threading
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class ModelManager:
    """模型单例管理器，线程安全"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def init_models(self, yolo_path, device='cpu'):
        """初始化所有模型（仅首次调用生效）"""
        if self._initialized:
            return
        print(f"[ModelManager] 加载 YOLO 模型: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        print(f"[ModelManager] 加载 FaceNet 模型, 设备: {device}")
        self.facenet_model = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
        self.device = device
        self._initialized = True
        print("[ModelManager] 模型加载完成")

    def get_yolo(self):
        return self.yolo_model

    def get_facenet(self):
        return self.facenet_model


# ---- 预处理优化 ----

def preprocess_face_batch(face_imgs, target_size=(160, 160)):
    """
    批量预处理人脸图像，替代逐张处理
    输入: face_imgs - list of BGR numpy arrays
    输出: 批量化后的张量 [B, 3, 160, 160]，已完成归一化
    """
    if not face_imgs:
        return None

    batch = []
    for face_img in face_imgs:
        if face_img is None or face_img.size == 0:
            continue
        # BGR → RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # resize 到 160x160
        face_resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)
        # 转为 tensor [3, 160, 160]，归一化到 [0, 1]
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
        batch.append(face_tensor)

    if not batch:
        return None
    return torch.stack(batch)  # [B, 3, 160, 160]


def extract_face_features_batch(face_tensor_batch, facenet_model):
    """
    批量提取人脸特征
    输入: face_tensor_batch [B, 3, 160, 160]
    输出: numpy array [B, 512]，已完成 L2 归一化
    """
    if face_tensor_batch is None:
        return np.array([])
    with torch.no_grad():
        embeddings = facenet_model(face_tensor_batch)  # [B, 512]
        # 批量 L2 归一化
        l2_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized = embeddings / (l2_norm + 1e-8)
    return normalized.cpu().numpy()


# ---- 保留向后兼容的单张接口 ----

def preprocess_face_img_unified(face_img):
    """保留原有单张接口，内部复用批处理"""
    result = preprocess_face_batch([face_img])
    return result if result is None else result[0:1]


def extract_face_feature_unified(face_tensor, facenet_model):
    """保留原有单张接口，内部复用批处理"""
    if face_tensor is None:
        return None
    features = extract_face_features_batch(face_tensor, facenet_model)
    return features[0] if len(features) > 0 else None