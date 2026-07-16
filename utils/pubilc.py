import cv2
import torch


def preprocess_face_img_unified(face_img):
    """统一的人脸预处理逻辑：BGR输入 → RGB → 160x160 → 归一化张量"""
    if face_img is None or face_img.size == 0:
        return None
    # BGR转RGB（仅转一次，避免重复）
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # 强制调整为160x160（FaceNet要求）
    face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
    # 转换为Tensor并归一化（0-1）
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0
    # 添加批次维度 [1, 3, 160, 160]
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor


def extract_face_feature_unified(face_tensor, facenet_model):
    """统一的人脸特征提取逻辑"""
    if face_tensor is None:
        return None
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
        # L2归一化（避免特征幅值影响相似度）
        l2_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        normalized_embedding = embedding / (l2_norm + 1e-8)  # 加小值避免除零
    # 转为numpy并压缩维度 (1,512) → (512,)
    return normalized_embedding.squeeze().cpu().numpy()