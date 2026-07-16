"""
人脸匹配引擎 —— 向量化批量计算
将原有的 O(n×m) 逐个余弦相似度计算优化为矩阵运算

关键改进:
1. 数据库特征预存为 numpy 矩阵，一次性计算所有相似度
2. 自动过滤零范数特征，避免除零
3. 可配置阈值，返回 Top-K 候选
"""
import pickle
import numpy as np


class FaceMatcher:
    """人脸特征匹配器"""

    def __init__(self, threshold=0.5):
        """
        threshold: 余弦相似度阈值，默认 0.5
        """
        self.threshold = threshold
        self._db_matrix = None      # [N, 512] 数据库特征矩阵
        self._db_names = []          # 对应的姓名列表
        self._dirty = True           # 是否需要重建矩阵

    def load_database(self, db_features):
        """
        从数据库特征列表构建匹配矩阵
        db_features: list of (name, feature_bytes)
        """
        features = []
        names = []
        for db_name, db_face_feature_bytes in db_features:
            if not db_face_feature_bytes:
                continue
            feat = pickle.loads(db_face_feature_bytes)
            feat = np.asarray(feat).squeeze()
            if feat.shape != (512,):
                continue
            # 确保已 L2 归一化
            norm = np.linalg.norm(feat)
            if norm < 1e-8:
                continue
            features.append(feat / norm)
            names.append(db_name)

        if features:
            self._db_matrix = np.stack(features)  # [N, 512]
        else:
            self._db_matrix = np.empty((0, 512))
        self._db_names = names
        self._dirty = False

    def match(self, query_features):
        """
        批量匹配多个查询特征
        query_features: list of numpy arrays, 每个 (512,)
        返回: list of dict [{'name', 'similarity', 'index'}]
        """
        if self._db_matrix is None or self._db_matrix.shape[0] == 0:
            return [{'name': '未匹配', 'similarity': 0.0, 'index': -1}] * len(query_features)

        if not query_features:
            return []

        # 构建查询矩阵 [M, 512]，同时做 L2 归一化
        query_list = []
        for feat in query_features:
            feat = np.asarray(feat).squeeze()
            if feat.shape != (512,):
                continue
            norm = np.linalg.norm(feat)
            if norm < 1e-8:
                continue
            query_list.append(feat / norm)

        if not query_list:
            return [{'name': '未匹配', 'similarity': 0.0, 'index': -1}] * len(query_features)

        query_matrix = np.stack(query_list)  # [M, 512]

        # ---- 核心: 一次矩阵乘法替代双重循环 ----
        # similarity[i,j] = cos(query_i, db_j)
        similarity_matrix = query_matrix @ self._db_matrix.T  # [M, N]

        # 每行取最大值
        best_indices = np.argmax(similarity_matrix, axis=1)   # [M]
        best_scores = np.max(similarity_matrix, axis=1)        # [M]

        results = []
        for i in range(len(query_features)):
            score = float(best_scores[i]) if i < len(best_scores) else 0.0
            idx = int(best_indices[i]) if i < len(best_indices) else -1
            if score >= self.threshold and idx >= 0:
                results.append({
                    'name': self._db_names[idx],
                    'similarity': round(score, 4),
                    'index': idx
                })
            else:
                results.append({
                    'name': '未匹配',
                    'similarity': round(score, 4),
                    'index': -1
                })
        return results

    def match_single(self, query_feature):
        """匹配单个特征"""
        return self.match([query_feature])[0]