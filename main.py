"""
人脸识别系统 

"""
import pickle
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox
from ultralytics import YOLO
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

from my_ui.face_ui import Ui_Form
from utils.ThreadClass import CameraThread, MysqliteThread
from utils.pubilc import preprocess_face_img_unified, extract_face_feature_unified
from utils.model_manager import ModelManager, preprocess_face_batch, extract_face_features_batch
from utils.face_matcher import FaceMatcher


# 模型和项目路径
YOLO_MODEL_PATH = r"D:\Desktop\face_ai\best.pt"


# ---- 异常钩子 ----
def custom_excepthook(exc_type, exc_value, exc_tb):
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("=== PyQt 程序错误 ===")
    print(error_msg)
    QtWidgets.QMessageBox.critical(
        None,
        "程序运行错误",
        f"发生未处理的错误：\n\n{error_msg}",
        QtWidgets.QMessageBox.Ok
    )

sys.excepthook = custom_excepthook


class FaceRecognitionSystem(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.photo_path = None
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # ---- 模型单例初始化（全局一次，不在线程中重复加载）----
        mgr = ModelManager()
        mgr.init_models(YOLO_MODEL_PATH, device='cpu')

        # 槽函数初始化
        self.bind_buttons()
        # 初始化数据库
        self.db_init()
        # 初始化人脸特征变量
        self.save_face_feature = None
        # 保存照片标志位
        self.save_pic_flag = False
        # 保护 save_pic_flag 的线程锁
        self.save_pic_lock = threading.Lock()

        # 人脸匹配器 —— 批量向量化计算
        self.face_matcher = FaceMatcher(threshold=0.5)

        # 摄像头线程（不在构造时启动）
        self.camera_thread = None

        # 初始化数据库管理表格
        self.init_manage_table()

    def bind_buttons(self):
        self.ui.pushButton_14.clicked.connect(self.switch_to_input_page)
        self.ui.pushButton_15.clicked.connect(self.switch_to_recognize_page)
        self.ui.pushButton_16.clicked.connect(self.switch_to_manage_page)
        self.ui.pushButton_17.clicked.connect(self.close_app)

        # 人脸录入按钮
        self.ui.pushButton_5.clicked.connect(self.open_folder_btn)
        self.ui.pushButton_8.clicked.connect(self.open_cap_btn)
        self.ui.pushButton_7.clicked.connect(self.save_img_btn_slot)
        self.ui.pushButton_6.clicked.connect(self.close_cap_btn)
        self.ui.pushButton_9.clicked.connect(self.save_person_info_btn)

        # 人脸识别按钮
        self.ui.pushButton_22.clicked.connect(self.open_folder_recognize_face_btn)
        self.ui.pushButton_23.clicked.connect(self.recognize_open_cap_btn)
        self.ui.pushButton_25.clicked.connect(self.recognize_close_cap_btn)

        # 数据库管理按钮
        self.ui.pushButton_2.clicked.connect(self.query_table)
        self.ui.pushButton_3.clicked.connect(self.delete_selected_row)
        self.ui.pushButton_4.clicked.connect(self.refresh_table)

    # ---- 界面切换 ----
    def switch_to_input_page(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def switch_to_recognize_page(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def switch_to_manage_page(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def close_app(self):
        self.close()

    def closeEvent(self, a0):
        result = QtWidgets.QMessageBox.question(
            self, "提示", "确定要退出吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if result == QtWidgets.QMessageBox.Yes:
            # 清理摄像头
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.running = False
                self.camera_thread.wait(1000)
            a0.accept()
        else:
            a0.ignore()

    # ---- 数据库初始化 ----
    def db_init(self):
        self.my_sqlite = MysqliteThread()
        self.my_sqlite.sql_face_feature_signal.connect(self.find_db_result)
        self.my_sqlite.query_result_signal.connect(self.fill_table)
        self.my_sqlite.start()

    # ---- 人脸录入 ----

    def open_folder_btn(self):
        self.save_pic_path = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择图片", "../face_ui", "(*.jpg *.png *.jpeg *.bmp *.gif)"
        )[0][0]
        print(self.save_pic_path)
        if self.save_pic_path:
            self.label_2_original, self.label_3_original, self.face_boxes, self.face_feature_infos = \
                self.detect_face_by_yolo(self.save_pic_path)
            self.ui.lineEdit.setText(self.save_pic_path)
            self.show_label_2(self.label_2_original)

            if len(self.face_boxes) == 0:
                QMessageBox.warning(self, "提示", "未检测到人脸")
            elif len(self.face_boxes) == 1:
                self.show_label_3(self.label_3_original)
            else:
                QMessageBox.warning(self, "提示",
                    f"检测到{len(self.face_boxes)}张人脸，请重新上传图片")
        else:
            print("没有选择图片文件")

    def open_cap_btn(self):
        if self.camera_thread and self.camera_thread.isRunning():
            QMessageBox.warning(self, "警告", "摄像头已打开，请勿重复启动！")
            return
        self.start_camera_thread()

    def close_cap_btn(self):
        if self.camera_thread:
            self.camera_thread.running = False
            self.camera_thread.wait(1000)
            self.camera_thread.send_result_img.disconnect(self.show_label_2)
            self.camera_thread.send_original_frame.disconnect(self.save_img_btn)
            self.camera_thread.deleteLater()
            self.camera_thread = None
        self.ui.label_2.clear()

    def start_camera_thread(self):
        # skip_frames=2: 每 2 帧检测一次，大幅降低 CPU
        self.camera_thread = CameraThread(YOLO_MODEL_PATH, skip_frames=2)
        self.camera_thread.send_result_img.connect(self.show_label_2)
        self.camera_thread.send_original_frame.connect(self.save_img_btn)
        self.camera_thread.start()

    def save_img_btn_slot(self):
        """拍照按钮点击槽函数"""
        with self.save_pic_lock:
            if self.save_pic_flag:
                QMessageBox.warning(self, "提示", "正在处理拍照，请稍候！")
                return
            self.save_pic_flag = True
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.running = False

    def save_img_btn(self, frame):
        """摄像头帧回调 - 处理拍照逻辑"""
        with self.save_pic_lock:
            if not self.save_pic_flag:
                return
            self.save_pic_flag = False

        try:
            converted_image = frame.convertToFormat(QImage.Format_RGB32)
            converted_image = converted_image.scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.save_pic_path = r"./saved_picture/save_pic.jpg"
            save_success = converted_image.save(self.save_pic_path, "jpg", 95)
            if not save_success:
                QMessageBox.warning(self, "提示", "图片保存失败！请检查路径权限")
                if self.camera_thread:
                    self.camera_thread.running = True
                return

            label_2, label_3, boxes, features = self.detect_face_by_yolo(self.save_pic_path)
            self.label_2_original, self.label_3_original = label_2, label_3
            self.face_boxes, self.face_feature_infos = boxes, features

            if len(self.face_boxes) == 0:
                QMessageBox.warning(self, "提示", "未检测到人脸")
            elif len(self.face_boxes) == 1:
                self.show_label_3(self.label_3_original)
                self.save_face_feature = self.face_feature_infos[0]
            else:
                QMessageBox.warning(self, "提示",
                    f"检测到{len(self.face_boxes)}张人脸，请重新拍照")
                self.ui.label_2.clear()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"拍照处理失败：{str(e)}")
            print(f"拍照异常详情：{traceback.format_exc()}")
        finally:
            if self.camera_thread and not self.camera_thread.isFinished():
                self.camera_thread.running = True

    def save_person_info_btn(self):
        """保存个人信息到数据库"""
        self.name = self.ui.lineEdit_3.text()
        self.age = self.ui.lineEdit_4.text()
        self.studentID = self.ui.lineEdit_2.text()
        self.sex = "男" if self.ui.radioButton.isChecked() else \
                   "女" if self.ui.radioButton_2.isChecked() else None

        if not self.name or not self.studentID:
            QMessageBox.warning(self, "提示", "姓名和学号不能为空")
            return

        if not self.age.isdigit():
            QMessageBox.warning(self, "提示", "年龄必须是数字")
            return

        with open(self.save_pic_path, 'rb') as f:
            photo_data = f.read()

        if self.save_face_feature is not None:
            self.save_face_feature = self.save_face_feature.squeeze()
            if len(self.save_face_feature) != 512:
                QMessageBox.warning(self, "提示", "人脸特征维度错误，保存失败")
                return

        self.save_face_feature_bytes = pickle.dumps(self.save_face_feature)

        self.my_sqlite.q_deal_sql_cmd.put({
            "cmd": self.my_sqlite.INSERT_NEW,
            "content": {
                "name": self.name,
                "age": self.age,
                "sex": self.sex,
                "studentID": self.studentID,
                "time": None,
                "photo_data": QByteArray(photo_data),
                "face_feature": QByteArray(self.save_face_feature_bytes)
            }
        })

        # 清空输入
        self.ui.lineEdit_3.clear()
        self.ui.lineEdit_4.clear()
        self.ui.lineEdit_2.clear()
        self.ui.lineEdit.clear()
        self.ui.label_3.clear()

    # ---- 人脸检测（集成批量预处理）----

    def detect_face_by_yolo(self, file_path):
        """YOLO 人脸检测 + 批量 FaceNet 特征提取"""
        face_boxes = []
        face_feature_infos = []

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "提示", "图片读取失败！")
            return None, None, [], []

        mgr = ModelManager()
        yolo_model = mgr.get_yolo()
        facenet_model = mgr.get_facenet()

        results = yolo_model(image, conf=0.45)
        boxes = results[0].boxes
        face_img = None
        face_imgs = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # 边界裁剪
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_face = image[y1:y2, x1:x2].copy()
            face_imgs.append(crop_face)
            face_boxes.append([x1, y1, x2, y2])
            # 在图像上画框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- 批量提取特征 ----
        if face_imgs:
            face_tensor_batch = preprocess_face_batch(face_imgs)
            if face_tensor_batch is not None:
                features = extract_face_features_batch(face_tensor_batch, facenet_model)
                face_feature_infos = [f for f in features]

        face_img = face_imgs[0] if face_imgs else None

        # 转换为 QImage
        h, w, ch = image.shape
        label_2_original = QImage(image.data, w, h, ch * w, QImage.Format_BGR888).copy()

        label_3_original = None
        if face_img is not None:
            h_f, w_f, ch_f = face_img.shape
            label_3_original = QImage(face_img.data, w_f, h_f, ch_f * w_f, QImage.Format_BGR888).copy()

        return label_2_original, label_3_original, face_boxes, face_feature_infos

    # ---- 图像显示 ----

    def show_label_2(self, label_2_original):
        pixmap = QPixmap.fromImage(label_2_original).scaled(
            self.ui.label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.ui.label_2.setPixmap(pixmap)
        self.ui.label_2.setAlignment(Qt.AlignCenter)

    def show_label_3(self, label_3_original):
        pixmap = QPixmap.fromImage(label_3_original).scaled(
            self.ui.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.ui.label_3.setPixmap(pixmap)
        self.ui.label_3.setAlignment(Qt.AlignCenter)

    def show_label_9(self, label_9_original):
        pixmap = QPixmap.fromImage(label_9_original).scaled(
            self.ui.label_9.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.ui.label_9.setPixmap(pixmap)
        self.ui.label_9.setAlignment(Qt.AlignCenter)

    # ---- 人脸识别 ----

    def open_folder_recognize_face_btn(self):
        self.recognize_path, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择图片", "../", "(*.jpg *.png *.jpeg *.bmp *.gif)"
        )
        self.face_infos = []
        if self.recognize_path:
            self.label_2_original, self.label_3_original, self.face_boxes, self.face_feature_infos = \
                self.detect_face_by_yolo(self.recognize_path[0])

            self.ui.lineEdit_6.setText(self.recognize_path[0])
            self.show_label_9(self.label_2_original)

            for i in range(len(self.face_boxes)):
                face_box = self.face_boxes[i]
                new_face_feature = self.face_feature_infos[i]
                self.face_infos.append([face_box, new_face_feature])

        self.my_sqlite.q_deal_sql_cmd.put({"cmd": self.my_sqlite.SELECT_FACE_FEATURE})

    # ---- 核心优化：向量化人脸匹配 ----

    def find_db_result(self, sql_data_dict):
        """接收数据库特征，执行向量化匹配"""
        print("开始识别人脸匹配")
        all_match_results = []

        if self.my_sqlite.SELECT_FACE_FEATURE not in sql_data_dict:
            return

        db_features = sql_data_dict[self.my_sqlite.SELECT_FACE_FEATURE]

        # 使用 FaceMatcher 加载数据库特征
        self.face_matcher.load_database(db_features)

        if not hasattr(self, 'face_infos') or len(self.face_infos) == 0:
            self.ui.textBrowser.setText("无有效人脸特征待匹配")
            return

        # 收集有效查询特征
        query_features = []
        valid_indices = []
        for i, face_data in enumerate(self.face_infos):
            face_box, feat = face_data
            feat = np.asarray(feat).squeeze()
            if feat.shape == (512,) and np.linalg.norm(feat) > 1e-8:
                query_features.append(feat)
                valid_indices.append(i)

        if not query_features:
            self.ui.textBrowser.setText("未检测到有效人脸特征")
            return

        # ---- 批量向量化匹配（替代逐个循环）----
        match_results = self.face_matcher.match(query_features)

        # 构建结果映射回原始索引
        results_map = {}
        for vi, mr in zip(valid_indices, match_results):
            results_map[vi] = mr

        for i, face_data in enumerate(self.face_infos):
            face_box = face_data[0]
            if i in results_map:
                mr = results_map[i]
                all_match_results.append({
                    "box": face_box,
                    "name": mr['name'],
                    "similarity": mr['similarity']
                })
            else:
                all_match_results.append({
                    "box": face_box,
                    "name": "未匹配",
                    "similarity": 0.0
                })

        # 展示结果
        self.ui.textBrowser.clear()
        if not all_match_results:
            self.ui.textBrowser.setText("未检测到有效人脸特征")
            return

        result_text = "人脸匹配结果：\n"
        for idx, res in enumerate(all_match_results, 1):
            result_text += f"\n第{idx}个人脸："
            result_text += f"\n- 位置：{res['box']}"
            result_text += f"\n- 匹配姓名：{res['name']}"
            result_text += f"\n- 相似度：{res['similarity']}\n"
        self.ui.textBrowser.setText(result_text)

    # ---- 摄像头人脸识别 ----

    def recognize_open_cap_btn(self):
        print("打开人脸识别摄像头")
        self.camera_thread = CameraThread(YOLO_MODEL_PATH, skip_frames=2)
        self.camera_thread.send_result_img.connect(self.show_label_9)
        self.camera_thread.send_face_feature_box.connect(self.get_face_feature_box)
        self.camera_thread.start()

    def recognize_close_cap_btn(self):
        print("关闭人脸识别摄像头")
        if self.camera_thread:
            self.camera_thread.running = False
            self.camera_thread.wait(1000)
            self.camera_thread.send_result_img.disconnect(self.show_label_9)
            self.camera_thread.send_face_feature_box.disconnect(self.get_face_feature_box)
            self.camera_thread.deleteLater()
            self.camera_thread = None
        self.ui.label_9.clear()
        self.ui.textBrowser.clear()

    def get_face_feature_box(self, face_boxes, face_features):
        """接收摄像头人脸框和特征，触发数据库匹配"""
        self.face_infos = []
        if len(face_boxes) == 0 or len(face_features) == 0:
            self.ui.textBrowser.setText("未检测到人脸")
            return

        for box, feat in zip(face_boxes, face_features):
            if feat is not None:
                feat = np.asarray(feat).squeeze()
                if feat.shape == (512,):
                    self.face_infos.append([box, feat])

        if self.face_infos:
            self.ui.textBrowser.setText("正在匹配人脸...")
            self.my_sqlite.q_deal_sql_cmd.put({"cmd": self.my_sqlite.SELECT_FACE_FEATURE})
        else:
            self.ui.textBrowser.setText("检测到人脸但特征提取失败")

    # ---- 数据库管理表格 ----

    def init_manage_table(self):
        self.ui.tableWidget.setColumnCount(8)
        self.ui.tableWidget.setHorizontalHeaderLabels([
            "ID", "姓名", "年龄", "性别", "学号", "录入时间", "照片", "人脸特征"
        ])
        self.ui.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ui.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ui.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.ui.tableWidget.verticalHeader().setVisible(False)

    def fill_table(self, data):
        self.ui.tableWidget.setRowCount(0)
        if not data:
            return
        for row_idx, row_data in enumerate(data):
            self.ui.tableWidget.insertRow(row_idx)
            for col_idx, cell_data in enumerate(row_data):
                if col_idx == 6 or col_idx == 7:
                    cell_text = "存在" if cell_data else "不存在"
                else:
                    cell_text = str(cell_data) if cell_data is not None else ""
                item = QtWidgets.QTableWidgetItem(cell_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.ui.tableWidget.setItem(row_idx, col_idx, item)

    def refresh_table(self):
        self.my_sqlite.q_deal_sql_cmd.put({
            "cmd": self.my_sqlite.SELECT_ALL,
            "content": None
        })

    def query_table(self):
        keyword = self.ui.lineEdit_5.text().strip()
        if not keyword:
            QMessageBox.warning(self, "提示", "请输入查询的姓名关键词！")
            return
        self.my_sqlite.q_deal_sql_cmd.put({
            "cmd": self.my_sqlite.SELECT_BY_KEYWORD,
            "content": keyword
        })

    def delete_selected_row(self):
        selected_items = self.ui.tableWidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "提示", "请先选中要删除的行！")
            return
        row_idx = selected_items[0].row()
        id_item = self.ui.tableWidget.item(row_idx, 0)
        if not id_item:
            QMessageBox.warning(self, "提示", "未找到该行的ID信息！")
            return
        id_to_delete = int(id_item.text())

        reply = QMessageBox.question(
            self, "确认删除",
            f"是否确定删除ID为{id_to_delete}的这条数据？\n删除后无法恢复！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.my_sqlite.q_deal_sql_cmd.put({
            "cmd": self.my_sqlite.DELETE_BY_ID,
            "content": id_to_delete
        })
        self.refresh_table()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.setWindowTitle("人脸识别系统")
    window.show()
    sys.exit(app.exec_())