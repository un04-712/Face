"""
优化后的摄像头线程
- 跳帧机制：每 skip_frames 帧才提取一次特征，减少 CPU 负担
- 特征缓存：同一张人脸短时间内不重新提取特征
- 模型单例：通过 ModelManager 共享 YOLO 和 FaceNet
- 内存安全：深拷贝帧数据，避免 QImage 野指针
"""
import time
import queue
import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtWidgets import QMessageBox
from .model_manager import ModelManager, preprocess_face_batch, extract_face_features_batch


class CameraThread(QThread):
    send_result_img = pyqtSignal(QImage)
    send_original_frame = pyqtSignal(QImage)
    send_face_feature_box = pyqtSignal(list, list)

    def __init__(self, yolo_model_path, skip_frames=2):
        """
        skip_frames: 跳帧间隔。1=每帧检测, 2=隔1帧检测, 3=隔2帧检测。
        推荐值 2~3，大幅降低 CPU 占用同时保持流畅
        """
        super().__init__()
        self.running = True
        self.skip_frames = skip_frames
        self.frame_count = 0

        # 从单例获取模型（不再重复加载）
        mgr = ModelManager()
        self.model = mgr.get_yolo()
        self.facenet_model = mgr.get_facenet()

        self.face_boxes = []
        self.face_features = []

        # 特征缓存：记录上一帧的特征，避免频繁重提取
        self._last_features = []
        self._last_boxes = []
        self._cache_ttl = 0.3  # 缓存有效期（秒），超过则强制刷新
        self._last_detect_time = 0

    def run(self):
        self.cap = cv2.VideoCapture(0)
        # 设置摄像头参数优化
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 降低摄像头分辨率以加速处理（可选）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # ---- 跳帧逻辑 ----
            # skip_frames=2 意味着每 2 帧检测一次，大幅降低 CPU
            do_detect = (self.frame_count % self.skip_frames == 0)

            # 检查缓存是否过期
            now = time.time()
            cache_valid = (now - self._last_detect_time) < self._cache_ttl

            if do_detect:
                # 深拷贝帧用于绘制（避免在原始帧上画框后影响后续处理）
                display_frame = frame.copy()

                # YOLO 检测（关闭日志减少输出）
                results = self.model.predict(frame, verbose=False, conf=0.45)
                boxes = results[0].boxes

                self.face_boxes.clear()
                self.face_features.clear()

                if boxes is not None and len(boxes) > 0:
                    # 收集所有人脸裁剪图
                    face_imgs = []
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # 边界裁剪防止越界
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        face_img = frame[y1:y2, x1:x2].copy()
                        face_imgs.append(face_img)
                        self.face_boxes.append([x1, y1, x2, y2])

                        # 在显示帧上画框
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ---- 批量提取特征（关键优化）----
                    if face_imgs:
                        face_tensor_batch = preprocess_face_batch(face_imgs)
                        if face_tensor_batch is not None:
                            features = extract_face_features_batch(face_tensor_batch, self.facenet_model)
                            self.face_features = [f for f in features]

                    # 更新缓存
                    self._last_features = self.face_features.copy()
                    self._last_boxes = self.face_boxes.copy()
                    self._last_detect_time = now
                else:
                    self._last_features = []
                    self._last_boxes = []
                    self._last_detect_time = now
            else:
                # 使用缓存特征
                self.face_boxes = self._last_boxes.copy()
                self.face_features = self._last_features.copy()
                display_frame = frame.copy()

            # ---- 发送信号 ----
            # 显示帧：先转 RGB 再构造 QImage
            rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            # 创建独立的 QImage（bytesPerLine 必须正确）
            rgb_img_qt = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.send_result_img.emit(rgb_img_qt)

            # 原始帧（用于拍照）
            # 同样需要 .copy() 避免野指针
            original_bgr = frame.copy()
            h_o, w_o, ch_o = original_bgr.shape
            original_qt = QImage(original_bgr.data, w_o, h_o, w_o * 3, QImage.Format_BGR888).copy()
            self.send_original_frame.emit(original_qt)

            # 人脸框和特征
            self.send_face_feature_box.emit(
                [b.copy() if hasattr(b, 'copy') else list(b) for b in self.face_boxes],
                [f.copy() if hasattr(f, 'copy') else f for f in self.face_features]
            )

            cv2.waitKey(1)

        self.cap.release()

    def stop(self):
        self.running = False


class MysqliteThread(QThread):
    INSERT_NEW = 1
    SELECT_BY_KEYWORD = 2
    SELECT_FACE_FEATURE = 3
    UPDATE_SORT = 4
    SELECT_ALL = 5
    DELETE_BY_ID = 6

    query_result_signal = pyqtSignal(list)
    sql_face_feature_signal = pyqtSignal(dict)
    update_sort_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.q_deal_sql_cmd = queue.Queue(5)
        try:
            self.database = QSqlDatabase.addDatabase('QSQLITE')
            self.database.setDatabaseName("./db/student.db")
            if self.database.open():
                self.check_and_create_table()
                self.query = QSqlQuery()
            else:
                raise Exception("数据库打开失败")
        except Exception as e:
            print("无法连接数据库", e)

    def check_and_create_table(self):
        check_query = QSqlQuery(self.database)
        check_query.exec("SELECT name FROM sqlite_master WHERE type='table' AND name='student_info';")
        if not check_query.next():
            print("正在创建表...")
            create_sql = """
                CREATE TABLE student_info(
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    姓名 TEXT NOT NULL,
                    年龄 INTEGER,
                    性别 TEXT,
                    学号 INTEGER UNIQUE,
                    录入时间 TIMESTAMP,
                    照片 BLOB,
                    人脸特征 BLOB
                )
            """
            if check_query.exec(create_sql):
                print("学生信息表创建成功！")
                return True
            else:
                print(f"表创建失败: {check_query.lastError().text()}")
                return False
        else:
            print("学生信息表已存在")
            return True

    def run(self):
        while True:
            q_data = self.q_deal_sql_cmd.get()
            cmd = q_data["cmd"]
            if cmd == self.INSERT_NEW:
                content = q_data["content"]
                self.insert_info(content)
            elif cmd == self.SELECT_BY_KEYWORD:
                content = q_data["content"]
                for name in content:
                    sql = f"select * from student_info where 姓名 like '%{name}%'"
                    self.query.exec_(sql)
                    self.query.bindValue(":keyword", name)
                    if not self.query.exec_(sql):
                        print(f"{self.query.lastError().text()}")
                        self.query_result_signal.emit([])
                    else:
                        result = []
                        while self.query.next():
                            row = []
                            for i in range(self.query.record().count()):
                                row.append(self.query.value(i))
                            result.append(row)
                        self.query_result_signal.emit(result)
            elif cmd == self.SELECT_FACE_FEATURE:
                sql = "select * from student_info"
                self.query.prepare(sql)
                self.query.exec_()
                data = []
                while self.query.next():
                    db_name = self.query.value(1)
                    db_feature = self.query.value(7)
                    data.append((db_name, db_feature))
                self.sql_face_feature_signal.emit({self.SELECT_FACE_FEATURE: data})

            elif cmd == self.UPDATE_SORT:
                self.database.transaction()
                try:
                    sql_temp_table = "create table temp_table as select * from student_info"
                    self.query.prepare(sql_temp_table)
                    if not self.query.exec():
                        raise Exception(f"创建临时表失败：{self.query.lastError().text()}")
                    sql_clera_original_table = "DELETE from student_info"
                    self.query.prepare(sql_clera_original_table)
                    if not self.query.exec():
                        raise Exception(f"删除原表失败：{self.query.lastError().text()}")
                    sql_insert_data = """
                            INSERT INTO student_info(姓名,年龄,性别,学号,录入时间,照片,人脸特征)
                            SELECT 姓名,年龄,性别,学号,录入时间,照片,人脸特征
                            from temp_table
                    """
                    self.query.prepare(sql_insert_data)
                    if not self.query.exec():
                        raise Exception(f"插入数据失败：{self.query.lastError().text()}")
                    self.database.commit()
                except Exception as e:
                    print("执行排序数据操作失败", e)
                    self.database.rollback()

            elif cmd == self.SELECT_ALL:
                sql = "select * from student_info"
                self.query.prepare(sql)
                if self.query.exec_():
                    result = []
                    while self.query.next():
                        row = []
                        for i in range(self.query.record().count()):
                            row.append(self.query.value(i))
                        result.append(row)
                    self.query_result_signal.emit(result)
                else:
                    self.query_result_signal.emit([])

            elif cmd == self.DELETE_BY_ID:
                id_to_delete = q_data["content"]
                self.database.transaction()
                try:
                    sql = "DELETE FROM student_info WHERE ID = :id"
                    self.query.prepare(sql)
                    self.query.bindValue(":id", id_to_delete)
                    if self.query.exec_():
                        self.database.commit()
                        print(f"ID为{id_to_delete}的数据删除成功")
                    else:
                        raise Exception(self.query.lastError().text())
                except Exception as e:
                    self.database.rollback()
                    print(f"删除失败：{e}")
                    QMessageBox.critical(None, "删除失败", f"数据库删除错误：{str(e)}")

    def insert_info(self, content):
        if content["time"] is None:
            content["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        check_sql = f"select COUNT(*) from student_info where 学号 = {content['studentID']}"
        if not self.query.exec_(check_sql):
            print("学号检查查询执行失败")
            return False
        self.query.next()
        if self.query.value(0) > 0:
            print(f"学号 {content['studentID']} 已存在")
            return False

        sql = """
            INSERT INTO student_info(姓名,年龄,性别,学号,录入时间,照片,人脸特征)
            VALUES(:name, :age, :sex, :studentID, :time, :photo_data, :face_feature)
        """
        self.query.prepare(sql)
        self.query.exec_()
        self.query.addBindValue(content["name"])
        self.query.addBindValue(content["age"])
        self.query.addBindValue(content["sex"])
        self.query.addBindValue(content["studentID"])
        self.query.addBindValue(content["time"])
        self.query.addBindValue(content["photo_data"])
        self.query.addBindValue(content["face_feature"])

        if self.query.exec_():
            print(f"学生 {content['name']} 信息添加成功！")
            return True
        else:
            print(f"添加失败: {self.query.lastError().text()}")
            return False