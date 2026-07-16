import os.path
import datetime
from PyQt5.QtCore import QByteArray
from PyQt5.QtSql import QSqlQuery, QSqlDatabase


class MysqlLite:
    def __init__(self):
        """连接数据库"""
        try:
            self.database = QSqlDatabase.addDatabase('QSQLITE')
            self.database.setDatabaseName("student.db")
            if self.database.open():
                self.check_and_create_table()
                self.query = QSqlQuery()
            else:
                raise Exception("数据库打开失败")
        except Exception as e:
            print("无法连接数据库", e)

    def operation_sql(self, sql):
        """执行SQL语句"""
        self.query = QSqlQuery(self.database)
        return self.query.exec_(sql)

    def selectData(self, sql):
        """查询数据"""
        self.operation_sql(sql)
        result = []
        while self.query.next():
            row = []
            for i in range(self.query.record().count()):
                row.append(self.query.value(i))
            result.append(row)
        return result

    def check_and_create_table(self):
        """检查表是否存在，不存在则创建"""
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

    def search_table(self, name):
        """按姓名查询"""
        sql = f"select * from student_info where 姓名 like '%{name}%'"
        return self.selectData(sql)

    def add_info(self, name, age, sex, number, time=None, photo_path=None, face_feature=None):
        """添加信息到数据库"""
        # 自动生成录入时间
        if time is None:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 检查学号是否重复
        check_sql = f"select COUNT(*) from student_info where 学号 = {number}"
        if not self.query.exec_(check_sql):
            print("学号检查查询执行失败")
            return False
        self.query.next()
        if self.query.value(0) > 0:
            print(f"学号 {number} 已存在")
            return False

        # 读取照片数据
        photo_data = None
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, 'rb') as f:
                photo_data = f.read()
        elif isinstance(photo_path, bytes):  # 直接传入二进制数据
            photo_data = photo_path

        # 插入数据
        sql = """
            INSERT INTO student_info(姓名,年龄,性别,学号,录入时间,照片,人脸特征)
            VALUES(?,?,?,?,?,?,?)
        """
        self.query.prepare(sql)
        self.query.addBindValue(name)
        self.query.addBindValue(age)
        self.query.addBindValue(sex)
        self.query.addBindValue(number)
        self.query.addBindValue(time)
        self.query.addBindValue(QByteArray(photo_data) if photo_data else None)
        self.query.addBindValue(QByteArray(face_feature) if face_feature else None)

        if self.query.exec_():
            print(f"学生 {name} 信息添加成功！")
            return True
        else:
            print(f"添加失败: {self.query.lastError().text()}")
            return False

    def delete_info(self, name):
        """删除指定姓名的记录"""
        sql = f"DELETE FROM student_info WHERE 姓名 LIKE '%{name}%'"
        return self.operation_sql(sql)

    def close(self):
        """关闭数据库连接"""
        if self.database.isOpen():
            self.database.close()