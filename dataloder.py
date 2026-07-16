

import os


# 创建目录
os.makedirs('./dataset',exist_ok=True)

os.makedirs('./dataset/images',exist_ok=True)
os.makedirs('./dataset/images/train',exist_ok=True)
os.makedirs('./dataset/images/valid',exist_ok=True)
os.makedirs('./dataset/images/test',exist_ok=True)
os.makedirs('./dataset/labels',exist_ok=True)
os.makedirs('./dataset/labels/train',exist_ok=True)
os.makedirs('./dataset/labels/valid',exist_ok=True)
os.makedirs('./dataset/labels/test',exist_ok=True)
print('dataset目录结构创建成功')


# 获取data目录中文件内容
files = os.listdir('./data3')
import random
# 打乱列表里面的元素
random.shuffle(files)
print('打乱之后的数据',files)


import shutil

images = []

for file in files:
    if file[-3:]=='jpg':
        images.append(file)

print(images)
# 获取图片数量
images_count =  len(images)
print(images_count)

# 获取训练集图片的数量
num_train = images_count * 0.9
num_valid = images_count * 0.1
num_test = images_count - num_train - num_valid
print(num_train,num_valid,num_test)

count = 0
for image in images:
    # 拷贝训练集
    if count < num_train:
        shutil.copy('./data3/'+image,'./dataset/images/train/')
        shutil.copy('./data3/' + image.replace('jpg','txt'), './dataset/labels/train/')
    # 拷贝验证集
    elif count < num_valid+num_train:
        shutil.copy('./data3/' + image, './dataset/images/valid/')
        shutil.copy('./data3/' + image.replace('jpg', 'txt'), './dataset/labels/valid/')
    # 拷贝测试集
    else:
        shutil.copy('./data3/' + image, './dataset/images/test/')
        shutil.copy('./data3/' + image.replace('jpg', 'txt'), './dataset/labels/test/')

    print(f'{image}已写入')
    count+=1
