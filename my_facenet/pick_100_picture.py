import os
import random
import shutil

# 指定包含图片的文件夹路径
images_directory = r'D:\Desktop\face_ai\data'

# 支持的图片文件扩展名列表
supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# 获取文件夹中所有图片文件的路径
image_files = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(images_directory)
    for f in filenames if any(f.lower().endswith(ext) for ext in supported_extensions)
]

# 确保图片文件的数量大于200，否则随机选择数量将等于文件夹中图片的总数
num_images_to_choose = min(100, len(image_files))

# 随机选择200张图片
randomly_chosen_images = random.sample(image_files, num_images_to_choose)

# 打印随机选择的图片文件路径
for image_path in randomly_chosen_images:
    print(image_path)

# 复制到另一个文件夹
destination_directory = './new_img'
for image_path in randomly_chosen_images:
    shutil.copy(image_path, destination_directory)
