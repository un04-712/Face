import os
from PIL import Image


def add_border_to_images(directory, output_size=(750, 750), border_color=(255, 255, 255)):
    """
    在目录中的所有图片添加白色边框，并调整大小为300x300。
    参数:
    directory (str): 包含图片的目录。
    output_size (tuple): 所需的输出大小（宽度，高度）。
    border_color (tuple): 边框颜色，采用RGB格式。
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 打开图片
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            # 计算边框大小
            current_size = img.size
            new_width = output_size[0]
            new_height = output_size[1]
            left = (new_width - current_size[0]) // 2
            top = (new_height - current_size[1]) // 2
            right = new_width - current_size[0] - left
            bottom = new_height - current_size[1] - top
            # 添加边框并调整大小
            img_with_border = Image.new("RGB", output_size, border_color)
            img_with_border.paste(img, (left, top))
            img_with_border.save(img_path)  # 保存图片，覆盖原图


add_border_to_images("./new_img")
