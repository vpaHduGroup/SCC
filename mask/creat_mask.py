import os
from PIL import Image


def create_black_image(filename, width, height):
    image = Image.new('RGB', (width, height), (0, 0, 0))
    image.save(filename)


def check_and_create_images(folder_a, folder_b):
    for file_name in os.listdir(folder_b):
        file_path_a = os.path.join(folder_a, file_name)
        file_path_b = os.path.join(folder_b, file_name)

        if os.path.isfile(file_path_b):  # ÎÄ¼þbÖÐ´æÔÚÍ¬ÃûÍ¼Æ¬
            if not os.path.isfile(file_path_a):  # ÎÄ¼þaÖÐ²»´æÔÚÍ¬ÃûÍ¼Æ¬
                img = Image.open(file_path_b)
                width, height = img.size
                create_black_image(file_path_a, width, height)
                print(f"Created black image: {file_path_a}")
        else:  # ÎÄ¼þbÖÐ²»´æÔÚÍ¬ÃûÍ¼Æ¬
            print(f"No corresponding image in Folder B: {file_path_a}")


# ²âÊÔ
check_and_create_images("/home/zhoufangtao/Datasets/screen_image/train_and_valid_synthetic/mask/croped_test_new", "/home/zhoufangtao/Datasets/screen_image/croped_test_new")
