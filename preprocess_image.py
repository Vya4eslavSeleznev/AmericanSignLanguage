import os
import cv2
import random
import argparse

from imutils import paths
from tqdm import tqdm


if __name__ == '__main__':
    # парсер аргументов
    parser = argparse.ArgumentParser()
    # добавляем аргумент кол-ва изображений для каждой категории
    parser.add_argument('-n', '--num-images', default=1000, type=int,
                        help='number of images to preprocess for each category')
    # парсим аргументы
    args = vars(parser.parse_args())
    print(f"Preprocessing {args['num_images']} from each category...")

    # список путей до изображений для обучения
    image_paths = list(paths.list_images('../input/asl_alphabet_train/asl_alphabet_train'))
    dir_paths = os.listdir('../input/asl_alphabet_train/asl_alphabet_train')
    dir_paths.sort()
    root_path = '../input/asl_alphabet_train/asl_alphabet_train'

    # проходимся по каждой категории изображений
    for idx, dir_path in tqdm(enumerate(dir_paths), total=len(dir_paths)):
        # список
        all_images = os.listdir(f"{root_path}/{dir_path}")
        os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)
        # в каждой категории обрабатываем изображения в количестве, указанном в аргументе
        for i in range(args['num_images']):
            # генерируем случайный идентификатор в диапазоне от 0 до 2999
            rand_id = (random.randint(0, 2999))
            image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
            image = cv2.resize(image, (224, 224))
            # сохраняем обработанное изображение в соответствующей директории
            cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)

    print('DONE')
