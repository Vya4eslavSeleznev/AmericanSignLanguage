import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths


# скрипт для маппинга путей до изображений к лейблам
# изображения считываются с диска во время обучения


if __name__ == '__main__':
    # пути до предобработанных изображений
    image_paths = list(paths.list_images('../input/preprocessed_image'))

    # создаем DataFrame
    data = pd.DataFrame()

    labels = []
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        label = image_path.split(os.path.sep)[-2]

        # сохраняем отнасительный путь для маппинга изображения к таргету
        data.loc[i, 'image_path'] = image_path

        labels.append(label)

    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print(f"The first one hot encoded labels: {labels[0]}")
    print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
    print(f"Total instances: {len(labels)}")

    for i in range(len(labels)):
        index = np.argmax(labels[i])
        data.loc[i, 'target'] = int(index)

    data = data.sample(frac=1).reset_index(drop=True)  # shuffle the dataset

    data.to_csv('../input/data.csv', index=False)  # save as CSV file

    print('Saving the binarized labels as pickled file')
    joblib.dump(lb, '../outputs/lb.pkl')  # pickle the binarized labels

    print(data.head(10))
