# python test.py --img A_test.jpg

import torch
import joblib
import numpy as np
import cv2
import argparse
import albumentations
import time
import cnn_models

parser = argparse.ArgumentParser()  # construct argument parser
parser.add_argument('-i', '--img', default='A_test.jpg', type=str,
                    help='path for the image to test on')  # parse arguments
args = vars(parser.parse_args())

lb = joblib.load('../outputs/lb.pkl')  # load label binarizer

aug = albumentations.Compose([albumentations.Resize(224, 224, always_apply=True)])

model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth'))

image = cv2.imread(f"../input/asl_alphabet_test/asl_alphabet_test/{args['img']}")
image_copy = image.copy()
 
image = aug(image=np.array(image))['image']
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float)
image = image.unsqueeze(0)
 
start = time.time()
outputs = model(image)
_, predicts = torch.max(outputs.data, 1)
print(f"Predicted output: {lb.classes_[predicts]}")
end = time.time()
print(f"Prediction time: {(end-start):.3f} seconds")
 
cv2.putText(image_copy, lb.classes_[predicts], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
cv2.imshow('image', image_copy)
cv2.imwrite(f"../outputs/{args['img']}", image_copy)
cv2.waitKey(0)
