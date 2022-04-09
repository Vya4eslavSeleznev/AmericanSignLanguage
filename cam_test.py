import torch
import joblib
import numpy as np
import cv2
import cnn_models


if __name__ == '__main__':
    lb = joblib.load('../outputs/lb.pkl')  # load label binarizer

    model = cnn_models.CustomCNN()
    model.load_state_dict(torch.load('../outputs/model.pth'))  # load model

    cap = cv2.VideoCapture(0)  # initialize video capture
    if not cap.isOpened():
        print('Failed to initialize video capture.')

    frame_width = int(cap.get(3))  # get frame width
    frame_height = int(cap.get(4))  # get frame height
    out = cv2.VideoWriter('../outputs/cam_test.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          60,
                          (frame_width, frame_height))  # define codec and create VideoWriter object

    while cap.isOpened() & (cv2.waitKey(27) & 0xFF != ord('q')):
        _, frame = cap.read()  # capture each frame of the video
        # get the hand area on the video capture screen
        cv2.rectangle(frame, (100, 100), (324, 324), (0, 0, 0), 2)  # draw rectangle on frame

        image = cv2.resize(frame[100:324, 100:324], (224, 224))  # get hand area
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0)

        outputs = model(image)
        _, predicts = torch.max(outputs.data, 1)

        cv2.putText(frame, lb.classes_[predicts], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imshow('image', frame)
        out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
