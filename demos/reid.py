import sys
import time
import os
import argparse
import numpy as np
import cv2

import nnio


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()

    # Load object detector
    objdet = nnio.zoo.onnx.detection.SSDMobileNetV1()
    objdet_preproc = objdet.get_preprocessing()

    # Create database
    database = nnio.utils.HumanDataBase()

    # Load reid model
    reid = nnio.zoo.onnx.reid.OSNet()
    reid_preproc = reid.get_preprocessing()

    # Open web camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        image_rgb = image[:,:,::-1].copy() # to RGB

        # Pass to the neural network
        image_prepared = objdet_preproc(image_rgb)
        boxes = objdet(image_prepared)

        # Draw boxes
        for box in boxes:
            if box.label == 'person':
                # Crop this person image
                h, w, _ = image_rgb.shape
                crop = image_rgb[
                    int(h * box.y_min): int(h * box.y_max),
                    int(w * box.x_min): int(w * box.x_max)
                ]
                crop_prepared = reid_preproc(crop)
                vec = reid(crop_prepared)
                # Find this person in the database
                key = database.find_closest(vec)
                box.label = 'person ' + key
            image = box.draw(image)

        database.optimize()

        # Display the resulting frame
        cv2.imshow('image', image)
        # cv2.imshow('crop', crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

