import cv2
import numpy as np
import nnio


def main():
    # Load model
    model = nnio.zoo.edgetpu.segmentation.DeepLabV3(device='CPU')

    # Get preprocessing function
    preproc = model.get_preprocessing()

    # Color map
    colormap = np.random.random([21, 3])
    colormap[0, :] = 0

    # Open web camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        w, h, _ = image.shape
        image_rgb = image[:,:,::-1] # to RGB

        # Pass to the neural network
        image_prepared = preproc(image_rgb)
        segmentation_map = model(image_prepared)

        # Parse output
        segmentation_image = np.zeros(list(segmentation_map.shape) + [3])
        for i in range(len(colormap)):
            segmentation_image[segmentation_map == i] = colormap[i]
        segmentation_image = cv2.resize(segmentation_image, (h, w))

        # Display the resulting frame
        cv2.imshow('image', image)
        cv2.imshow('segmentation_image', segmentation_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()













