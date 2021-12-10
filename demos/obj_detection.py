import cv2
import nnio


def main():
    # Load model
    model = nnio.zoo.onnx.detection.SSDMobileNetV1()

    # Get preprocessing function
    preproc = model.get_preprocessing()
    
    # Open web camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        image_rgb = image[:,:,::-1] # to RGB

        # Pass to the neural network
        image_prepared = preproc(image_rgb)
        boxes = model(image_prepared)

        # Draw boxes
        for box in boxes:
            image = box.draw(image)

        # Display the resulting frame
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()













