import cv2


class DetectionBox:
    def __init__(
        self,
        x_min,
        y_min,
        x_max,
        y_max,
        label=None,
        score=1.0,
    ):
        '''
        
        :parameter x_min: ``float`` in range ``[0, 1]``.
            Relative x (width) coordinate of top-left corner.
        :parameter y_min: ``float`` in range ``[0, 1]``.
            Relative y (height) coordinate of top-left corner.
        :parameter x_max: ``float`` in range ``[0, 1]``.
            Relative x (width) coordinate of bottom-right corner.
        :parameter y_max: ``float`` in range ``[0, 1]``.
            Relative y (height) coordinate of bottom-right corner.
        :parameter label: ``str`` or ``None``.
            Class label of the detected object.
        :parameter score: ``float``.
            Detection score
        '''
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.label = label
        self.score = score

    def draw(
        self,
        image,
        color=(255,0,0),
        stroke_width=2,
        text_color=(255,0,0),
        text_width=2,
    ):
        '''
        Draws the detection box on an image

        :parameter image: numpy array.
        :parameter color: RGB color of the frame.
        :parameter stroke_width: boldness of the frame.
        :parameter text_color: RGB color of the text.
        :parameter text_width: boldness of the text.

        :return: Image with the box drawn on it.
        '''
        # Box corners:
        start_point = (
            int(image.shape[1] * self.x_min),
            int(image.shape[0] * self.y_min),
        )
        end_point = (
            int(image.shape[1] * self.x_max),
            int(image.shape[0] * self.y_max),
        )
        # Draw rectangle
        # pylint: disable=no-member
        image = cv2.rectangle(
            image,
            start_point,
            end_point,
            color, stroke_width)
        # Draw text
        if self.label is not None:
            # pylint: disable=no-member
            image = cv2.putText(
                image,
                self.label,
                (start_point[0], start_point[1] + 20 + stroke_width),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, text_color, text_width, cv2.LINE_AA
            )
        return image

    def __str__(self):
        template = 'nnio.DetectionBox(x_min={}, y_min={}, x_max={}, y_max={}, label="{}", score={})'
        s = template.format(
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.label,
            self.score
        )
        return s
