import cv2
import numpy as np


class ColorDetector:
    def __init__(self, color_definitions, color_name, video_source=0):
        """
        Initialize the ColorDetector.

        Parameters:
            color_definitions (dict): Dictionary of color names and their corresponding BGR values.
            color_name (str): The color to be detected.
            video_source (int): Video capture device index (default is 0 for the default camera).
        """
        self.color_definitions = color_definitions
        self.color_name = color_name
        self.video_source = video_source

    @staticmethod
    def optimize_color_range(color):
        """
        Convert BGR color to HSV and optimize the color range.

        Parameters:
            color (list): BGR color values.

        Returns:
            tuple: Lower and upper HSV limits.
        """
        color_np = np.uint8([[color]])
        hsv_color = cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)

        lower_limit = hsv_color[0][0][0] - 10, 100, 100
        upper_limit = hsv_color[0][0][0] + 10, 255, 255

        return np.array(lower_limit, dtype=np.uint8), np.array(upper_limit, dtype=np.uint8)

    def detect_color(self, frame):
        """
        Detect the specified color in the given frame and draw rectangles around the detected objects.

        Parameters:
            frame (numpy.ndarray): Input frame.

        Returns:
            numpy.ndarray: Frame with rectangles drawn around detected objects.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_limit, upper_limit = self.optimize_color_range(self.color_definitions[self.color_name])

        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        return frame

    def run_detection(self):
        """
        Run the color detection process using the specified color and video source.
        """
        cap = cv2.VideoCapture(self.video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_with_rectangles = self.detect_color(frame)

            cv2.imshow('frame', frame_with_rectangles)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    color_definitions = {
        'yellow': [0, 255, 255],
        'blue': [255, 0, 0],
        'red': [0, 0, 245],
        'orange': [0, 165, 255]
        # Add more colors as needed
    }

    color_name = 'blue'  # Change this to the desired color

    color_detector = ColorDetector(color_definitions, color_name)
    color_detector.run_detection()


if __name__ == "__main__":
    main()
