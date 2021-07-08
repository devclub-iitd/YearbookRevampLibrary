import math
import cv2
import mediapipe as mp
import os
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    if 2 * a[1] < b[1] + c[1]:
        angle = 180 + angle

    return angle


# Used to crop in photo after rotation
def rotatedRectWithMaxArea(w, h, angle):
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:

        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:

        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(wr), int(hr)


def ShoulderStraighten(ImagePath,OutputPath = None):
    # mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Read the image
        img_initial = cv2.imread(ImagePath)

        # Process the image
        image = cv2.cvtColor(img_initial, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]


        if left_shoulder and right_shoulder:

            # Using nose to get which side up
            if nose:
                angle_out = calculate_angle([nose.x, nose.y], [left_shoulder.x, left_shoulder.y],
                                            [right_shoulder.x, right_shoulder.y])
            else:
                angle_out = calculate_angle([0, 0], [left_shoulder.x, left_shoulder.y],
                                            [right_shoulder.x, right_shoulder.y])

            rows, cols, ht = image.shape

            matrix = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle_out, 1)

            rows_final, cols_final = rotatedRectWithMaxArea(rows, cols, angle_out)

            output_image = cv2.warpAffine(image, matrix, (rows_final, cols_final), borderValue=(255, 255, 255))

            final_image =  output_image
        else:
            final_image = img_initial

        if OutputPath == None:
            return final_image
        else:
            if not cv2.imwrite(OutputPath, final_image):
                raise Exception("Could not write image")



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\Input")
    outputPath = os.path.join(BASE_DIR + "\\Output")

    for file in os.listdir(inputPath):
        output = ShoulderStraighten(inputPath + "\\" + file)
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), output):
            raise Exception("Could not write image")
