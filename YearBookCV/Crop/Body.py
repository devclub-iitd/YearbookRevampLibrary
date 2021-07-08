import cv2
from YearBookCV.Background.Remove import RemoveBackground
import mediapipe as mp
import os


# To Detect if in circle or not
def incircle(point, radius):
    if (point[0] - radius) ** 2 + (point[1] - radius) ** 2 <= radius ** 2:
        return True
    else:
        return False


def CropBody(img, size):
    rows, cols, c = img.shape
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True

        landmarks = results.pose_landmarks.landmark

        # Find location of nose, to center circle around it
        Nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        if Nose:
            RowCenter = int(Nose.y * rows)
            ColCenter = int(Nose.x * cols)
        else:
            RowCenter = int(rows/2)
            ColCenter = int(cols/2)

        #Calculates maximum possible radius size
        if cols < rows:
            radius = int(cols / 2)
            ColCenter = radius
            if RowCenter < radius:
                RowCenter = radius
            elif rows - RowCenter < radius:
                RowCenter = rows - radius
        else:
            radius = int(rows / 2)
            RowCenter = radius
            if ColCenter < radius:
                ColCenter = radius
            elif cols - ColCenter < radius:
                ColCenter = cols - radius

        ImageOut = RemoveBackground(img, (255, 255, 255))

        #Crops image
        face_img_raw = ImageOut[RowCenter - radius:RowCenter + radius, ColCenter - radius:ColCenter + radius]

        #Convert to include an alpha value
        face_img = cv2.cvtColor(face_img_raw, cv2.COLOR_BGR2BGRA)


        for rows in range(face_img.shape[0]):
            for cols in range(face_img.shape[1]):
                if not incircle([rows, cols], radius):
                    face_img[rows][cols][3] = 0

        face_img_final = cv2.resize(face_img, (size, size))
        return face_img_final


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\Input")
    outputPath = os.path.join(BASE_DIR + "\\Output")

    for file in os.listdir(inputPath):
        print(file)
        img_initial = cv2.imread(inputPath + "\\" + file)
        outImg = CropBody(img_initial, 30)
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), outImg):
            raise Exception("Could not write image")
