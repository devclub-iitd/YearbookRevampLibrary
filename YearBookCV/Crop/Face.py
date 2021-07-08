import cv2
import os
import numpy as np


def incircle(point, radius):
    if (point[0] - radius) ** 2 + (point[1] - radius) ** 2 <= radius ** 2:
        return True
    else:
        return False


def CropFace(img_initial, size,output_path = None) :

    final_img = []
    grayImg = cv2.cvtColor(img_initial, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)

    # print(faces)
    if len(faces) > 0:
        for face in faces:

            face_img_raw = img_initial[face[1]:face[1] + face[2], face[0]:face[0] + face[2]]

            face_img = cv2.cvtColor(face_img_raw, cv2.COLOR_BGR2BGRA)


            radius = int(face[2] / 2)

            for rows in range(face_img.shape[0]):
                for cols in range(face_img.shape[1]):
                    if not incircle([rows, cols], radius):
                        face_img[rows][cols][3] = 0

            face_img_final = cv2.resize(face_img, (size, size))

            final_img.append(face_img_final)
    else:
        blankimg = np.zeros((size, size), dtype=np.uint8)
        final_img.append( cv2.cvtColor(blankimg, cv2.COLOR_GRAY2BGR))

    if output_path == None:
        return final_img
    else:
        for img in final_img:
            if not cv2.imwrite(output_path, img):
                raise Exception("Could not write image")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\Input")
    outputPath = os.path.join(BASE_DIR + "\\Output")

    for file in os.listdir(inputPath):
        print(file)
        img_initial = cv2.imread(inputPath + "\\" + file)
        outImg = CropFace(img_initial, 30)
        outImg = outImg[0]
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), outImg):
            raise Exception("Could not write image")
