import cv2
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation



def RemoveBackground(img_path,color,output_path = None):
    img = cv2.imread(img_path)
    segment = SelfiSegmentation(0)
    img_out = segment.removeBG(img,color)
    if output_path == None:
        return img_out
    else:
        if not cv2.imwrite(output_path, img_out):
            raise Exception("Could not write image")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\Input")
    outputPath = os.path.join(BASE_DIR + "\\Output")
    for file in os.listdir(inputPath):
        img_out = RemoveBackground(inputPath + "\\" + file, (255, 255, 255))
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), img_out):
            raise Exception("Could not write image")


