import cv2
import numpy as np
import os
import matplotlib as plt

def main():
    scale = 0.7

    files = os.listdir("D:\HOC\Thesis result\S2Looking\S2Looking\\train\Image1")

    for f in files:
        id = os.path.basename(f)[:-4]
        i1 = cv2.imread(f"D:\HOC\Thesis result\S2Looking\S2Looking\\train\Image1\\{id}.png")
        i2 = cv2.imread(f"D:\HOC\Thesis result\S2Looking\S2Looking\\train\Image2\\{id}.png")

        l1 = cv2.imread(f"D:\HOC\Thesis result\S2Looking\S2Looking\\train\label1\\{id}.png")
        l2 = cv2.imread(f"D:\HOC\Thesis result\S2Looking\S2Looking\\train\label2\\{id}.png")

        s1 = cv2.addWeighted(i1, 0.7, l1, 0.3, 0)
        s1 = cv2.resize(s1, (int(s1.shape[0]*scale), int(s1.shape[1]*scale)))

        s2 = cv2.addWeighted(i2, 0.7, l2, 0.3, 0)
        s2 = cv2.resize(s2, (int(s2.shape[0]*scale), int(s2.shape[1]*scale)))

        cv2.imshow("s1", s1)
        cv2.imshow("s2", s2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()