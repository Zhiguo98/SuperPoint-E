import cv2
import numpy as np
import matplotlib.pyplot as plt

# find feature points
# Replace with your image path
img_path = '/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00030.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Unable to load image from path:", img_path)
else:
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    img_with_keypoints = np.copy(img)


    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(img_with_keypoints, (x, y), 1, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("Interest Points Overlay")
    plt.show()
