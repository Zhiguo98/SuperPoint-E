import cv2
import numpy as np
import matplotlib.pyplot as plt

# find feature points
img_path = '/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00030.jpg'  # 替换为您的图像路径
img = cv2.imread(img_path)

if img is None:
    print("Unable to load image from path:", img_path)
else:
    # 如果图像加载成功，继续处理
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # 创建一个和输入图像大小相同的空图像
    img_with_keypoints = np.copy(img)

    # 手动绘制小的、实心的绿色点
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(img_with_keypoints, (x, y), 1, (0, 255, 0), -1)  # -1表示实心圆

    # 显示带有特征点的图像
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("Interest Points Overlay")
    plt.show()