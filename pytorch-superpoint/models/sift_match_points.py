
import cv2
import matplotlib.pyplot as plt

def find_and_draw_matches(img1_path, img2_path):
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # convert it to the grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    
    sift = cv2.SIFT_create()

    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

   
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

   
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    img_matches[img_matches == 255] = 0

    
    plt.imshow(img_matches)
    plt.title("Matched Points")
    plt.show()

if __name__ == "__main__":
    # Replace with your image path
    img1_path = "/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00177.jpg"
    # Replace with your image path
    img2_path = "/Users/zhiguoma/Desktop/master/homework/image_understanding/dataset/EndoJPEG/hyperK_000/00178.jpg" # Replace with your image path

    find_and_draw_matches(img1_path, img2_path)
