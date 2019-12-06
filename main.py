import cv2
import argparse
from face_swap import swap_face

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src_path", required=True, help="source image path")
ap.add_argument("-d", "--dst_path", required=True, help="destination image path")
args = ap.parse_args()

img1 = cv2.imread(args.src_path)
img2 = cv2.imread(args.dst_path)

swapped = swap_face(img1, img2)

cv2.namedWindow("face swapped", cv2.WINDOW_NORMAL)
cv2.imshow("face swapped", swapped)
cv2.waitKey(0)
cv2.destroyAllWindows()