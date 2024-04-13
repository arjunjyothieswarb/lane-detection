"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
def main(argv):
 
#  default_file = "C:\\Users\\ajbbh\\OneDrive\\Pictures\\angular_z.png"
#  filename = argv[0] if len(argv) > 0 else default_file
#  # Loads an image
#  src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
#  # Check if image is loaded fine
#  if src is None:
#     print ('Error opening image!')
#     print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
#     return -1
 
    # vid = cv.VideoCapture(0)
    vid = cv.VideoCapture("../../data/video1.mp4")

    thresh_min_l = 60 * np.pi/180
    thresh_max_l = 80 * np.pi/180

    # thresh_max_r = np.pi - thresh_min_l
    # thresh_min_r = np.pi - thresh_max_l

    thresh_max_r = 100 * np.pi/180
    thresh_min_r = 90 * np.pi/180

    # thresh_max = (90 * np.pi/180) + thresh_min

    while(True):
        ret, src = vid.read()
        if not ret:
            break
        frame_dim = np.shape(src)
        bound = int(frame_dim[0]/2)
        src = src[bound:frame_dim[0]][:][:]
        # print(np.shape(src))
        # break
        dst = cv.Canny(src, 50, 200, None, 3)
        
        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        
        lines = cv.HoughLines(dst, 1,5 * np.pi / 180, 180, None, 0, 0)
        # print(lines)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                if not ((theta > thresh_min_l and theta < thresh_max_l) or (theta > thresh_min_r and theta < thresh_max_r)):
                    continue
                
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        
        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        # cv.waitKey()
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    return 0
 
if __name__ == "__main__":
 main(sys.argv[1:])