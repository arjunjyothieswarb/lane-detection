import cv2 as cv
import numpy as np
import math

def apply_canny_edge_detection(frame, upper_thresh=30, lower_thresh=10):
    grayscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv.Canny(blurred, lower_thresh, upper_thresh)

    ker_size = 3
    kernel = np.ones((ker_size,ker_size))
    closed = cv.dilate(edges, kernel)
    
    return closed

def define_region_of_interest(frame):
    frame_height, frame_width = frame.shape[:2]
    top_width = frame_width // 3
    bottom_width = frame_width
    mid_height = 3 * frame_height // 4
    mask_vertices = np.array([
        [(frame_width // 2 - top_width // 2, mid_height),
         (frame_width // 2 + top_width // 2, mid_height),
         (frame_width, frame_height),
         (0, frame_height)]
    ])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, mask_vertices, 255)
    masked_frame = cv.bitwise_and(frame, mask)
    return masked_frame

def identify_lane_lines(frame, detected_lines):
    lanes_left = []
    lanes_right = []
    for line in detected_lines:
        x1, y1, x2, y2 = line.reshape(4)
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        if slope < 0:
            lanes_left.append([x1, y1, x2, y2])
        else:
            lanes_right.append([x1, y1, x2, y2])
    return lanes_left, lanes_right

def draw_lane_lines(frame, lanes_left, lanes_right):
    lane_visualization = np.zeros_like(frame)
    for lane in lanes_left + lanes_right:
        x1, y1, x2, y2 = lane
        cv.line(lane_visualization, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lane_visualization

def debug_draw(frame, lanes):
    lane_visualization = np.zeros_like(frame)
    for lane in lanes:
        x1, y1, x2, y2 = lane
        cv.line(lane_visualization, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lane_visualization


def verify_lines(lanes, side):
    c = 20
    AGC_verifed_lanes = []

    if side == 'l':
        lower_bound = -90
        upper_bound = -45 + c
    elif side == 'r':
        lower_bound = 45 - c
        upper_bound = 90

    for [x1, y1, x2, y2] in lanes:
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = math.degrees(fit[0])

        if lower_bound <= slope and slope <= upper_bound:
            AGC_verifed_lanes.append([x1, y1, x2, y2])
    
    return AGC_verifed_lanes

def split_lanes(width, hough_lines):
    left_lanes = []
    right_lanes = []

    for detec_line in hough_lines:
        x1, y1, x2, y2 = detec_line.reshape(4)        
        if (x1 + x2)/2 > width/2:
            right_lanes.append([x1, y1, x2, y2])
        else:
            left_lanes.append([x1, y1, x2, y2])

    return left_lanes, right_lanes

# Main execution block
video_capture = cv.VideoCapture(r"data\video\video3.mp4")
while video_capture.isOpened():
    successful_frame_read, frame = video_capture.read()
    if not successful_frame_read:
        break

    # Getting the edge image and defining roi
    canny_edges = apply_canny_edge_detection(frame)
    roi_frame = define_region_of_interest(canny_edges)

    # Getting the hough lines from the image
    hough_lines = cv.HoughLinesP(roi_frame, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=150)
    dim = np.shape(frame)
    
    # Splitting the left and right lanes
    lanes_left, lanes_right = split_lanes(dim[1], hough_lines)

    # Filtering the lines using AGC
    lanes_left = verify_lines(lanes_left, "l")
    lanes_right = verify_lines(lanes_right, "r")
    
    lane_lines_image = draw_lane_lines(frame, lanes_left, lanes_right)
    
    combined_output = cv.addWeighted(frame, 0.9, lane_lines_image, 1, 1)

    cv.imshow("Lane Lines", combined_output)

    # Debug block, pay no heed
    left_lane_image = debug_draw(frame, lanes_left)
    right_lane_image = debug_draw(frame, lanes_right)

    debug_line = lane_lines_image + canny_edges.reshape(dim[0],dim[1],1)

    combined_output_left = cv.addWeighted(frame, 0.9, left_lane_image, 1, 1)
    combined_output_right = cv.addWeighted(frame, 0.9, right_lane_image, 1, 1)

    cv.imshow("Left", combined_output_left)
    cv.imshow("Right", combined_output_right)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
