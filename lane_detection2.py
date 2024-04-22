import cv2 as cv
import numpy as np
import math

def apply_canny_edge_detection(frame, upper_thresh=30, lower_thresh=10):
    grayscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv.Canny(blurred, lower_thresh, upper_thresh)
    
    ker_size = 5
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
    # cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    for lane in lanes_left + lanes_right:
        # x1, y1, x2, y2 = lane
        a = math.cos(lane[0][1])
        b = math.sin(lane[0][1])
        x0 = a * lane[0][0]
        y0 = b * lane[0][0]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(lane_visualization, pt1, pt2, (0, 255, 0), 5)
    return lane_visualization

def debug_draw(frame, lanes):
    lane_visualization = np.zeros_like(frame)
    # cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    for lane in lanes:
        # x1, y1, x2, y2 = lane
        a = math.cos(lane[0][1])
        b = math.sin(lane[0][1])
        x0 = a * lane[0][0]
        y0 = b * lane[0][0]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(lane_visualization, pt1, pt2, (0, 255, 0), 5)
    return lane_visualization


def verify_lines(lanes, side):
    c = 20
    AGC_verifed_lanes = []

    if side == 'l':
        lower_bound = 30 #135
        upper_bound = 60 #180
    elif side == 'r':
        lower_bound = 0
        upper_bound = 60

    lower_bound = math.radians(lower_bound)
    upper_bound = math.radians(upper_bound)


    for lane in lanes:
        # fit = np.polyfit((x1, x2), (y1, y2), 1)
        # slope = math.degrees(fit[0])

        if lower_bound <= lane[0][1] and lane[0][1] <= upper_bound:
            AGC_verifed_lanes.append(lane)
            # print('1')
    # exit()
    
    return AGC_verifed_lanes

def split_lanes(width, hough_lines):
    left_lanes = []
    right_lanes = []
    
    left_lower_b = math.radians(20)
    left_higher_b = math.radians(55)

    right_lower_b = math.radians(135)
    right_higher_b = math.radians(180)

    for detec_line in hough_lines:
        if left_lower_b <= detec_line[0][1] and detec_line[0][1] <= left_higher_b:
            left_lanes.append(detec_line)
        elif right_lower_b <= detec_line[0][1] and detec_line[0][1] <= right_higher_b:
            right_lanes.append(detec_line)

    return left_lanes, right_lanes

# Main execution block
video_capture = cv.VideoCapture(r"data\video\video3.mp4")
while video_capture.isOpened():
    successful_frame_read, frame = video_capture.read()
    if not successful_frame_read:
        break

    canny_edges = apply_canny_edge_detection(frame, lower_thresh=30)
    roi_frame = define_region_of_interest(canny_edges)
    hough_lines = cv.HoughLines(roi_frame, 2, np.pi / 180, 150, None, 0, 0)
    
    
    # Splitting the lanes
    dim = np.shape(frame)
    lanes_left, lanes_right = split_lanes(dim[1], hough_lines)

    # Adding Angle Geometric Constraint
    # lanes_left = verify_lines(lanes_left, "l")
    # lanes_right = verify_lines(lanes_right, "r")

    lane_lines_image = draw_lane_lines(frame, lanes_left, lanes_right)
    # left_lane_image = debug_draw(frame, lanes_left)
    # right_lane_image = debug_draw(frame, lanes_right)
    
    # dim = np.shape(canny_edges)
    # debug_frame = canny_edges.reshape(dim[0],dim[1],1)
    # debug_frame = lane_lines_image[:] + debug_frame
    
    combined_output = cv.addWeighted(frame, 0.9, lane_lines_image, 1, 1)   
    # combined_output = cv.addWeighted(debug_frame, 0.9, lane_lines_image, 1, 1)     
    # combined_output_left = cv.addWeighted(frame, 0.9, left_lane_image, 1, 1)
    # combined_output_right = cv.addWeighted(frame, 0.9, right_lane_image, 1, 1)

    cv.imshow("Lane Lines", combined_output)
    # cv.imshow("Left", combined_output_left)
    # cv.imshow("Right", combined_output_right)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
