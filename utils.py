import math
import cv2
from shapely.geometry import Polygon
import numpy as np
import os
import yaml

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_within_field_polygon(player_bbox, field_bbox):
    rect_p = Polygon([(player_bbox[0], player_bbox[1]), (player_bbox[2], player_bbox[1]), (player_bbox[2], player_bbox[3]), (player_bbox[0], player_bbox[3])])
    field_p = Polygon(field_bbox)
    return rect_p.intersects(field_p)
        

def is_within_field(player_bbox, field_bbox, court_center_y, court_center_x):
    x1_player, y1_player, x2_player, y2_player, conf = player_bbox
    x1_field, y1_field, x2_field, y2_field = field_bbox

    # Check if player is below or above the net line even if little bit
    if y1_player < court_center_y:
        # Player is above the net line
        y_check = y2_player
    else:
        # Player is below the net line
        y_check = y1_player

    # Check if player is on the left or right side of the court
    if x1_player < court_center_x:
        # Player is on the left side of the court
        return (x1_field <= x2_player <= x2_field and y1_field <= y_check <= y2_field)
    else:
        # Player is on the right side of the court
        return (x1_field <= x1_player <= x2_field and y1_field <= y_check <= y2_field)

def frame_to_timestamp(frame_number, FPS):
    total_seconds = frame_number / FPS
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def intersect_line_segment(p1, p2, q1, q2):
    """
    Check if the line segment (p1, p2) intersects the line segment (q1, q2).
    """
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    return False
    
def time_stamp_to_frames(time_stamp, FPS):
    # Split the time stamp into hours, minutes, and seconds
    hh, mm, ss = map(int, time_stamp.split(':'))
    # Calculate the total number of seconds
    total_seconds = hh * 3600 + mm * 60 + ss
    # Calculate the frame number
    frame_number = total_seconds * FPS
    return frame_number

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter_area

def non_max_suppression(bboxes, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression to eliminate redundant overlapping bounding boxes.
    
    Parameters:
        bboxes (list): List of bounding boxes with confidence [(x1, y1, x2, y2, conf), ...].
        iou_threshold (float): Intersection over Union (IoU) threshold for suppression.

    Returns:
        list: Filtered list of bounding boxes after applying NMS with integer coordinates.
    """
    if len(bboxes) == 0:
        return []
    
    # Convert bounding boxes to a NumPy array
    bboxes = np.array(bboxes)
    
    # Extract coordinates and confidence scores
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    
    # Compute the area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort the bounding boxes by their confidence scores in descending order
    order = scores.argsort()[::-1]
    
    # List to hold the indices of the selected bounding boxes
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute the intersection areas
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep bounding boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    # Convert coordinates to integers before returning
    result = []
    for i in keep:
        x1, y1, x2, y2, score = bboxes[i]
        result.append([int(x1), int(y1), int(x2), int(y2), score])
    
    return result

from pprint import pprint
def map_bbox_doubles(new_bboxes, homography_bbox1, homography_bbox2,view, prev_trackers, frame_number=None):
    MAX_DISTANCE_FOR_MAPPING = 150
    if len(new_bboxes) == 0:
        if prev_trackers=={}:
            return {view+'0':homography_bbox1, view+'1':homography_bbox2}
        else:
            return {view+'0':prev_trackers[view+'0'], view+'1':prev_trackers[view+'1']}
            
    elif len(new_bboxes) ==1:
        bbox = new_bboxes[0][:-1]
        bbox_bottom_centre = ((bbox[0] + bbox[2]) // 2, bbox[3])
        if view == 'top':
            dist_1= calculate_distance(bbox_bottom_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[3]])
            dist_2= calculate_distance(bbox_bottom_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[3]])
        else: #TO BE CHANGED
            dist_1= calculate_distance(bbox_bottom_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[1]])
            dist_2= calculate_distance(bbox_bottom_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[1]])
        if  dist_1<dist_2:
            if dist_1>MAX_DISTANCE_FOR_MAPPING:
                if prev_trackers=={}:
                    return {view+'0':homography_bbox1, view+'1':homography_bbox2}
                else:
                    return {view+'0':prev_trackers[view+'0'], view+'1':prev_trackers[view+'1']}
            else:
                if prev_trackers=={}:
                    return {view+'0':bbox, view+'1':homography_bbox2}
                else:
                    return {view+'0':bbox, view+'1':prev_trackers[view+'1']}
        else:
            if dist_2>MAX_DISTANCE_FOR_MAPPING:
                if prev_trackers=={}:
                    return {view+'0':homography_bbox1, view+'1':homography_bbox2}
                else:
                    return {view+'0':prev_trackers[view+'0'], view+'1':prev_trackers[view+'1']}
            else:
                if prev_trackers=={}:
                    return {view+'0':homography_bbox1, view+'1':bbox}
                else:
                    return {view+'0':prev_trackers[view+'0'], view+'1':bbox}
    else:
        if len(new_bboxes) > 2:
            print("More than 2 players detected", len(new_bboxes), view)
            
        assert len(new_bboxes) == 2
        bbox1, bbox2 = new_bboxes
        bbox1 = bbox1[:-1]
        bbox2 = bbox2[:-1]
        bbox1_centre = ((bbox1[0] + bbox1[2]) // 2, bbox1[3])
        bbox2_centre = ((bbox2[0] + bbox2[2]) // 2, bbox2[3])
        if view == 'top':
            dist_1_1 = calculate_distance(bbox1_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[3]])
            dist_1_2 = calculate_distance(bbox1_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[3]])
            dist_2_1 = calculate_distance(bbox2_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[3]])
            dist_2_2 = calculate_distance(bbox2_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[3]])
        else:
            dist_1_1 = calculate_distance(bbox1_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[1]])
            dist_1_2 = calculate_distance(bbox1_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[1]])
            dist_2_1 = calculate_distance(bbox2_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[1]])
            dist_2_2 = calculate_distance(bbox2_centre, [(homography_bbox2[0]+homography_bbox2[2])//2, homography_bbox2[1]])
        
        distances = [dist_1_1, dist_1_2, dist_2_1, dist_2_2]
        min_index = 0
        min_dist = 10000000000
        for i,dist in enumerate(distances):
            if dist < min_dist:
                min_dist = dist
                min_index = i
            
            
        if dist_1_1 + dist_2_2 < dist_1_2 + dist_2_1:
            
            current_mapping= {view+'0':bbox1, view+'1':bbox2}
            
            
            if dist_1_1 >MAX_DISTANCE_FOR_MAPPING or dist_2_2 >MAX_DISTANCE_FOR_MAPPING:
                # Map based on the closest box
                if min_index==0:
                    current_mapping[view+'0']= bbox1
                    current_mapping[view+'1']= prev_trackers[view+'1'] if prev_trackers!={} else bbox2
                elif min_index==1:
                    current_mapping[view+'1']= bbox1
                    current_mapping[view+'0']= prev_trackers[view+'0'] if prev_trackers!={} else bbox2
                elif min_index==2:
                    current_mapping[view+'0']= bbox2
                    current_mapping[view+'1']= prev_trackers[view+'1'] if prev_trackers!={} else bbox1
                else:
                    current_mapping[view+'1']= bbox2
                    current_mapping[view+'0']= prev_trackers[view+'0'] if prev_trackers!={} else bbox1
            
        else:
            # Map based on error
            current_mapping= {view+'0':bbox2, view+'1':bbox1}
            
            if dist_1_2 >MAX_DISTANCE_FOR_MAPPING or dist_2_1 >MAX_DISTANCE_FOR_MAPPING:
                # Map based on the closest box
                if min_index==0:
                    current_mapping[view+'0']= bbox1
                    current_mapping[view+'1']= prev_trackers[view+'1'] if prev_trackers!={} else bbox2
                elif min_index==1:
                    current_mapping[view+'1']= bbox1
                    current_mapping[view+'0']= prev_trackers[view+'0'] if prev_trackers!={} else bbox2
                elif min_index==2:
                    current_mapping[view+'0']= bbox2
                    current_mapping[view+'1']= prev_trackers[view+'1'] if prev_trackers!={} else bbox1
                else:
                    current_mapping[view+'1']= bbox2
                    current_mapping[view+'0']= prev_trackers[view+'0'] if prev_trackers!={} else bbox1
           
        return current_mapping  
    
def map_bbox_singles(new_bboxes, homography_bbox1,view, prev_trackers, frame_number=None):
    MAX_DISTANCE_FOR_MAPPING = 150
    if len(new_bboxes) == 0:
        if prev_trackers=={}:
            return {view+'0':homography_bbox1}
        else:
            return {view+'0':prev_trackers[view+'0']}
            
    elif len(new_bboxes) ==1:
        bbox = new_bboxes[0][:-1]
        bbox_bottom_centre = ((bbox[0] + bbox[2]) // 2, bbox[3])
        if view == 'top':
            dist_1= calculate_distance(bbox_bottom_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[3]])
        else: #TO BE CHANGED
            dist_1= calculate_distance(bbox_bottom_centre, [(homography_bbox1[0]+homography_bbox1[2])//2, homography_bbox1[1]])
        if dist_1>MAX_DISTANCE_FOR_MAPPING:
            if prev_trackers=={}:
                return {view+'0':homography_bbox1}
            else:
                return {view+'0':prev_trackers[view+'0']}
        else:
            if prev_trackers=={}:
                return {view+'0':bbox}
            else:
                return {view+'0':bbox}
        
        
        
def get_intersection(rect, field):
    rect_p = Polygon([(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])])
    field_p = Polygon(field)
    return (rect_p.intersection(field_p)).area / rect_p.area

    
def get_intersection_ratio_point_in_polygon(rect, field):
    from shapely.geometry import Point, Polygon

    def is_point_in_polygon(x, y, poly):
        return poly.contains(Point(x, y))

    rect_p = Polygon([(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])])
    field_p = Polygon(field)
    
    points_inside = sum(is_point_in_polygon(pt[0], pt[1], field_p) for pt in rect_p.exterior.coords)
    total_points = len(rect_p.exterior.coords)
    
    return points_inside / total_points
def draw_polygon(image, vertices, color = (0,255,255), thickness=2):
    # Convert vertices to an appropriate format for cv2.polylines
    points = np.array(vertices, np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Draw the polygon on the image
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    return image


def draw_player_legends(frame, player_image_dir, NUM_PLAYERS_PER_TEAM):
    """
    Places four small images on the middle right side of the frame and adds player names above them.

    Args:
    - frame (numpy.ndarray): The larger frame.
    - player_image_dir (str): Directory where player images are stored.

    Returns:
    - numpy.ndarray: The final frame with small images placed and names added.
    """
    cropped_images_dir = os.path.join(player_image_dir, "cropped_player_legends")
    small_images = []
    player_names = []

    for side in ["top", "bottom"]:
        for i in range(NUM_PLAYERS_PER_TEAM):
            filename = f"{side}{i}.jpg"
            img_path = os.path.join(cropped_images_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                small_images.append(img)
                player_names.append(filename.split('.')[0])  # Extract name without file extension
    
    frame_height, frame_width = frame.shape[:2]

    # Calculate the starting x position (right side)
    x_start = frame_width - max([img.shape[1] for img in small_images]) - 10  # 10 pixels padding

    # Calculate the total height of small images with padding
    total_small_height = sum([img.shape[0] for img in small_images]) + 60  # 10 pixels padding between images

    # Calculate the starting y position (middle vertically)
    y_start = (frame_height - total_small_height) // 2

    # Place the images and add names
    y_offset = y_start
    for img, name in zip(small_images, player_names):
        img_height, img_width = img.shape[:2]

        # Ensure the small image fits within the frame
        if y_offset + img_height <= frame_height:
            # Add the player name above the image
            text_position = (x_start, y_offset - 10)
            cv2.putText(frame, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Place the image
            frame[y_offset:y_offset+img_height, x_start:x_start+img_width] = img
        y_offset += img_height + 20  # Add padding between images

    return frame
            
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)  # Specify the loader for safe loading

    

def get_players_inside_court(player_bounding_boxes, global_const, view='top'):
    MIN_IOU = 0.25
    # Create a new list to store valid bounding boxes
    
    valid_bounding_boxes = []
    view = view.upper()
    
    if view == 'TOP':
        court_center_x = global_const["COURT_CENTER_X"]
        court_center_y = global_const['NET_Y_FOR_PLAYER_SEPERATION']
        # Get court boundaries from global constants
        field_bbox = global_const["FIELD_BOUNDARY"]
           
    else:
        court_center_x = global_const[f"{view}_VIEW_COURT_CENTER_X"]
        court_center_y = global_const[f'{view}_VIEW_NET_Y_FOR_PLAYER_SEPERATION']
        # Get court boundaries from global constants
        field_bbox = global_const[f"{view}_VIEW_FIELD_BOUNDARY"]
        
    # Iterate over the player bounding boxes
    for player_box in player_bounding_boxes:
        if is_within_field_polygon(player_box,
                           field_bbox):
            valid_bounding_boxes.append(player_box)

    # Assign the filtered bounding boxes back to the original list
    new_player_bounding_boxes = valid_bounding_boxes
    top_bbox = []
    bottom_bbox = []
    
    if len(new_player_bounding_boxes) >= 2* global_const['NUM_PLAYERS_PER_TEAM']:
      
        # Calculate IoU for each bounding box with the field
        for i in range(len(new_player_bounding_boxes)):
            x1, y1, x2, y2, conf = new_player_bounding_boxes[i]
            intersection_val = get_intersection((x1, y1, x2, y2), field_bbox)
            new_player_bounding_boxes[i] = (x1, y1, x2, y2, intersection_val)

        # Sort in descending order of IoU (used as confidence)
        new_player_bounding_boxes.sort(key=lambda x: x[4], reverse=True)

        top_player_count =0 
        bottom_player_count = 0
        for box in new_player_bounding_boxes:
            x1, y1, x2, y2, iou = box
            if iou < MIN_IOU:
                continue
            if y2 < court_center_y and top_player_count < global_const['NUM_PLAYERS_PER_TEAM']:
                top_bbox.append([x1, y1, x2, y2, iou])
                top_player_count += 1
            
            elif y2 >= court_center_y and bottom_player_count < global_const['NUM_PLAYERS_PER_TEAM']:
                bottom_bbox.append([x1, y1, x2, y2, iou])
                bottom_player_count += 1
    else:
        # Separate bounding boxes into top and bottom
        for i in range(len(new_player_bounding_boxes)):
            x1, y1, x2, y2, conf = new_player_bounding_boxes[i]
            intersection_val = get_intersection((x1, y1, x2, y2), field_bbox)
            new_player_bounding_boxes[i] = (x1, y1, x2, y2, intersection_val)

        # Sort in descending order of IoU (used as confidence)
        new_player_bounding_boxes.sort(key=lambda x: x[4], reverse=True)
        top_player_count =0 
        bottom_player_count = 0
        for box in new_player_bounding_boxes:
            x1, y1, x2, y2, iou = box
            if iou<MIN_IOU:
                continue
            if y2 < court_center_y and top_player_count < global_const['NUM_PLAYERS_PER_TEAM']:
                top_bbox.append([x1, y1, x2, y2, iou])
                top_player_count += 1
            elif  y2 >= court_center_y and bottom_player_count < global_const['NUM_PLAYERS_PER_TEAM']:
                bottom_bbox.append([x1, y1, x2, y2,iou])
                bottom_player_count += 1
    
    return top_bbox, bottom_bbox
