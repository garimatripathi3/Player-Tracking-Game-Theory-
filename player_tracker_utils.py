import math
import cv2
import cvzone
import numpy as np
from players_tracking.utils import *
from players_tracking.homography_utils import *
from sklearn.cluster import KMeans
from pprint import pprint
from utils import *
#changes made by Garima
def compute_histograms_and_concatenate(image):
    histograms = []
    for color_space in [cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB]:
        converted_img = cv2.cvtColor(image, color_space)
        hist = cv2.calcHist([converted_img], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        normalized_hist = cv2.normalize(hist, None).flatten()
        histograms.append(normalized_hist)
    return np.concatenate(histograms)
def calculate_color_space_histograms(data, conversion_operations):
    histograms = []
    for flag, conversion_function in conversion_operations:
        if conversion_function is cv2.cvtColor:
            converted_data = conversion_function(data.reshape(-1, 1, 3).astype(np.uint8), flag)
            # Calculate histogram and normalize
            hist = cv2.calcHist([converted_data], [0], None, [256], [0, 256])
            normalized_hist = cv2.normalize(hist, hist).flatten()
            histograms.append(normalized_hist)
        else:
            # Assume it's the identity function for RGB to RGB
            hist = cv2.calcHist([data.reshape(-1, 1, 3).astype(np.uint8)], [0], None, [256], [0, 256])
            normalized_hist = cv2.normalize(hist, hist).flatten()
            histograms.append(normalized_hist)
    return histograms
def identity_function(image):
    return image

def colour_matching_algorithm(top_bbox, 
                              bottom_bbox,
                              NUM_PLAYERS_PER_TEAM,
                              teams,
                              match_variables,
                              frame,
                              initialize,
                              global_const):
    '''
    Function to match player bounding boxes based on color.

    Parameters:
    - top_bbox: Bounding boxes for the top team
    - bottom_bbox: Bounding boxes for the bottom team
    - NUM_PLAYERS_PER_TEAM: Number of players per team
    - player_tracker_color: Dictionary containing tracked player colors
    - player_tracker: Dictionary containing tracked player positions
    - frame: Current video frame
    - initialize: Boolean indicating if this is the initialization step
    - global_const: Dictionary containing global constants
    - deciding_factor: Factor to decide player color matching (default is None)

    Returns:
    - to_be_rechecked: Boolean indicating if color recheck is needed
    - persons_mapped: Dictionary indicating which players were mapped
    - draw_frame: Processed frame with bounding boxes and labels
    '''
    persons_mapped = {'top': [0, 0], 'bottom': [0, 0]}
    marked_boxes = []
    draw_frame = frame.copy()
    to_be_rechecked = False
    
    bkg_color = match_variables.bkg_color_range 
    if bkg_color:   
        bkg_l_start,bkg_l_end = bkg_color[0]
        bkg_a_start,bkg_a_end = bkg_color[1]
        bkg_b_start,bkg_b_end = bkg_color[2]
    
    # Iterate over top and bottom teams
    for side in teams:
        boxes = top_bbox if side == 'top' else bottom_bbox
        if len(boxes) != NUM_PLAYERS_PER_TEAM:
            to_be_rechecked = True
        
        # Case of singles match
        if len(boxes) == 1 and match_variables.match_type != "doubles":
            teams[side].players[0].tracker.append(boxes[0])
            marked_boxes.append(boxes[0])
            continue
        
        bbox_colour_diff = []  # List to store color differences

        # Iterate over players in the team
        for i in range(NUM_PLAYERS_PER_TEAM):
            
            for idx, bbox in enumerate(boxes):
                x1, y1, x2, y2,_ = bbox
                shirt_roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2] if side == "top" else \
                            frame[y1 + (y2 - y1) // 2:y2, x1:x2]
                
               
                shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
                shirt_roi_hsv_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
                shirt_roi_rgb_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2RGB)
              
                
                stacked_shirt_roi = np.concatenate((shirt_roi_lab_color, shirt_roi_hsv_color, shirt_roi_rgb_color),axis = 2)
                stacked_shirt_roi = np.mean(stacked_shirt_roi.reshape(shirt_roi.shape[0], shirt_roi.shape[1], 3, 3), axis=3)
                bkg_mask_l = (stacked_shirt_roi[:, :, 0] > bkg_l_start) & (stacked_shirt_roi[:, :, 0] < bkg_l_end)
                bkg_mask_a = (stacked_shirt_roi[:, :, 1] > bkg_a_start) & (stacked_shirt_roi[:, :, 1] < bkg_a_end)
                bkg_mask_b = (stacked_shirt_roi[:, :, 2] > bkg_b_start) & (stacked_shirt_roi[:, :, 2] < bkg_b_end)
                bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
                player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask
    
                if teams[side].to_do_mask:
                
                    filtered_color_values = stacked_shirt_roi[player_mask]
                else:
                    filtered_color_values = stacked_shirt_roi[np.ones_like(player_mask, dtype=bool)]
               
                if filtered_color_values.size > 0:
                    mean_col = np.mean(filtered_color_values, axis=0)
                else:
                    mean_col = np.array([0, 0, 0])

                #Changes made by Garima 
                color_conversion_flags = [
                        (cv2.COLOR_RGB2HSV, cv2.cvtColor), 
                        (cv2.COLOR_RGB2LAB, cv2.cvtColor)
                        ] 
                reshaped_rgb_data = filtered_color_values.reshape(-1, 1, 3).astype(np.uint8)
                color_space_histograms = calculate_color_space_histograms(reshaped_rgb_data, color_conversion_flags)
                final_feature_vector = np.concatenate(color_space_histograms)
                color_diff = np.linalg.norm(np.array(final_feature_vector) - np.array(teams[side].players[i].color))
                
                
                bbox_colour_diff.append(color_diff)
                
                if bbox in marked_boxes:
                    continue
        
        # Didn't get two boxes
        if len(bbox_colour_diff) != 4:
            continue
        
        # Initialize only when both players are found
        if (initialize and not teams[side].ignore_color_for_tracking) or \
            teams[side].players[0].tracker ==[] or teams[side].players[1].tracker ==[]:
            
            # Assign player tracker based on color similarity
            error_1 = abs(bbox_colour_diff[0] + bbox_colour_diff[3])
            error_2 = abs(bbox_colour_diff[1] + bbox_colour_diff[2])
            if error_1 < error_2:
                teams[side].players[0].tracker.append(boxes[0])
                teams[side].players[1].tracker.append(boxes[1])
            else:
                teams[side].players[0].tracker.append(boxes[1])
                teams[side].players[1].tracker.append(boxes[0])

        else:
            # Map the people using this data
            sum_diff = sum(bbox_colour_diff)
            confidence = [(1 - colour_diff / sum_diff) for colour_diff in bbox_colour_diff]
            confidence_sum_0 = sum(confidence[:2])
            confidence_sum_1 = sum(confidence[2:])
            
            for idx, val in enumerate(confidence):
                confidence[idx] = val / (confidence_sum_0 if idx < 2 else confidence_sum_1)

            world_distance = calculate_distance_in_world_frame(teams[side].players, global_const[f"i_H_{side}"], NUM_PLAYERS_PER_TEAM)
            
            # Map player using previous position
            # World distance is less than threshold and color has confidence difference is greater than 0.1
            if (world_distance < global_const["COLOUR_TO_D_THRESHOLD"] and 
                (abs(confidence[0] - confidence[1]) > 0.1 or abs(confidence[2] - confidence[3]) > 0.1)
                and not teams[side].ignore_color_for_tracking):
                
                error_1 = abs(bbox_colour_diff[0] + bbox_colour_diff[3])
                error_2 = abs(bbox_colour_diff[1] + bbox_colour_diff[2])
                
                if error_1 < error_2:
                    teams[side].players[0].tracker.append(boxes[0])
                    teams[side].players[1].tracker.append(boxes[1])
                else:
                    teams[side].players[0].tracker.append(boxes[1])
                    teams[side].players[1].tracker.append(boxes[0])
                
                marked_boxes.extend(boxes)
                continue

            else: # Map with distance if the world distance is greater than threshold or confidence is low
                
                # Recheck if the world distance is less than threshold
                if world_distance < global_const["COLOUR_TO_D_THRESHOLD"]:
                    to_be_rechecked = True

                for i in range(NUM_PLAYERS_PER_TEAM):
                    if persons_mapped[side][i] == 0:
                        min_dist = float('inf')
                        min_dist_bbox = None
                        
                        player_prev_box = teams[side].players[i].tracker[-1]
                        player_prev_coordinate = ((player_prev_box[0] + player_prev_box[2]) // 2, (player_prev_box[1] + player_prev_box[3]) // 2)
                        
                        boxes = top_bbox if side == 'top' else bottom_bbox
                        
                        for bbox in boxes:
                            if bbox in marked_boxes:
                                continue
                            x1, y1, x2, y2,_ = bbox
                            current_coordinate = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                            movement = calculate_distance(player_prev_coordinate, current_coordinate)
                        
                            if movement < min_dist:
                                min_dist = movement
                                min_dist_bbox = bbox
                        
                        if min_dist_bbox is not None:
                            teams[side].players[i].tracker.append(min_dist_bbox)
                            marked_boxes.append(min_dist_bbox)
                            persons_mapped[side][i] = 1
            
    return to_be_rechecked, persons_mapped, teams, draw_frame



def track_players_movement(frame, teams, match_variables, top_bbox, bottom_bbox, global_const, initialize=False):
    '''
    Function to track the movement of the players.

    Parameters:
    - frame: Current video frame
    - top_bbox: Bounding boxes for the top team
    - bottom_bbox: Bounding boxes for the bottom team
    - global_const: Dictionary containing global constants
    - initialize: Boolean indicating if this is the initialization step
    - deciding_factor: Factor to decide player color matching

    Returns:
    - recheck_color: Indicator if color recheck is needed
    - player_tracker: Updated player tracker
    - player_tracker_color: Updated player tracker color
    - frame: Processed frame
    '''
    NUM_PLAYERS_PER_TEAM = global_const['NUM_PLAYERS_PER_TEAM']

    # Perform color matching algorithm to map players
    to_be_rechecked, persons_mapped, teams, frame = colour_matching_algorithm(
        top_bbox, bottom_bbox, NUM_PLAYERS_PER_TEAM, 
        teams,match_variables, frame, 
        initialize, global_const
    )
    recheck_color = 25 if to_be_rechecked else -1
        

    # Append the same bbox if a player is not mapped
    for side in teams:
        for i in range(NUM_PLAYERS_PER_TEAM):
            if persons_mapped[side][i] == 0:
                if teams[side].players[i].tracker != []:
                    teams[side].players[i].tracker.append(teams[side].players[i].tracker[-1])
                else:
                    teams[side].players[i].tracker.append([0, 0, 1, 1])

    # Pop the first element from the player tracker if the length is greater than 30
    for key in teams:
        for i in range(NUM_PLAYERS_PER_TEAM):
            if len(teams[key].players[i].tracker) > 30:
                teams[key].players[i].tracker.pop(0)

    return recheck_color, teams, frame



























