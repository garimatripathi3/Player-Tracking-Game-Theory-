import cv2
from tqdm import tqdm
import numpy as np
import os

from players_tracking.utils import get_players_inside_court

def get_players_mean_code_values(
                                all_frames_with_req_num_of_players,
                                video_variables,
                                match_variables,
                                global_const, teams):
    
    all_color_values = {'top': [], 'bottom': []}
        
    ## now will iterate over all the frames and get lab values for all the players in it
    for frame_num in tqdm(all_frames_with_req_num_of_players):

        # code = get_color_conversion_code()
        
        # (color, bbox, frame_number, player_pixel_ratio>0.1)
        color_value = get_all_players_color_code_value(frame_num,
                                            video_variables, match_variables,
                                            {team: teams[team].to_do_mask for team in teams},
                                            global_const)
        
        
        all_color_values["top"].extend(color_value['top'])
        all_color_values["bottom"].extend(color_value['bottom'])
        # code = get_color_conversion_code()
    
    return all_color_values
def normalize_histogram(hist):
    return hist / hist.sum() if hist.sum() > 0 else hist


def compute_histograms_and_concatenate(image):
    histograms = []
    for color_space in [cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB]:
        converted_img = cv2.cvtColor(image, color_space)
        hist = cv2.calcHist([converted_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        normalized_hist = cv2.normalize(hist, None).flatten()
        histograms.append(normalized_hist)
    return np.concatenate(histograms)
def calculate_color_space_histograms(data, conversion_operations):
    histograms = []
    for flag, conversion_function in conversion_operations:
        if conversion_function is cv2.cvtColor:
            converted_data = conversion_function(data.reshape(-1, 1, 3).astype(np.uint8), flag)
         
            hist = cv2.calcHist([converted_data], [0], None, [256], [0, 256])
            normalized_hist = cv2.normalize(hist, hist).flatten()
            histograms.append(normalized_hist)
        else:
            hist = cv2.calcHist([data.reshape(-1, 1, 3).astype(np.uint8)], [0], None, [256], [0, 256])
            normalized_hist = cv2.normalize(hist, hist).flatten()
            histograms.append(normalized_hist)
    return histograms
def identity_function(image):
    return image

color_conversion_flags = [
    (cv2.COLOR_RGB2HSV, cv2.cvtColor),
    (cv2.COLOR_RGB2LAB, cv2.cvtColor),
    (None, identity_function)  
]
def get_all_players_color_code_value(frame_number,
                                     video_variables,
                                     match_variables, 
                                     to_do_bkg_sub_for_all_frames, 
                                     global_const,
                                     ):
    '''
    Function to get the lab values of the players' shirts.

    '''
    
    video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame = video_variables.video_capture.read()
    
    YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(frame_number), [])
    top_bbox, bottom_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)

    color_values_res = {'top': [], 'bottom': []}
    
    bkg_color = match_variables.bkg_color_range   
    bkg_l_start,bkg_l_end = bkg_color[0]
    bkg_a_start,bkg_a_end = bkg_color[1]
    bkg_b_start,bkg_b_end = bkg_color[2]
    
    
    for side in ['top', 'bottom']:
        boxes = top_bbox if side == 'top' else bottom_bbox
        color_values = []
        to_do_bkg_sub = to_do_bkg_sub_for_all_frames[side]
        

        for player_idx in range(global_const["NUM_PLAYERS_PER_TEAM"]): 
            
            if len(boxes) > player_idx:
                x1, y1, x2, y2,_ = boxes[player_idx]
                
                if side == "top":
                    shirt_roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                else:
                    shirt_roi = frame[y1 + (y2 - y1) // 2:y2, x1:x2]
                    
 
                shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
                frame_roi_lab_color = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
                shirt_roi_req_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
                
                bkg_mask_l = (shirt_roi_lab_color[:, :, 0] > bkg_l_start) & (shirt_roi_lab_color[:, :, 0] < bkg_l_end)
                bkg_mask_a = (shirt_roi_lab_color[:, :, 1] > bkg_a_start) & (shirt_roi_lab_color[:, :, 1] < bkg_a_end)
                bkg_mask_b = (shirt_roi_lab_color[:, :, 2] > bkg_b_start) & (shirt_roi_lab_color[:, :, 2] < bkg_b_end)
                bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
                
                frame_bkg_mask_l = (frame_roi_lab_color[:, :, 0] > bkg_l_start) & (frame_roi_lab_color[:, :, 0] < bkg_l_end)
                frame_bkg_mask_a = (frame_roi_lab_color[:, :, 1] > bkg_a_start) & (frame_roi_lab_color[:, :, 1] < bkg_a_end)
                frame_bkg_mask_b = (frame_roi_lab_color[:, :, 2] > bkg_b_start) & (frame_roi_lab_color[:, :, 2] < bkg_b_end)
                frame_bkg_mask = frame_bkg_mask_l & frame_bkg_mask_a & frame_bkg_mask_b
                
                player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask 
                
                mask_image = (player_mask * 255).astype(np.uint8)
                player_pixel_ratio = np.count_nonzero(bkg_mask == 0) / bkg_mask.size
                
                if to_do_bkg_sub:
                    filtered_color_values = shirt_roi_req_color[player_mask]
                else:
                    filtered_color_values = shirt_roi_req_color[np.ones_like(player_mask, dtype=bool)]
                         
                color_conversion_flags = [
                    (cv2.COLOR_RGB2HSV, cv2.cvtColor),  # Convert to HSV
                    (cv2.COLOR_RGB2LAB, cv2.cvtColor)
                      ] 
                reshaped_rgb_data = filtered_color_values.reshape(-1, 1, 3).astype(np.uint8)
                color_space_histograms = calculate_color_space_histograms(reshaped_rgb_data, color_conversion_flags)
                final_feature_vector = np.concatenate(color_space_histograms)
            
                color_values.append((final_feature_vector, [x1, y1, x2, y2], frame_number, player_pixel_ratio>0.1))
            else:
                print(f"Player {player_idx} not found on {side} side")
        
        if len(color_values) == global_const["NUM_PLAYERS_PER_TEAM"]:
            color_values_res[side] = color_values
        
    return color_values_res
def calculate_histograms(image):
    '''
    Calculate concatenated histograms for HSV and RGB color spaces of an image.
    '''
    histograms = []
    for color_space in [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB,cv2.COLOR_BGR2RGB ]:  # Consider adding other spaces as needed
        converted_img = cv2.cvtColor(image, color_space)
        hist = cv2.calcHist([converted_img], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        histograms.append(hist)
    return np.concatenate([h.flatten() for h in histograms])        
def get_player_without_bkg(match_variables,shirt_roi):
    bkg_color = match_variables.bkg_color_range   
    bkg_l_start,bkg_l_end = bkg_color[0]
    bkg_a_start,bkg_a_end = bkg_color[1]
    bkg_b_start,bkg_b_end = bkg_color[2]

    shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
    bkg_mask_l = (shirt_roi_lab_color[:, :, 0] > bkg_l_start) & (shirt_roi_lab_color[:, :, 0] < bkg_l_end)
    bkg_mask_a = (shirt_roi_lab_color[:, :, 1] > bkg_a_start) & (shirt_roi_lab_color[:, :, 1] < bkg_a_end)
    bkg_mask_b = (shirt_roi_lab_color[:, :, 2] > bkg_b_start) & (shirt_roi_lab_color[:, :, 2] < bkg_b_end)
    bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
    
    player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask 
    
    player_mask_img = (player_mask * 255).astype(np.uint8)
    
    return player_mask_img
def get_color_of_player(frame, bbox):
    '''
    Function to get the combined color features of the player's shirt 
    from LAB, HSV, and RGB color spaces, including histograms.
    '''
    x1, y1, x2, y2 = bbox
    
    shirt_roi = frame[y1:y2, x1:x2]
    shirt_roi_lab = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
    hist_lab = cv2.calcHist([shirt_roi_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    shirt_roi_hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([shirt_roi_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    shirt_roi_rgb = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2RGB)
    hist_rgb = cv2.calcHist([shirt_roi_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    combined_color_features = np.concatenate([hist_lab, hist_hsv, hist_rgb])
    
    return combined_color_features

