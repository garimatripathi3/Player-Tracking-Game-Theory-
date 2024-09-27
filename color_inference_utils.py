import os
import json
import cv2
import numpy as np
from players_tracking.utils import get_players_inside_court
from players_tracking.cspace_utils import get_color_of_player

def classificaion_on_sampled_images(teams,
                                    all_players_color_values_each_space,
                                    video_variables,
                                    match_variables):
        
    for side in teams:
        # best_space = teams[side].best_space
        color_values = all_players_color_values_each_space[side] #changes_made_today
        for idx in range(0,len(color_values),2):
                    
            if idx + 1 >= len(color_values):
                continue
            
            frame_no = color_values[idx][2] ## at idx 2 we have the frame number 
            # print("frame number: ", frame_no)
            
            if frame_no != color_values[idx][2]:
                print("Frame number mismatch")
                
            bbox_1 = color_values[idx][1]
            bbox_2= color_values[idx+1][1]
            
            x1, y1, x2, y2 = bbox_1
            x3, y3, x4, y4 = bbox_2
            
            video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = video_variables.video_capture.read()
            
            img_1 = frame[y1:y2, x1:x2]
            img_2 = frame[y3:y4, x3:x4]
            
            out_path = match_variables.output_image_dir
            
            lab_val_1 = color_values[idx][0]
            lab_val_2 = color_values[idx+1][0]
            
            centroids_lab_1 = teams[side].players[0].color
            centroids_lab_2 = teams[side].players[1].color
            
            error11_22 = np.linalg.norm(np.array(lab_val_1) - np.array(centroids_lab_1)) + np.linalg.norm(np.array(lab_val_2) - np.array(centroids_lab_2))
            error12_21 = np.linalg.norm(np.array(lab_val_1) - np.array(centroids_lab_2)) + np.linalg.norm(np.array(lab_val_2) - np.array(centroids_lab_1))
            
            os.makedirs(f"{out_path}/{side}_c1", exist_ok=True)
            os.makedirs(f"{out_path}/{side}_c2", exist_ok=True)
            
            if error11_22 < error12_21:
                cv2.imwrite(f"{out_path}/{side}_c1/{frame_no}_{int(lab_val_1[0])}_{int(lab_val_1[1])}_{int(lab_val_1[2])}.jpg", img_1)
                cv2.imwrite(f"{out_path}/{side}_c2/{frame_no}_{int(lab_val_2[0])}_{int(lab_val_2[1])}_{int(lab_val_2[2])}.jpg", img_2)
            else:
                cv2.imwrite(f"{out_path}/{side}_c2/{frame_no}_{int(lab_val_1[0])}_{int(lab_val_1[1])}_{int(lab_val_1[2])}.jpg", img_1)
                cv2.imwrite(f"{out_path}/{side}_c1/{frame_no}_{int(lab_val_2[0])}_{int(lab_val_2[1])}_{int(lab_val_2[2])}.jpg", img_2)
      

def save_player_legends_doubles(teams, match_variables, all_color_values_for_initialization, video_variables):
        
    image_output_path = os.path.join(match_variables.output_image_dir,"cropped_player_legends")
    
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)    
    
    for side in teams:
        p1_idx = (teams[side].players[0].closest_index)
        p2_idx = (teams[side].players[1].closest_index)
        best_space = teams[side].best_space
        
        bbox_1 = all_color_values_for_initialization[side][p1_idx][1]
        bbox_2 = all_color_values_for_initialization[side][p2_idx][1]
        
        x1, y1, x2, y2 = bbox_1
        x3, y3, x4, y4 = bbox_2
        
        frame_p1_no = all_color_values_for_initialization[side][p1_idx][2]
        frame_p2_no = all_color_values_for_initialization[side][p2_idx][2]
        
        video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_p1_no)
        ret, frame_p1 = video_variables.video_capture.read()

        video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_p2_no)
        ret, frame_p2 = video_variables.video_capture.read()

        img_1 = frame_p1[y1:y2, x1:x2]
        img_2 = frame_p2[y3:y4, x3:x4]
        
        if side == "top":
            img_1_cropped = img_1[:(y2 - y1) // 2, :]
            img_2_cropped = img_2[:(y4 - y3) // 2, :]
        else:
            img_1_cropped = img_1[(y2 - y1) // 2:, :]
            img_2_cropped = img_2[(y4 - y3) // 2:, :]
            
        if side == 'top':
            cv2.imwrite(f"{image_output_path}/top0.jpg", img_1_cropped)
            cv2.imwrite(f"{image_output_path}/top1.jpg", img_2_cropped)
        else:
            cv2.imwrite(f"{image_output_path}/bottom0.jpg", img_1_cropped)
            cv2.imwrite(f"{image_output_path}/bottom1.jpg", img_2_cropped)
 

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
 
def save_player_legends_singles(teams, match_variables, video_variables, global_const):
        
    image_output_path = os.path.join(match_variables.output_image_dir,"cropped_player_legends")
    
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)    
    
    rally_idx = 2
    YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(match_variables.rally_times[rally_idx]), [])
    top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)
    
    frame_number = match_variables.rally_times[rally_idx] 
    while len(top_player_bbox) != 1 or len(bottom_player_bbox) != 1:
        frame_number +=1
        YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(frame_number), [])
        top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)
    
    top_player_bbox = top_player_bbox[0]
    bottom_player_bbox = bottom_player_bbox[0]
    
    video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame = video_variables.video_capture.read()
    
    x1, y1, x2, y2,_ = top_player_bbox
    x3, y3, x4, y4, _ = bottom_player_bbox
    
    for side in teams:
        if side == "top":
            img_1 = frame[y1:y2, x1:x2]
        else:
            img_1 = frame[y3:y4, x3:x4]
        
        if side == 'top':
            cv2.imwrite(f"{image_output_path}/top0.jpg", img_1)
        else:
            cv2.imwrite(f"{image_output_path}/bottom0.jpg", img_1)
    

    
    
           
import json
import numpy as np

class NumpyFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def create_tracker_sampler_json_doubles(teams,all_color_values_for_initialization, match_variables):
    top_p1_best_frames = []
    top_p2_best_frames = []
    bottom_p1_best_frames = []
    bottom_p2_best_frames = []
    
    for side in teams:
        if side == 'top':
            p1_idx_list = teams[side].players[0].closest_4_indices
            best_space = teams[side].best_space
            for idx in p1_idx_list:
                # print(len(all_color_values_for_initialization))
                # print(all_color_values_for_initialization[side])
                # print(all_color_values_for_initialization[side][idx])
                # print(all_color_values_for_initialization[side][idx][2])
                frame_num = all_color_values_for_initialization[side][idx][2]
                bbox = all_color_values_for_initialization[side][idx][1]
                top_p1_best_frames.append([frame_num, bbox])
            
            p2_idx_list = teams[side].players[1].closest_4_indices
            for idx in p2_idx_list:
                frame_num = all_color_values_for_initialization[side][idx][2]
                bbox = all_color_values_for_initialization[side][idx][1]
                top_p2_best_frames.append([frame_num, bbox])
        
        if side == 'bottom':
            p1_idx_list = teams[side].players[0].closest_4_indices
            for idx in p1_idx_list:
                frame_num = all_color_values_for_initialization[side][idx][2]
                bbox = all_color_values_for_initialization[side][idx][1]
                bottom_p1_best_frames.append([frame_num, bbox])
            
            p2_idx_list = teams[side].players[1].closest_4_indices
            for idx in p2_idx_list:
                frame_num = all_color_values_for_initialization[side][idx][2]
                bbox = all_color_values_for_initialization[side][idx][1]
                bottom_p2_best_frames.append([frame_num, bbox])
                    
    
    frames_to_sample = {
        
        'top': {'player1_frame_and_bbox': top_p1_best_frames,
                'player2_frame_and_bbox': top_p2_best_frames, 
                'background_sub': teams['top'].to_do_mask, 
                'ignore_color_for_tracking': teams['top'].ignore_color_for_tracking,
                # 'best_color_space' : teams['top'].best_space.color_space,
                'top0_color': list(teams['top'].players[0].color),
                'top1_color': list(teams['top'].players[1].color),
                'num_frames_to_same_cluster': teams['top'].num_frames_same_cluster,
                'cluster_ratio': teams['top'].cluster_length_ratio
                },
        'bottom': {'player1_frame_and_bbox': bottom_p1_best_frames,
                    'player2_frame_and_bbox': bottom_p2_best_frames,
                    'background_sub': teams['bottom'].to_do_mask,
                    'ignore_color_for_tracking': teams['bottom'].ignore_color_for_tracking,
                    # 'best_color_space' : teams['bottom'].best_space.color_space,
                    'bottom0_color': list(teams['bottom'].players[0].color),
                    'bottom1_color': list(teams['bottom'].players[1].color),
                    'bottom_num_frames_to_same_cluster': teams['bottom'].num_frames_same_cluster,
                    'cluster_ratio': teams['bottom'].cluster_length_ratio
                    }
    }

    output_path = os.path.join(match_variables.output_image_dir, "player_tracker_best_boxes.json")
    with open(output_path, 'w') as json_file:
        json.dump(frames_to_sample, json_file, indent=4, cls=NumpyFloatEncoder)
            
def create_tracker_sampler_json_singles(teams, match_variables, video_variables, global_const):
    num_instances_each_player = 0
    top_best_frames = []
    bottom_best_frames = []
    
    rally_idx = 1
    YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(match_variables.rally_times[rally_idx]), [])
    top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)
    
    frame_number = match_variables.rally_times[rally_idx] 
    
    while num_instances_each_player != 4:
        
        frame_number +=1
        YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(frame_number), [])
        top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)
        
        if len(top_player_bbox) == 1 and len(bottom_player_bbox) == 1:
            top_best_frames.append([frame_number, top_player_bbox[0][:4]])
            bottom_best_frames.append([frame_number, bottom_player_bbox[0][:4]])
            num_instances_each_player += 1
    
    video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, top_best_frames[0][0])
    _, frame = video_variables.video_capture.read()
    top_color = list(get_color_of_player(frame, top_best_frames[0][1]))
    bottom_color = list(get_color_of_player(frame, bottom_best_frames[0][1]))        
    
    frames_to_sample = {
        'top': {'player1_frame_and_bbox': top_best_frames,
                'background_sub': False, 
                'ignore_color_for_tracking': False,
                'best_color_space' : 'HSV',
                'top0_color': top_color,
                },
        'bottom': {'player1_frame_and_bbox': bottom_best_frames,
                    'background_sub': False,
                    'ignore_color_for_tracking': False,
                    'best_color_space' : 'HSV',
                    'bottom0_color': bottom_color,
                    }
    }

    output_path = os.path.join(match_variables.output_image_dir, "player_tracker_best_boxes.json")
    with open(output_path, 'w') as json_file:
        json.dump(frames_to_sample, json_file, indent=4)
        
