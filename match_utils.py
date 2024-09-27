from tqdm import tqdm
import cv2
import os
import json
import numpy as np

from players_tracking.utils import load_config, get_players_inside_court
from players_tracking.bkg_color_utils import get_space_range_using_histograms

class MatchVariables():
    
    def __init__(self, rally_times, view_dir, config_file, write_video, view, main_view_frame_offset, match_type='doubles'):
        
        self.rally_times = rally_times
        self.view_dir = view_dir
        self.config = load_config(config_file)
        self.output_image_dir = ''
        
        self.write_video = write_video
        self.view = view
        self.main_view_frame_offset = main_view_frame_offset 
        
        self.bkg_color_range = None 
        self.match_type = match_type
        self.num_players_per_team = 2 if self.match_type == 'doubles' else 1
        self.yolo_data = self.get_yolo_data()
    
    def get_yolo_data(self):
        all_yolo_data = {}
        for rally_idx in range(len(self.rally_times[:-1])):
            yolo_path = os.path.join(self.view_dir, f"rally_{rally_idx}", "yolo_detections", "yolo_detections.json")
            with open(yolo_path, "r") as fp:
                yolo_data = json.load(fp)
            for keys in yolo_data:
                if keys not in all_yolo_data:
                    all_yolo_data[keys] = yolo_data[keys]    
            
        return all_yolo_data
    
    
    def get_court_background_color_range(self, 
                                     video_variables,
                                     global_const: list):

        rallies_to_get_frames_from = [2]
        num_frames_from_each_rally = 1  ##right now manually given, change later

        for rally_idx in rallies_to_get_frames_from:
            
            ## take frames from this rally
            video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.rally_times[rally_idx])

            frames_with_background_only = []

            for idx in range(num_frames_from_each_rally):

                success, frame =video_variables.video_capture.read()
                mask =np.zeros((frame.shape[0], frame.shape[1]))
                cv2.fillPoly(mask, [np.array(global_const["FIELD_BOUNDARY"], np.int32)], 255)
                
                YOLO_BOXES_FOR_CURR_FRAME = self.yolo_data.get(str(self.rally_times[rally_idx] + idx), [])

                #Remove players from mask
                for bbox in YOLO_BOXES_FOR_CURR_FRAME:
                    mask[bbox[1]-30:bbox[3]+30, bbox[0]-30:bbox[2]+30] = 0
                    
                # Remove the white points from the frame
                white_mask = cv2.inRange(frame, (200, 200, 200), (255, 255, 255))
                mask[white_mask == 255] = 0
                
                mask = mask.astype('uint8')

                ## using the mask get the background part from the frame
                frame_with_background_only = cv2.bitwise_and(frame, frame, mask=mask)
                frames_with_background_only.append(frame_with_background_only)

        ## FROM the selected frames with background will extract the lab range
        all_lab_ranges = []
        for background_frame in frames_with_background_only:
            print(len(background_frame))
            ranges = get_space_range_using_histograms(background_frame,self.view_dir,self.write_video)
            all_lab_ranges.append(ranges)
            
        all_lab_ranges = np.array(all_lab_ranges)

        mean_range_L_start = np.mean(all_lab_ranges[:, 0])
        mean_range_L_end = np.mean(all_lab_ranges[:, 1])
        mean_range_A_start = np.mean(all_lab_ranges[:, 2])
        mean_range_A_end = np.mean(all_lab_ranges[:, 3])
        mean_range_B_start = np.mean(all_lab_ranges[:, 4])
        mean_range_B_end = np.mean(all_lab_ranges[:, 5])
        
        self.bkg_color_range =  [(mean_range_L_start, mean_range_L_end), (mean_range_A_start, mean_range_A_end), (mean_range_B_start, mean_range_B_end)]

    

  
    def get_frames_with_req_num_players_in_match(self, video_variables,
                                          global_const: dict, num_of_player_per_team_to_check= None):
    
        all_frames_with_req_players = []
        rally_times = self.rally_times
        
        expected_num_players = global_const["NUM_PLAYERS_PER_TEAM"] 
        
        global_const["NUM_PLAYERS_PER_TEAM"] = num_of_player_per_team_to_check if num_of_player_per_team_to_check is not None else expected_num_players
        
        for idx in tqdm(range(2,min(12,len(rally_times)-1)), desc=f"Getting frames with all {global_const['NUM_PLAYERS_PER_TEAM']} players"):
            
            if idx == len(rally_times)-1:
                break
            
            frames_with_req_players =  self.get_frames_with_req_num_players_in_rally(video_variables, 
                                                                        global_const, idx,
                                                                        num_frames_to_pick = 10)
            
            all_frames_with_req_players.extend(frames_with_req_players) 
            
        # Resetting the global constant
        global_const["NUM_PLAYERS_PER_TEAM"] = expected_num_players
        
        return all_frames_with_req_players
    
    def get_frames_with_req_num_players_in_rally(self, 
                                        video_variables,
                                        global_const: dict, rally_idx,
                                        num_frames_to_pick = 10):
    
            
        frame_numbers_picked = []
        start_frame = self.rally_times[rally_idx]
        end_frame = self.rally_times[rally_idx+1]
        
        curr_frame_number = start_frame
        
        num_frames_with_req_player = 0 # Number of frames where 4 players are present
        missing_req_player_count = 0 # Number of frames where 4 players are missing
        selection_start = 0 
        
        while curr_frame_number < end_frame and len(frame_numbers_picked) < num_frames_to_pick:
            
            if selection_start == 1:
                video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_number)
            
            success, frame = video_variables.video_capture.read()
            if not success:
                break
            
            YOLO_BOXES_FOR_CURR_FRAME = self.yolo_data.get(str(curr_frame_number), [])
            top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME,
                                                                        global_const)
            
            if len(top_player_bbox) == global_const["NUM_PLAYERS_PER_TEAM"] and len(bottom_player_bbox) == global_const["NUM_PLAYERS_PER_TEAM"]:
                num_frames_with_req_player += 1
                missing_req_player_count = 0
            else:
                missing_req_player_count += 1
                if missing_req_player_count > video_variables.fps * 1:
                    num_frames_with_req_player = 0
            
            if num_frames_with_req_player < global_const["COUNT_FRAMES_BEFORE_START"]:
                curr_frame_number += 1
                if curr_frame_number == end_frame:
                    break
                continue
            
            if num_frames_with_req_player == global_const["COUNT_FRAMES_BEFORE_START"]:
                selection_start = 1
            
            if len(top_player_bbox) == global_const["NUM_PLAYERS_PER_TEAM"] and len(bottom_player_bbox) == global_const["NUM_PLAYERS_PER_TEAM"]:
                if curr_frame_number not in frame_numbers_picked:
                    frame_numbers_picked.append(curr_frame_number)
            
            curr_frame_number += 5
        return frame_numbers_picked