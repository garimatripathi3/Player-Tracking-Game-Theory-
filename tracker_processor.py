import os 
import time
import numpy as np
import shutil
import json
import yaml
from yaml import FullLoader
import cv2
import pandas as pd
import cvzone
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pprint import pprint

from players_tracking.utils import *
from players_tracking.player_tracker_utils import *
from players_tracking.homography_utils import *
from players_tracking.bkg_color_utils import *
from players_tracking.initialization_utils import *
from players_tracking.team_utils import *
from players_tracking.match_utils import *
from players_tracking.cspace_utils import *
from players_tracking.color_inference_utils import *


class PlayerTrackerProcessor:
    def __init__(self, video_path, config_file, view_dir, rally_times,view, write_video=False, main_view_frame_offset =0, match_type='doubles'):
        """
        Initializes the PlayerTrackerProcessor object with the given parameters.

        Args:
            video_path (str): Path to the input video file.
            config_file (str): Path to the configuration YAML file.
            view_dir (str): Directory path for saving output files.
            rally_times (list): List of frame indices representing the start of each rally.
            write_video (bool): Flag indicating whether to write the output video with player tracking.
        """
        
        self.match_variables = MatchVariables(rally_times,\
            view_dir,
            config_file,
            write_video,
            view,
            main_view_frame_offset,
            match_type=match_type)
        print('Match Type:', self.match_variables.match_type)
        self.video_variables = VideoVariables(video_path)
        
        self.teams = {'top': Team('top',match_type), 'bottom': Team('bottom',match_type)}
        
        # Initialize global constants
        self.global_const = initialize_global_constants(self.match_variables,self.video_variables)
        
        # Initialize global variables
        self.global_variables = initialize_global_variables()
        
        # Create output directories
        self.match_variables.output_image_dir = create_output_paths(self.match_variables) 
        
        start_time = time.time()
        
        if view == 'top':
                
            self.match_variables.get_court_background_color_range(self.video_variables,
                                                                self.global_const)
            
            self.all_frames_with_req_num_of_players = \
                self.match_variables.get_frames_with_req_num_players_in_match(self.video_variables, 
                                                                            self.global_const)
            print('Number of frames with required number of players:', len(self.all_frames_with_req_num_of_players))
            
            # Checking Wrong match type case
            if self.match_variables.match_type == 'singles':
                all_frames_with_four_players = self.match_variables.get_frames_with_req_num_players_in_match(self.video_variables, 
                                                                                                            self.global_const,
                                                                                                            num_of_player_per_team_to_check=2)
                print('Number of frames with 4 players:', len(all_frames_with_four_players))
                
                if len(all_frames_with_four_players) > 0.3*len(self.all_frames_with_req_num_of_players):
                    print('Match type is wrong')
                    self.match_variables.match_type = 'unknown'
            
            if len(self.all_frames_with_req_num_of_players) < 10:
                print(f"Ran wrong match type, given match type is {self.match_variables.match_type}")
                self.match_variables.match_type = 'unknown' 
            
            
            self.inititalize_teams(start_time)
        
     

    def inititalize_teams(self, start_time):    
        if self.match_variables.match_type == 'doubles': 
            for team in self.teams:
                self.teams[team].check_to_do_bkg_sub(self.all_frames_with_req_num_of_players,
                                                    self.global_const,
                                                    self.video_variables.video_capture,
                                                    self.match_variables)
        
            self.all_players_color_values_each_space = get_players_mean_code_values(self.all_frames_with_req_num_of_players,
                                                                                                        self.video_variables,
                                                                                                        self.match_variables,
                                                                                                        self.global_const,
                                                                                               self.teams)
            # print(self.all_players_color_values_each_space, "space"  )                                                                         
            for team in self.teams:
                # if self.teams[team].best_space is None:
                #     print(f"\nInitializing colors for team: {team}")
                self.teams[team].assign_color_values_for_initialization(self.all_players_color_values_each_space[team],  # c_space is not needed for scaled data
                                                                        self.match_variables, 
                                                                        self.video_variables)
                print(f"Assigned colors based on scaled histograms for team {team}")
            #         self.teams[team].assign_color_values_for_initialization(self.all_players_color_values_each_space[c_space][team],
            #                                                                 c_space, 
            #                                                                 self.match_variables, 
            #                                                                 self.video_variables)
                centroid_1 = self.teams[team].players[0].color
                centroid_2 = self.teams[team].players[1].color
                
                distance_between_centroids = np.linalg.norm(np.array(centroid_1) - np.array(centroid_2))
                
            #
                print(f"Centroid 1: {self.teams[team].players[0].color}")
                print(f"Centroid 2: {self.teams[team].players[1].color}")
                # print(f"Current space: {c_space.color_space}")
                print(f"Cluster ratio for {team} is {self.teams[team].cluster_length_ratio}")
                print(f"Number of frames in same cluster for {team} is {self.teams[team].num_frames_same_cluster}")
                print(f"Distance between centroids for {team} is {distance_between_centroids}")

            setup_time = time.time()
            print('Time for setup : ', setup_time-start_time)
            if self.match_variables.write_video:
                classificaion_on_sampled_images(self.teams,
                                                self.all_players_color_values_each_space,  #changes_made
                                                self.video_variables,
                                                self.match_variables)
                
            save_player_legends_doubles(self.teams, self.match_variables,
                                self.all_players_color_values_each_space,  #changes_made,
                                self.video_variables)
            
            create_tracker_sampler_json_doubles(self.teams,self.all_players_color_values_each_space,  #changes_made
                                        self.match_variables)
            
            print('Time for execution after setup : ', time.time()-setup_time)
        elif self.match_variables.match_type == 'singles':
                
            save_player_legends_singles(self.teams, self.match_variables, self.video_variables, self.global_const)
            
            create_tracker_sampler_json_singles(self.teams, self.match_variables, self.video_variables, self.global_const)
          
    
    def get_output_paths(self):
        cur_rally_dir = os.path.join(self.match_variables.view_dir, f'rally_{self.global_variables["rally_idx"]}')
        output_tracker_path = os.path.join(cur_rally_dir, "player_json_trackers")
        output_video_path = os.path.join(cur_rally_dir, "player_video_trackers")
        self.global_const['output_video_path'] = output_video_path
        self.global_const['output_tracker_path'] = output_tracker_path
        if self.match_variables.view == 'main' and os.path.exists(output_tracker_path.replace('main', 'top')):
            self.top_view_tracker_path = output_tracker_path.replace('main', 'top')
            
    
    
    def process_video(self, start_frame, end_frame, rally_idx, write_video=False):
        """
        Processes the video to track player movements within the specified frame range.

        Args:
            start_frame (int): The starting frame number for the processing.
            end_frame (int): The ending frame number for the processing.
            rally_idx (int): The index of the rally being processed.
            write_video (bool): Flag to determine if the processed video should be saved with drawn rectangles.

        Returns:
            None
        """
        
        # Number of frames where 4 players are present
        num_frames_with_req_players = 0
        
        if rally_idx > 0:
            num_frames_with_req_players = 9
            
        # Number of frames where 4 players are missing
        missing_req_count = 0
        
        # Number of frames to check based on color
        recheck_color = int(self.video_variables.fps)
        
        done_frames = [] # to not repeat frame for output_video
        
        self.global_variables['rally_idx'] = rally_idx
        self.get_output_paths()
        
        # Initialize video writer if write_video is True
        if write_video:
            OUT = cv2.VideoWriter(f"{self.global_const['output_video_path']}/rally_{self.global_variables['rally_idx']}.mp4",
                                self.video_variables.fourcc, self.video_variables.fps, (self.video_variables.frame_width, self.video_variables.frame_height))
        
        output_json_path = f"{self.global_const['output_tracker_path']}/rally_{self.global_variables['rally_idx']}_trackers.json"
        
        # Set the current frame number to the start frame
        self.global_variables["frame_number"] = start_frame
        final_json = []
        
            
        # Loop through the frames until the end frame is reached
        self.video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.global_variables["frame_number"])
        
        while self.global_variables["frame_number"] != end_frame:
            success, frame = self.video_variables.video_capture.read()
            if not success:
                break
            
            # Draw field boundary if write_video is True
            if write_video and self.global_variables["frame_number"] not in done_frames:
                draw_polygon(frame, self.global_const["FIELD_BOUNDARY"], (0, 255, 255), 2)
                frame = draw_player_legends(frame,self.match_variables.output_image_dir, self.global_const['NUM_PLAYERS_PER_TEAM'])
                
            done_frames.append(self.global_variables["frame_number"])

            # YOLO detection to find players in the current frame
            PLAYERS_IN_VIDEO_YOLO = self.match_variables.yolo_data.get(str(self.global_variables["frame_number"]), [])
            top_player_bbox, bottom_player_bbox = get_players_inside_court(PLAYERS_IN_VIDEO_YOLO, self.global_const)
            
            # Check if there are exactly two players in both top and bottom regions of the court
            if (len(top_player_bbox) == self.global_const["NUM_PLAYERS_PER_TEAM"] and 
            len(bottom_player_bbox) == self.global_const["NUM_PLAYERS_PER_TEAM"]):
                
                num_frames_with_req_players += 1
                missing_req_count = 0
            else:
                missing_req_count += 1
                if missing_req_count > int(self.video_variables.fps) :
                    num_frames_with_req_players = 0
                
            if num_frames_with_req_players < self.global_const["COUNT_FRAMES_BEFORE_START"]:
                if write_video:
                    OUT.write(frame)
                final_json.append({"frame_number": self.global_variables["frame_number"], "player_trackers": {}})
                self.global_variables["frame_number"] += 1
                if self.global_variables["frame_number"] == end_frame:
                    break
                continue

            # Initialize player trackers if the required number of frames before start is met
            if num_frames_with_req_players == self.global_const["COUNT_FRAMES_BEFORE_START"]:
                self.global_variables["start_tracking_frame_number"] = self.global_variables["frame_number"]
            
            # Track players' movement and recheck color if needed
            if self.global_variables["frame_number"] - self.global_variables["start_tracking_frame_number"] == 0:
                recheck_color_local, self.teams, frame = track_players_movement(
                                                                    frame,
                                                                    self.teams,
                                                                    self.match_variables,
                                                                    top_player_bbox,
                                                                    bottom_player_bbox,
                                                                    self.global_const,
                                                                    initialize=True)
                recheck_color = int(self.video_variables.fps)
            else:
                recheck_color -= 1
                
                if recheck_color == 0:
                    recheck_color_local, self.teams, frame = track_players_movement(
                                                                frame,
                                                                self.teams,
                                                                self.match_variables,
                                                                top_player_bbox,
                                                                bottom_player_bbox,
                                                                self.global_const,
                                                                initialize=True)
                    recheck_color = -1
                else:
                    recheck_color_local, self.teams, frame = track_players_movement(
                                                                    frame,
                                                                    self.teams,
                                                                    self.match_variables,
                                                                    top_player_bbox,
                                                                    bottom_player_bbox,
                                                                    self.global_const,
                                                                    initialize=False)
                    
                    if recheck_color_local > 0:
                        recheck_color = int(self.video_variables.fps)
            
            # Draw rectangles around the tracked players and label them
            if write_video:
                for team in self.teams:
                    for num, player in enumerate(self.teams[team].players):
                        x1, y1, x2, y2,_ = player.tracker[-1]
                        if num == 0:
                            frame = cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f"{team}{num}", (x1, y1), 1, 1, offset=10, colorT=(255, 255, 255), colorB=(0, 0, 255), colorR=(0, 0, 255))
                        if num == 1:
                            frame = cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (0, 255, 0), 2)
                            cvzone.putTextRect(frame, f"{team}{num}", (x1, y1), 1, 1, offset=10, colorT=(255, 255, 255), colorB=(0, 0, 255), colorR=(0, 255, 0))

            # Save the player trackers data in the JSON file
            temp_dict = dict()
            if self.global_variables["frame_number"] - self.global_variables["start_tracking_frame_number"] != 0:
                for team in self.teams:
                    for num, player in enumerate(self.teams[team].players):
                        x1, y1, x2, y2,_ = player.tracker[-1]
                        temp_dict[f"{team}{num}"] = [x1, y1, x2, y2]

            final_json.append({"frame_number": self.global_variables["frame_number"], "player_trackers": temp_dict})
            if write_video:
                OUT.write(frame)
            
            self.global_variables["frame_number"] += 1

            if self.global_variables["frame_number"] == end_frame:
                break
            
        with open(output_json_path, "w") as output_file:      
            json.dump(final_json, output_file)
        
        if write_video:
            OUT.release()


    def get_trackers_from_correlation(self, rally_idx, write_video=False):
        """
        Processes the video to track player movements within the specified frame range.

        Args:
            start_frame (int): The starting frame number for the processing.
            end_frame (int): The ending frame number for the processing.
            rally_idx (int): The index of the rally being processed.
            write_video (bool): Flag to determine if the processed video should be saved with drawn rectangles.

        Returns:
            None
        """
        
        self.global_variables['rally_idx'] = rally_idx
        self.get_output_paths()
        if write_video:
            OUT = cv2.VideoWriter(f"{self.global_const['output_video_path']}/rally_{self.global_variables['rally_idx']}.mp4",
                            self.video_variables.fourcc, self.video_variables.fps, (self.video_variables.frame_width, self.video_variables.frame_height))
        
        with open(self.top_view_tracker_path + f"/rally_{self.global_variables['rally_idx']}_trackers.json", "r") as f:
            top_trackers = json.load(f)
        
        
        # Set the current frame number to the start frame
        final_json = []
        
        
        # Getting Homography
        top_H = estimate_homography(self.global_const['TOP_VIEW_TOP_POINTS'], self.global_const['MAIN_VIEW_TOP_POINTS'])
        bottom_H = estimate_homography(self.global_const['TOP_VIEW_BOTTOM_POINTS'], self.global_const['MAIN_VIEW_BOTTOM_POINTS'])
        
        # Loop through the frames until the end frame is reached
        frame_number = top_trackers[0]["frame_number"]
        self.video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES,  frame_number + self.match_variables.main_view_frame_offset)
        
        for idx, frame_data in enumerate(top_trackers):
            frame_number = frame_data["frame_number"]
            main_view_frame_number = frame_number + self.match_variables.main_view_frame_offset
                
            if write_video:
                success, frame = self.video_variables.video_capture.read()
                if not success:
                    break
                cv2.putText(frame, f'{frame_number}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.line(frame, (self.global_const['MAIN_VIEW_NET_LINE'][0][0], self.global_const['MAIN_VIEW_NET_Y_FOR_PLAYER_SEPERATION']), (self.global_const['MAIN_VIEW_NET_LINE'][1][0], self.global_const['MAIN_VIEW_NET_Y_FOR_PLAYER_SEPERATION']), (255, 255, 255), 2)
                frame = draw_polygon(frame, self.global_const['MAIN_VIEW_FIELD_BOUNDARY'], (0, 255, 255), 2)
                frame = draw_player_legends(frame,self.match_variables.output_image_dir.replace('main', 'top'), self.global_const['NUM_PLAYERS_PER_TEAM'])

            
            top_view_trackers = frame_data["player_trackers"] 
            if top_view_trackers =={}:
                final_json.append({"frame_number": main_view_frame_number, "player_trackers": {}})
                if write_video:
                    OUT.write(frame)
                continue
            
            main_view_player_trackers = {}
            
            # Get the player trackers in the main view
            for view in self.teams:
                for num, players in enumerate(self.teams[view].players):
                    player_tracker = top_view_trackers[f'{view}{num}']
                    homography_matrix = top_H if view == 'top' else bottom_H
                    main_view_player_tracker = get_main_view_player_bbox(player_tracker, homography_matrix)
                    main_view_player_trackers[f'{view}{num}'] = main_view_player_tracker
                
            
            if write_video:
                for keys in main_view_player_trackers:
                    x1, y1, x2, y2 = main_view_player_trackers[keys]
                    if '0' in keys:
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            
            # YOLO detection to find players in the current frame
            PLAYERS_IN_VIDEO_YOLO = self.match_variables.yolo_data.get(str(main_view_frame_number), [])
            for player in PLAYERS_IN_VIDEO_YOLO:
                x1, y1, x2, y2,_ = player
                if write_video:
                    frame = cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0, 255, 255), 2)
                
            
            top_player_bbox, bottom_player_bbox = get_players_inside_court(PLAYERS_IN_VIDEO_YOLO, self.global_const, self.match_variables.view)
            
            if write_video:
                for bbox in top_player_bbox:
                    frame = cv2.rectangle(frame, (bbox[0]-20, bbox[1]+20), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    frame = cv2.putText(frame, f"{bbox[4]:.2f}", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                for bbox in bottom_player_bbox:
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    frame = cv2.putText(frame, f"{bbox[4]:.2f}", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
            previous_trackers = final_json[-1]["player_trackers"]
            
            if self.match_variables.match_type == 'doubles':
                top_mapping = map_bbox_doubles(top_player_bbox,main_view_player_trackers['top0'], main_view_player_trackers['top1'], 'top', previous_trackers, frame_number)
                bottom_mapping = map_bbox_doubles(bottom_player_bbox,main_view_player_trackers['bottom0'], main_view_player_trackers['bottom1'], 'bottom', previous_trackers,frame_number)
            else:
                top_mapping = map_bbox_singles(top_player_bbox,main_view_player_trackers['top0'], 'top', previous_trackers, frame_number)
                bottom_mapping = map_bbox_singles(bottom_player_bbox,main_view_player_trackers['bottom0'], 'bottom', previous_trackers,frame_number)
            
            temp_dict = dict()
            for key in top_mapping:
                temp_dict[key] = top_mapping[key]
            for key in bottom_mapping:
                temp_dict[key] = bottom_mapping[key]
            
            if write_video:
                for key in temp_dict:
                    x1, y1, x2, y2 = temp_dict[key]
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, key, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                
                OUT.write(frame)   
                

            final_json.append({"frame_number": main_view_frame_number, "player_trackers": temp_dict})

            self.global_variables["frame_number"] += 1
            

        # Open JSON file for writing player trackers data
        with open(f"{self.global_const['output_tracker_path']}/rally_{self.global_variables['rally_idx']}_trackers.json", "w") as output_file:
            json.dump(final_json, output_file)
    
