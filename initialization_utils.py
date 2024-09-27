import cv2
from utils import load_config
from players_tracking.homography_utils import calculate_homography_matrices
import ast
import os
from tqdm import tqdm
import json


class VideoVariables():
    def __init__(self,input_video_path):
        
        VIDEOCAPTURE = cv2.VideoCapture(input_video_path)
        self.input_video_path = input_video_path
        self.video_capture = VIDEOCAPTURE
        self.total_frames = int(VIDEOCAPTURE.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(VIDEOCAPTURE.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(VIDEOCAPTURE.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = VIDEOCAPTURE.get(cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        

    
        
def initialize_global_constants(match_variables,
                                video_variables):
    """
    Initializes the global constants required for player tracking.

    Returns:
        dict: Dictionary containing global constants.
    """
    global_const = {
        "INPUT_VIDEO": video_variables.input_video_path, 
        "NUM_PLAYERS_PER_TEAM": match_variables.num_players_per_team,
        "START_FRAME": 0,
        "YOLO_FRAMES_SKIP": match_variables.config['YOLO_FRAMES_SKIP'],
        "COURT_DEFINING_POINTS": None,
        "FIELD_BOUNDARY": None,
        "NET_LINE": None,
        "COLOR_MATCH_THRESHOLD": match_variables.config['COLOR_MATCH_THRESHOLD'],
        "COUNT_FRAMES_BEFORE_START": match_variables.config['COUNT_FRAMES_BEFORE_START'],
        "COLOUR_TO_D_THRESHOLD": match_variables.config['COLOUR_TO_D_THRESHOLD'],
        "NET_Y_FOR_PLAYER_SEPERATION": match_variables.config["COURT_MIDDLE_Y"],
        "COURT_CENTER_X": match_variables.config["COURT_CENTER_X"],
        "MAIN_VIEW_COURT_CENTER_X": match_variables.config["MAIN_VIEW_COURT_CENTER_X"],
        "MAIN_VIEW_NET_Y_FOR_PLAYER_SEPERATION": match_variables.config["MAIN_VIEW_COURT_MIDDLE_Y"]
    }
    
    # Court defining points = [top_left, top_right, net_left, net_right, bottom_left,bottom_right]
    COURT_DEFINING_POINTS_str = (match_variables.config['COURT_DEFINING_POINTS'])
    global_const["COURT_DEFINING_POINTS"] = ast.literal_eval(COURT_DEFINING_POINTS_str)
    MAIN_VIEW_COURT_DEFINING_POINTS_str = match_variables.config['MAIN_VIEW_COURT_DEFINING_POINTS']
    global_const["MAIN_VIEW_COURT_DEFINING_POINTS"] = ast.literal_eval(MAIN_VIEW_COURT_DEFINING_POINTS_str)
    
    # Field boundary = (net_left_x, top_y), (net_right_x, bottom_y)
    global_const["FIELD_BOUNDARY"] = [
        [global_const["COURT_DEFINING_POINTS"][0][0], global_const["COURT_DEFINING_POINTS"][0][1]],
        [global_const["COURT_DEFINING_POINTS"][1][0], global_const["COURT_DEFINING_POINTS"][1][1]],
        [global_const["COURT_DEFINING_POINTS"][5][0], global_const["COURT_DEFINING_POINTS"][5][1]],
        [global_const["COURT_DEFINING_POINTS"][4][0], global_const["COURT_DEFINING_POINTS"][4][1]]
    ]
    
    global_const["MAIN_VIEW_FIELD_BOUNDARY"] = [
        [global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][0][0], global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][0][1]],
        [global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][1][0], global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][1][1]],
        [global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][5][0], global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][5][1]],
        [global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][4][0], global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][4][1]]
    ]
    
    # Net line = [net_left, net_right]
    global_const["NET_LINE"] = [
        global_const["COURT_DEFINING_POINTS"][2], 
        global_const["COURT_DEFINING_POINTS"][3]
    ]
    
    global_const["MAIN_VIEW_NET_LINE"] = [
        global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][2], 
        global_const["MAIN_VIEW_COURT_DEFINING_POINTS"][3]
    ]

    global_const['TOP_VIEW_TOP_POINTS'] = match_variables.config['TOP_VIEW_TOP_POINTS']
    global_const['TOP_VIEW_BOTTOM_POINTS'] = match_variables.config['TOP_VIEW_BOTTOM_POINTS']
    global_const['MAIN_VIEW_TOP_POINTS'] = match_variables.config['MAIN_VIEW_TOP_POINTS']
    global_const['MAIN_VIEW_BOTTOM_POINTS'] = match_variables.config['MAIN_VIEW_BOTTOM_POINTS']
    
    
    iH1, iH2 = calculate_homography_matrices()
    global_const["i_H_bottom"] = iH2
    global_const["i_H_top"] = iH1
    
    return global_const

def initialize_global_variables():
        """
        Initializes the global variables required for player tracking.

        Returns:
            dict: Dictionary containing global variables.
        """
        global_variables = {
            "frame_number": 0,
            "rally_idx": 0,
        }
        return global_variables


def create_output_paths(match_variables):
    """
    Creates the output directories for trackers and video files.
    """
    
    for rally_idx in range(len(match_variables.rally_times)-1):
        
        cur_rally_dir = os.path.join(match_variables.view_dir, f'rally_{rally_idx}')
        output_tracker_path = os.path.join(cur_rally_dir, "player_json_trackers")
        output_video_path = os.path.join(cur_rally_dir, "player_video_trackers")
        
        if not os.path.exists(output_tracker_path):
            os.makedirs(output_tracker_path)
    
        if os.path.exists(output_video_path):
            os.system(f'rm -rf {output_video_path}')
        
        if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)
    
    # Create output directory for storing the images if they don't exist
    output_image_path = os.path.join(match_variables.view_dir, "player_images")
    
    if match_variables.write_video:
        
        if os.path.exists(output_image_path):
            os.system(f'rm -rf {output_image_path}')
            print('Deleted existing player_images folder')
    
        if not os.path.exists(output_image_path) and match_variables.match_type == 'doubles':
            os.makedirs(output_image_path)  
            for key in ['top', 'bottom']:
                for clusters in ['c1','c2']:
                    os.makedirs(os.path.join(output_image_path, f'{key}_{clusters}'))
    
    return output_image_path
