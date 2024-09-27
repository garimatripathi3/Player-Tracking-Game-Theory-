import os
from tqdm import tqdm
import json
from players_tracking.tracker_processor import PlayerTrackerProcessor
from glob import glob
import subprocess
from pprint import pprint
def  get_player_trackers(rally_times, video_path, player_tracking_viz_flag, store_player_images,booking_dir, match_id, view, logger=None):
    """
    Processes player tracking for a given video and rally times.
    
    Args:
        rally_times (list): List of frame indices representing the start of each rally.
        video_path (str): Path to the input video file.
        player_tracking_viz_flag (bool): Flag indicating whether to write the output video with player tracking.
        view_dir (str): Path to save the output files.
        view (str): View of the video (top or cross or main).
        logger (logging.Logger, optional): Logger object for logging messages. Defaults to None.
        
    Returns:
        bool: True if processing is successful, False otherwise.
    """
    
    preprocess_dir = os.path.join(booking_dir, 'preprocessing')
    view_dir = os.path.join(booking_dir, match_id, view)
    
    homography_config_path = os.path.join(preprocess_dir, 'config.yaml')
    rally_separation_file_path = os.path.join(preprocess_dir, 'rally_separation.json')
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        if logger:
            logger.error("Video file does not exist")
        return False
    
    # Check if the configuration file exists
    if not os.path.exists(homography_config_path):
        if logger:
            logger.error("Player Tracking Config file does not exist")
        return False
    
    main_video_frame_offset = 0
    with open(rally_separation_file_path, 'r') as f:
        rally_sep_data = json.load(f)  
    
    match_type = rally_sep_data[match_id]['type']    
    
    if view != 'top':
        main_video_frame_offset = rally_sep_data['main_video_frame_offset']      
    
    # Create a PlayerTrackerProcessor object
    processor = PlayerTrackerProcessor(video_path,
                                        homography_config_path,
                                        view_dir,
                                        rally_times,
                                        view,
                                        store_player_images,
                                        main_video_frame_offset,
                                        match_type)
    
    if processor.match_variables.match_type == 'unknown':
        logger.error(f"Match type MISMATCH, provided match type is {match_type}")
        return False
    
    # Process each rally in the video
    if view != 'main':
        for idx in tqdm(range(len(rally_times[:-1])), desc="Processing Player Tracking"):
            
            start_frame = rally_times[idx]
            end_frame = rally_times[idx + 1]
            rally_idx = idx
            processor.process_video(start_frame, end_frame, rally_idx, player_tracking_viz_flag)
        
    else:
        for idx in tqdm(range(len(rally_times[:-1])), desc="Correlating Player Trackes b/w views"):
            
            start_frame = rally_times[idx]
            end_frame = rally_times[idx + 1]
            rally_idx = idx
            processor.get_trackers_from_correlation(rally_idx, player_tracking_viz_flag)
        
    if player_tracking_viz_flag:
        video_files = glob(f'{view_dir}/rally_*/player_video_trackers/rally_*.mp4')
        video_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        input_file = os.path.join(view_dir, "videos_to_combine.txt")
        with open(input_file, "w") as f:
            for video in video_files:
                f.write(f"file '{video}'\n")
        
        output_file = os.path.join(view_dir, "player_tracking_out.mp4")
        
        ffmpeg_command = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", input_file,
            "-c", "copy",
            output_file
        ]
        subprocess.run(ffmpeg_command)

        # Clean up the temporary text file
        # os.remove(input_file)
    return True
