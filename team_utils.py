import cv2
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from players_tracking.utils import get_players_inside_court
from players_tracking.cspace_utils import *
from scipy.spatial.distance import euclidean
class  Player():
    def __init__(self):
        self.color = None
        self.tracker = []
        self.closest_index = None
        self.closest_4_indices = None
        
class Team():
    def __init__(self, side, match_type):
        
        self.best_space = None
        self.closest_indices = None
        self.closest_4_indices =  None
        self.num_frames_same_cluster = 0 
        self.cluster_length_ratio = None
        self.ignore_color_for_tracking = False
        self.to_do_mask =  None
        self.side = side
        
        self.players = self.initialize_players(match_type)
        
    def initialize_players(self, match_type):
        if match_type == 'doubles':
            return [Player(), Player()]
        else:
            return [Player()]
        
    def check_to_do_bkg_sub(self, frames_with_4_players, global_const, VIDEOCAPTURE, match_variables):
    
        to_do_sub_cnt = 0
        
        yolo_data = match_variables.yolo_data
        
        
        for frame_num in frames_with_4_players:
                    
            VIDEOCAPTURE.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = VIDEOCAPTURE.read()
            
            YOLO_BOXES_FOR_CURR_FRAME = yolo_data.get(str(frame_num), [])
            top_player_bbox, bottom_player_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)
        
            NUM_PLAYERS_PER_TEAM = global_const["NUM_PLAYERS_PER_TEAM"]
            
            bkg_color = match_variables.bkg_color_range    
            bkg_l_start,bkg_l_end = bkg_color[0]
            bkg_a_start,bkg_a_end = bkg_color[1]
            bkg_b_start,bkg_b_end = bkg_color[2]
            
            boxes = top_player_bbox if self.side == 'top' else bottom_player_bbox

            for i in range(NUM_PLAYERS_PER_TEAM): 
                    
                if len(boxes) > i:
                    x1, y1, x2, y2,_ = boxes[i]
                    
                    if self.side == "top":
                        shirt_roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                    else:
                        shirt_roi = frame[y1 + (y2 - y1) // 2:y2, x1:x2]
                        
                    shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
                    
                    bkg_mask_l = (shirt_roi_lab_color[:, :, 0] > bkg_l_start) & (shirt_roi_lab_color[:, :, 0] < bkg_l_end)
                    bkg_mask_a = (shirt_roi_lab_color[:, :, 1] > bkg_a_start) & (shirt_roi_lab_color[:, :, 1] < bkg_a_end)
                    bkg_mask_b = (shirt_roi_lab_color[:, :, 2] > bkg_b_start) & (shirt_roi_lab_color[:, :, 2] < bkg_b_end)
                    bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
                    
                    player_pixel_ratio = np.count_nonzero(bkg_mask == 0) / bkg_mask.size
                    
                    if player_pixel_ratio > 0.1 and player_pixel_ratio < 0.9:
                        to_do_sub_cnt += 1
        
        total_number_of_frames = len(frames_with_4_players)*2

        print(f'BKG subs for {self.side}',to_do_sub_cnt, total_number_of_frames, to_do_sub_cnt/total_number_of_frames)
        
        if to_do_sub_cnt/total_number_of_frames > 0.8:
            self.to_do_mask = True
            return True
        
     
        self.to_do_mask = False
        return False 

    def assign_color_values_for_initialization(self, color_code_values,  match_variables, video_variables):
    # Print the overall size of the color_code_values object
        size = len(color_code_values) if isinstance(color_code_values, (list, dict)) else "Unknown type"
        kmeans_col = KMeans(n_clusters=match_variables.num_players_per_team, random_state=42)
        data_col = np.array([col[0] for col in color_code_values])
        # print(data_col,"data_col")
        
        kmeans_col.fit(data_col)
        centroids = kmeans_col.cluster_centers_
        labels = kmeans_col.labels_
        cluster_0_indices = np.where(labels == 0)[0]
        cluster_1_indices = np.where(labels == 1)[0]
        
        # Calculating cluster ratio
        if len(cluster_0_indices) > len(cluster_1_indices):
            cluster_ratio = len(cluster_0_indices) / len(labels)
        else:
            cluster_ratio = len(cluster_1_indices) / len(labels)
        #Taking max cluster ratio 
        self.cluster_length_ratio = cluster_ratio
        

        #
        cluster_1_colors = data_col[cluster_0_indices]
        cluster_2_colors = data_col[cluster_1_indices]
        num_same_cluster = 0
        #sorted according to frame number 
        frame_sorted_color_objects = sorted(color_code_values, key=lambda x: x[2])
        
        for col_idx in range(0,len(frame_sorted_color_objects),2):
            
            if col_idx + 1 >= len(frame_sorted_color_objects):
                break
            
            col_obj_1 = frame_sorted_color_objects[col_idx]
            col_obj_2 = frame_sorted_color_objects[col_idx+1]
            print(col_obj_1, col_obj_2)

            print(col_obj_1[2], col_obj_2[2], "first player", "second player")

            if col_obj_1[2]!=col_obj_2[2]:
                print('Frame number mismatch',frame_sorted_color_objects[col_idx],frame_sorted_color_objects[col_idx+1])
           #original approach for colour space 
           
            # else:
            #     #we can update this logic 
            #     if col_obj_1[0] in cluster_1_colors and col_obj_2[0] in cluster_1_colors:
            #         num_same_cluster+=1
            #     elif col_obj_1[0] in cluster_2_colors and col_obj_2[0] in cluster_2_colors:
            #         num_same_cluster+=1

          #changes made by Garima for 512 features vectors           
        threshold = np.std(data_col) * 0.5 
        dist1 = euclidean(col_obj_1[0], centroids[labels[col_idx]])
        dist2 = euclidean(col_obj_2[0], centroids[labels[col_idx + 1]])
        if dist1 < threshold and dist2 < threshold and labels[col_idx] == labels[col_idx + 1]:
            num_same_cluster += 1

        self.num_frames_same_cluster = num_same_cluster  
         
        
        # Calculating closest indices
        for i, centroid in enumerate(centroids):
            distances = np.linalg.norm(data_col - centroid, axis=1)
            closest_idx = np.argmin(distances)
            self.players[i].closest_index = closest_idx
            
            # Find the indices of the closest 4 points to each centroid
            closest_4_indices_list = np.argsort(distances)[:4]
            self.players[i].closest_4_indices = closest_4_indices_list.tolist()
          
        # Assigning the color values to the players
        for num in range(len(centroids)):  
            self.players[num].color =centroids[num]
    
        if match_variables.write_video:
            # breakpoint()
            for idx in range(0,len(color_code_values),2):
                
                if idx + 1 >= len(color_code_values):
                    continue
                
                frame_no = color_code_values[idx][2] ## at idx 2 we have the frame number 
                
                if frame_no != color_code_values[idx][2]:
                    print("Frame number mismatch")
                    
                bbox_1 = color_code_values[idx][1]
                bbox_2= color_code_values[idx+1][1]
                
                x1, y1, x2, y2 = bbox_1
                x3, y3, x4, y4 = bbox_2
                
                video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = video_variables.video_capture.read()
                
                img_1 = frame[y1:y2, x1:x2]
                img_2 = frame[y3:y4, x3:x4]
                
                if self.side == "top":
                    img_1_cropped = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                    img_2_cropped = frame[y3:y3 + (y4 - y3) // 2, x3:x4]
                    
                else:
                    img_1_cropped = frame[y1 + (y2 - y1) // 2:y2, x1:x2]
                    img_2_cropped = frame[y3 + (y4 - y3) // 2:y4, x3:x4]
                    
                img1_player_mask = get_player_without_bkg(match_variables, img_1_cropped)
                img2_player_mask = get_player_without_bkg(match_variables, img_2_cropped)
                
                img_1_k_means_class = 0 if idx in cluster_0_indices else 1
                img_2_k_means_class = 0 if idx+1 in cluster_0_indices else 1
                color_val_1 = color_code_values[idx][0]
                color_val_2 = color_code_values[idx+1][0]
                
                out_path = match_variables.output_image_dir
                # color_space = c_space.color_space
                os.makedirs(f"{out_path}/{self.side}_kmeans_c1", exist_ok=True)
                os.makedirs(f"{out_path}/{self.side}_kmeans_c2", exist_ok=True)
                
                if img_1_k_means_class == 0:
                        cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}.jpg", img_1)
                        cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}_cropped.jpg", img_1_cropped)
                        cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}_mask.jpg", img1_player_mask)
                        
                else:
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}.jpg", img_1)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}_cropped.jpg", img_1_cropped)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_1[0])}_{int(color_val_1[1])}_{int(color_val_1[2])}_mask.jpg", img1_player_mask)
                
                if img_2_k_means_class == 0:
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}.jpg", img_2)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}_cropped.jpg", img_2_cropped)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c1/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}_mask.jpg", img2_player_mask)
                else:
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}.jpg", img_2)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}_cropped.jpg", img_2_cropped)
                    cv2.imwrite(f"{out_path}/{self.side}_kmeans_c2/{frame_no}_{int(color_val_2[0])}_{int(color_val_2[1])}_{int(color_val_2[2])}_mask.jpg", img2_player_mask)
                        
        
                        
        

    
    
    
    

        