import cv2
import numpy as np
import math
import os
import pickle

def estimate_homography(pts1, pts2):
    """
    Estimate the homography matrix between two sets of points.
    
    Args:
        pts1 (list): List of 2D points in the first image.
        pts2 (list): List of 2D points in the second image.
    
    Returns:
        numpy.ndarray: The homography matrix.
    """
    assert len(pts1) == len(pts2), "Number of points must match"
    
    # Convert points to numpy arrays
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Estimate homography matrix
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    
    return H


## def estimate_undist_homography(pts1, pts2):


# def estimate_undist_2d_point(point, H):

def undistort_points(points, centre_idx, court_idx, view, data_path):
    camera_objects_path = os.path.join(data_path, f"Centre_{centre_idx}/Court_{court_idx}")
    if not os.path.exists(os.path.join(camera_objects_path, f'{view}/images/{view[:-5]}_camera_object.pkl')):
        # logger.error(f"Camera object for top view not found in {camera_objects_path}")
        print(f"Camera object for top view not found in {camera_objects_path}")
        return
    
    cam = pickle.load(
        open(os.path.join(camera_objects_path, f'{view}/images/top_camera_object.pkl'), 'rb'))
    undist_points = cv2.undistortPoints(points, cam.camera_matrix, cam.distortion_coefficients, None, cam.new_camera_matrix)

    return undist_points

def undistort_image(frame, centre_idx, court_idx, view, data_path):
    camera_objects_path = os.path.join(data_path, f"Centre_{centre_idx}/Court_{court_idx}")
    if not os.path.exists(os.path.join(camera_objects_path, f'{view}/images/{view[:-5]}_camera_object.pkl')):
        # logger.error(f"Camera object for top view not found in {camera_objects_path}")
        print(f"Camera object for top view not found in {camera_objects_path}")
        return
    
    cam = pickle.load(
        open(os.path.join(camera_objects_path, f'{view}/images/top_camera_object.pkl'), 'rb'))
    
    undist_image = cv2.undistort(frame, cam.camera_matrix, cam.distortion_coefficients, None, cam.new_camera_matrix)
    return undist_image

def estimate_2d_point(point, H):
    """
    Estimate the 2D point in the image plane using the homography matrix.
    
    Args:
        point (tuple): The 2D point in the world coordinate system.
        H (numpy.ndarray): The homography matrix.
    
    Returns:
        numpy.ndarray: The estimated 2D point in the image plane.
    """
    p_prime = np.dot(H, np.array([point[0], point[1], 1]))
    p_prime = p_prime / p_prime[2]
    return p_prime

def calculate_inverse_homography(pts1, pts2):
    """
    Calculate the inverse homography matrix.
    
    Args:
        pts1 (list): List of 2D points in the first image.
        pts2 (list): List of 2D points in the second image.
    
    Returns:
        numpy.ndarray: The inverse homography matrix.
    """
    H = estimate_homography(pts1, pts2)
    iH = np.linalg.inv(H)
    return iH

def calculate_homography_matrices(image_points=[(92, 171), (563, 187), (515, 1168), (90, 1157)],
                                  net_points=[(37, 696), (567, 712)],
                                  world_points=[(0, 0), (610, 0), (610, 1340), (0, 1340)],
                                  world_net_points=[(0, 670), (610, 670)],
                                  pad_distance_x=75,
                                  pad_distance_y=50):
    """
    Calculate the inverse homography matrices iH1 and iH2.
    
    Args:
        image_points (Array): An Array of Tuples of 2D points in the image plane.
        net_points (Array): An Array of Tuples of 2D points on the net in the image plane.
        world_points (Array): An Array of Tuples of 3D points in the world coordinate system.
        world_net_points (Array): An Array of Tuples of 3D points on the net in the world coordinate system.
        pad_distance_x (int): Padding distance in the x-direction.
        pad_distance_y (int): Padding distance in the y-direction.
    
    Returns:
        numpy.ndarray: The inverse homography matrix iH1.
        numpy.ndarray: The inverse homography matrix iH2.
    """
    # Use the top 2 points to get the homography
    top_img_points = image_points[:2]
    net_img_points = net_points
    
    top_world_points = world_points[:2]
    net_world_points = world_net_points
    
    # print("Top World Points", top_world_points)
    # print("Net World Points", net_world_points)

    H = estimate_homography(top_world_points + net_world_points, top_img_points + net_img_points)
    iH1 = calculate_inverse_homography(top_img_points + net_img_points, top_world_points + net_world_points)
    
    # Use the bottom 2 points to get the homography
    bot_img_points = image_points[2:]
    net_img_points = net_points
    
    bot_world_points = world_points[2:]
    net_world_points = world_net_points
    
    H = estimate_homography(bot_world_points + net_world_points, bot_img_points + net_img_points)
    iH2 = calculate_inverse_homography(bot_img_points + net_img_points, bot_world_points + net_world_points)
    
    return iH1, iH2

def get_main_view_player_bbox(bbox, homography):
    
    x1, y1, x2, y2 = bbox
    
    p1 = estimate_2d_point((x1, y1), homography)
    p2 = estimate_2d_point((x2, y2), homography)
    return (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

def calculate_distance_in_world_frame(players, H, NUM_PLAYERS_PER_TEAM):
    coordinates = []
    for i in range(NUM_PLAYERS_PER_TEAM):
        player_prev_box = players[i].tracker[-1]
        player_prev_coordinate = ((player_prev_box[0] + player_prev_box[2]) // 2, (player_prev_box[3]) )
        world_coordinate = estimate_2d_point(player_prev_coordinate, H)
        coordinates.append(world_coordinate)
    
    
    # Calculate the distance between the players in the world frame
    distance = math.sqrt((coordinates[0][0] - coordinates[1][0])**2 + (coordinates[0][1] - coordinates[1][1])**2)
    return distance