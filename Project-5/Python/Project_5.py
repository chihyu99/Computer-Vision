import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares

def SIFT(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    corres = []
    for match in matches:
        p1 = keypoints_1[match.queryIdx].pt
        p2 = keypoints_2[match.trainIdx].pt
        corres.append([[*p1], [*p2]])

    # Visualize the result
    img12 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:1000], img2, flags=2)

    return corres, img12

def Homography(corres, pairs_num):
    A = np.zeros((pairs_num*2, 8))
    b = np.zeros((pairs_num*2, 1))

    for i in range(pairs_num):
        x, y = corres[i][0]
        xp, yp = corres[i][1]

        A[i*2] = np.array([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        A[i*2+1] = np.array([0, 0, 0, x, y, 1, -x*yp, -y*yp])

        b[i*2] = xp
        b[i*2+1] = yp

    H = np.append(np.linalg.pinv(A)@b, [1]).reshape(3,3)

    return H

def Homography_RANSAC(corres, rand_num, N, M):
    domain_coord = np.array([row[0] for row in corres])
    domain_coord = np.insert(domain_coord, 2, 1, axis = 1)
    range_coord = np.array([row[1] for row in corres])
    range_coord = np.insert(range_coord, 2, 1, axis = 1)

    inliers_num = -1
    inliers_idx = []

    for i in range(N):
        random_index = random.sample(range(len(corres)), rand_num)
        
        corres_rand = np.array([corres[i] for i in random_index])
        H_initial = Homography(corres_rand, pairs_num = rand_num)

        domain_coord_proj = H_initial@domain_coord.T
        domain_coord_proj = (domain_coord_proj / domain_coord_proj[2, :]).T

        error = np.abs(domain_coord_proj - range_coord)
        d = np.sum(error**2, axis = 1)
        idx = np.where(d <= delta**2)[0]

        if len(idx) > inliers_num:
            inliers_idx = idx
            inliers_num = len(inliers_idx)
            H_ransac = H_initial

        if len(idx) > M:
            break

    return H_ransac, inliers_idx


def visualize_RANSAC(img_left, img_right, img_id, corres, inliers_idx):
    img_in = np.concatenate((img_left, img_right), axis = 1)
    img_out = np.concatenate((img_left, img_right), axis = 1)

    for i in range(len(corres)):
        left = tuple(np.round(corres[i][0]).astype(int))
        right = tuple(np.round(corres[i][1]).astype(int) + np.array([img_left.shape[1], 0]))
        if i in inliers_idx:
            cv2.circle(img_in, left, radius = 4, color = (0,255,0), thickness = -1)
            cv2.circle(img_in, right, radius = 4, color = (0,255,0), thickness = -1)
            cv2.line(img_in, left, right, (0,255,0), 1)
        else:
            cv2.circle(img_out, left, radius = 4, color = (0,0,255), thickness = -1)
            cv2.circle(img_out, right, radius = 4, color = (0,0,255), thickness = -1)
            cv2.line(img_out, left, right, (0,0,255), 1)

    # Save the output image
    newStr = "img" + str(img_id) + str(img_id+1) + '_RANSAC_inliers.jpg'
    cv2.imwrite("/Users/" + newStr, img_in)
    print(newStr, "saved.")
    newStr = "img" + str(img_id) + str(img_id+1) + '_RANSAC_outliers.jpg'
    cv2.imwrite("/Users/" + newStr, img_out)
    print(newStr, "saved.")

    return None


def Homography_LLS(corres, inliers_idx):
    corres_in = np.array([corres[i] for i in inliers_idx])
    n = len(corres_in)

    A = np.zeros((n*2, 8))
    b = np.zeros((n*2, 1))

    for i in range(n):
        x, y = corres_in[i][0]
        xp, yp = corres_in[i][1]

        A[i*2] = np.array([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        A[i*2+1] = np.array([0, 0, 0, x, y, 1, -x*yp, -y*yp])

        b[i*2] = xp
        b[i*2+1] = yp

    A_pseudo = np.linalg.pinv(A.T@A)@A.T
    h = A_pseudo@b
    H_LLS = np.append(h, [1]).reshape(3,3)

    return H_LLS


def cost_function(h, domain_coord, range_coord):
    H = h.reshape(3,3)

    domain_coord = np.insert(domain_coord, 2, 1, axis = 1)
    range_coord = np.insert(range_coord, 2, 1, axis = 1)

    domain_coord_proj = H@domain_coord.T
    domain_coord_proj = (domain_coord_proj / domain_coord_proj[2, :]).T

    error = np.abs(domain_coord_proj - range_coord)
    error_norm = (error**2)[:,:2]

    return error_norm.flatten()


def Homography_LM(correspondences, inliers_idx):
    domain_coord_in = np.array([correspondences[i][0] for i in inliers_idx])
    range_coord_in = np.array([correspondences[i][1] for i in inliers_idx])

    LM = least_squares(cost_function, H_LLS.flatten(), args=(domain_coord_in, range_coord_in), method = "lm")
    h_LM = np.array(LM.x)
    h_LM /= h_LM[-1]

    H_LM = h_LM.reshape(3,3)

    return H_LM


def Proj_to_pano(panorama_img, H, img_src):
    single_img_height, single_img_width, _ = img_src.shape
    x = np.repeat(np.arange(single_img_width), single_img_height)
    y = np.tile(np.arange(single_img_height), single_img_width)
    w = np.ones((single_img_height*single_img_width), dtype = int)
    xyw = np.column_stack((x,y,w))

    new_coord = H@xyw.T
    new_coord /= new_coord[2,:]
    panorama_coord = (new_coord[:2]).T + np.array([single_img_width*2, 0])

    # Remove the coordinates that are negative or out of frame
    valid_coord = (panorama_coord[:,1] > 0) & (panorama_coord[:,1] < single_img_height-1)
    panorama_coord_valid = np.round(panorama_coord[valid_coord]).astype(int)
    xy_valid = xyw[valid_coord][:,:2]

    # Project to the panorama image
    for pano, xy in zip(panorama_coord_valid, xy_valid):
        if np.all(panorama_img[pano[1], pano[0]] == [0,0,0]):
            if 0<xy[1]<single_img_height-1 and 0<xy[0]<single_img_width-1 and 0<pano[1]<single_img_height-1 and 0<pano[0]<single_img_width-1:
                panorama_img[pano[1]-1:pano[1]+1, pano[0]-1:pano[0]+1] = img_src[xy[1]-1:xy[1]+1, xy[0]-1:xy[0]+1]
            else:
                panorama_img[pano[1]-1:pano[1]+1, pano[0]-1:pano[0]+1] = img_src[xy[1], xy[0]]

    return None


def panorama_generate(images_num, H_refine_all):
    # Load images
    image_set = {}
    for i in range(images_num):
        image_set['img%d' %i] = cv2.imread("/Users/%d.jpg"%(i))

    # Define the panorama image
    single_img_height, single_img_width, _ = image_set['img0'].shape
    panorama_height, panorama_width = single_img_height, single_img_width*5
    panorama_img = np.zeros((panorama_height, panorama_width, 3), dtype = np.uint8)

    mid = images_num//2
    panorama_img[:, single_img_width*2:single_img_width*3] = image_set['img%d' %mid]     # Set the middle image

    # Project all images to one
    Proj_to_pano(panorama_img, H = H_refine_all[1], img_src = image_set['img1'])
    Proj_to_pano(panorama_img, H = H_refine_all[0]@H_refine_all[1], img_src = image_set['img0'])
    Proj_to_pano(panorama_img, H = np.linalg.inv(H_refine_all[2]), img_src = image_set['img3'])
    Proj_to_pano(panorama_img, H = np.linalg.inv(H_refine_all[2]@H_refine_all[3]), img_src = image_set['img4'])

    return panorama_img


if __name__ == '__main__':

    ###### Automatically estimating homographies #####
    # Outlier rejection using the RANSAC algorithm
    epsilon = 0.6
    p = 0.99
    n = 5                                                          # Number of correspondences to choose in each trial
    N = np.ceil(np.log(1-p)/np.log(1-(1-epsilon)**n)).astype(int)  # Number of trials
    delta = 5                                                      # 3*sigma (sigma is set to a small number between 0.5 and 2)

    H_LLS_all = []
    H_LM_all = []
    for i in range(4):
        img1 = cv2.imread("/Users/%d.jpg"%(i))
        img2 = cv2.imread("/Users/%d.jpg"%(i+1))

        correspondences, img_SIFT = SIFT(img1, img2)

        # Save the output image
        newStr = "img%d%d"%(i, i+1) + '_SIFT' + '.jpeg'
        cv2.imwrite("/Users/" + newStr, img_SIFT)
        print(newStr, "saved.")

        n_total = len(correspondences)                              # Total number of correspondences
        M = np.ceil((1-epsilon)*n_total).astype(int)                # Number of inliers in the data

        H_ransac, inliers_idx = Homography_RANSAC(corres = correspondences, rand_num = n, N = N, M = M)
        visualize_RANSAC(img_left = img1, img_right = img2, img_id = i, corres = correspondences, inliers_idx = inliers_idx)

        H_LLS = Homography_LLS(correspondences, inliers_idx)        # Homography estimation using a Linear Least-Squares method
        H_LM = Homography_LM(correspondences, inliers_idx)          # Homography refinement using a Nonlinear Least-Squares approach

        H_LLS_all.append(H_LLS)
        H_LM_all.append(H_LM)
    print("Task 1.4 Done")
    
    #### Project all the views on to a fixed common frame #####
    output_LLS = panorama_generate(images_num = 5, H_refine_all = H_LLS_all)
    cv2.imwrite("Panorama_image_LLS.jpg", output_LLS)
    print("Panorama_image_LLS.jpg saved.")
    output_LM = panorama_generate(images_num = 5, H_refine_all = H_LM_all)
    cv2.imwrite("Panorama_image_LM.jpg", output_LM)
    print("Panorama_image_LM.jpg saved.")

    print("Task 1.5 Done")

