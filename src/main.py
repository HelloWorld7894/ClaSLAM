import cv2
import numpy as np
import math

def K_Clustering(image, n_clusters):
    """
    IN: BGR image, K (number of clusters)
    OUT: BGR image
    """

    Reshape = image.reshape((-1,3))
    Reshape = np.float32(Reshape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Reshape, n_clusters, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_res = res.reshape((image.shape))

    return img_res

def HoughLines(image_in, image_out, l_color):
    """
    IN: binary image
    OUT: BGR image
    """

    lines = cv2.HoughLines(image_in, 1, np.pi / 180, 135, None, 0, 0)
    line_points = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) #x, y notation
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image_out, pt1, pt2, l_color, 3, cv2.LINE_AA)

            line_points.append([pt1, pt2])

    return image_out, line_points

def Adjust_Gamma(image, gamma):
    """
    IN: BGR image, gamma
    OUT: BGR image with different Gamma
    """

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def Automatic_Thresh(gray_image): #https://stackoverflow.com/questions/41893029/opencv-canny-edge-detection-not-working-properly
    """
    IN: grayscale image
    OUT: lower, upper threshold (for edge detection and thresholding)
    """

    v = np.median(gray_image)
    sigma = 0.33

    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    return lower, upper

def Detect_Corners(image_in, image_out):
    """
    IN: grayscale image
    OUT: binary image
    """

    image_in = np.float32(image_in)
    corners = cv2.cornerHarris(image_in, 2, 5, 0.07)
  
    # Results are marked through the dilated corners
    corners = cv2.dilate(corners, None)

    image_out[corners > 0.01 * corners.max()] = 255
    return image_out

def Slice_image(image_in, strides):
    """
    IN: binary image, stride (y, x)
    OUT: sliced binary image
    """

    sliceY = math.ceil(image_in.shape[0] / strides[0])
    sliceX = math.ceil(image_in.shape[1] / strides[1])

    for iy in range(strides[0]): cv2.line(image_in, (iy * sliceY, 0), (iy * sliceY, image_in.shape[0]), 0, 3)
    for ix in range(strides[1]): cv2.line(image_in, (0, ix * sliceX), (image_in.shape[1], ix * sliceX), 0, 3)

    return image_in

def CalculateAngle(line):
    """
    calculate angle of one line

    IN: line input from houghlines (i guess)
    OUT: angle
    """

    d_x = line[0][0] - line[1][0]
    d_y = line[0][1] - line[1][1]

    deg = round(math.degrees(math.atan(abs(d_x) / abs(d_y))), 2)

    if d_x < 0 or d_y < 0:
        #is negative and so the actual angle is also negative
        line.append(-deg)
    else: line.append(deg)

def GroupLines(lines):
    pass #TODO: dodělat!

def WeightedArrays(arrays, mask = []):
    """
    sum two or more arrays together, if mask is not specified, it will be just a sum :)
    IN: list(arrays), mask(len=len(arrays))
    OUT: sum
    """
    
    if len(mask) == 0: mask = [1] * len(arrays)

    Sum = [0, 0, 0]

    for i in range(3):
        for i_arr in range(len(arrays)):
            Sum[i] += mask[i] * arrays[i_arr][i]

        Sum[i] = int(Sum[i] / 3)

    return Sum
            
            

#main code

vid = cv2.VideoCapture(0)
while(True):
      
    ret, frame = vid.read()

    #creating blank images and other image manipulation
    blank = np.zeros(frame.shape)
    corners = np.zeros(frame.shape[:2], dtype=np.uint8)

    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    #
    # Generating Canny, Harris and K-Means
    #

    #clustering
    img = Adjust_Gamma(frame, 0.4)
    img_cluster = K_Clustering(frame, 7) #TODO: maybe

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #edge detection
    upper, lower = Automatic_Thresh(img_gray)
    img_canny = cv2.Canny(img_gray, lower, upper)

    img_canny = Slice_image(img_canny, (5, 5))

    #corner detection
    corners = Detect_Corners(img_gray, corners)

    #
    # Fitting lines through interest points
    #

    blank, lines = HoughLines(img_canny, blank, (0, 0, 255))

    #
    # Line grouping by difference of angle and distance
    #

    for line in lines:
        CalculateAngle(line)

    max_angle_diff = 7.5
    KeyFunc = lambda elem : elem[2]

    lines.sort(key=KeyFunc)

    
    
        
    
    print(lines)

    #
    # Cluster RGB thresholding
    #
    cluster_pix = img_cluster.flatten()
    Range_Slice = int(cluster_pix.shape[0] / 3)
    min_vals = []
    target_arr = []
    
    #sort on Blue
    Arr_B = np.zeros(Range_Slice, dtype=np.uint8)
    for i in range(Range_Slice):
        i_B = i * 3
        Arr_B[i] = cluster_pix[i_B]

    B_indices = np.argsort(Arr_B, axis=0)
    B_min = np.where(B_indices==0)

    #sort on Green
    Arr_G = np.zeros(Range_Slice, dtype=np.uint8)
    for i in range(Range_Slice):
        i_G = i * 3 + 1
        Arr_G[i] = cluster_pix[i_B]

    G_indices = np.argsort(Arr_G, axis=0)
    G_min = np.where(G_indices==0)

    #sort on Red
    Arr_R = np.zeros(Range_Slice, dtype=np.uint8)
    for i in range(Range_Slice):
        i_R = i * 3 + 2
        Arr_R[i] = cluster_pix[i_B]

    R_indices = np.argsort(Arr_R, axis=0)
    R_min = np.where(R_indices==0) #returns index of minimal value (0)

    # TODO: not working properly (REVIEW)
    # Retrieve 3 lowest BGR intensities (B, G, R) - then, my really cool algorithm will choose
    # (this is going to be abomination)
    min_vals.extend([B_min, G_min, R_min])
    target_arr.extend([Arr_B, Arr_G, Arr_R])

    Pix_calc = [lambda x: (x+1, x+2), lambda x: (x+1, x-1), lambda x: (x-2, x-1)]

    min_pixels = []
    for i in range(3):
        a, b = Pix_calc[i](i)

        B = target_arr[i][min_vals[i][0][0]]
        G = target_arr[a][min_vals[i][0][0]+1]
        R = target_arr[b][min_vals[i][0][0]+2]

        min_pixels.append([B, G, R])

    print(min_pixels)

    #
    # weighted mean out of these 3 pixel intensities
    #

    Sum = WeightedArrays(min_pixels)
    
    #
    # RGB thresholding
    #
    LowerBound = min_pixels[0]
    print(LowerBound)
    UpperBound = [255, 255, 255]

    thresh = cv2.inRange(img_cluster, np.array(LowerBound, dtype=np.uint8), np.array(UpperBound, dtype=np.uint8))
    img_thresh = cv2.bitwise_and(thresh, thresh)

    #
    # Apply Blur to get better results
    #
    kernel = np.ones((5,5),np.float32) / 25
    img_thresh = cv2.filter2D(img_thresh, -1, kernel)

    #cv2.imshow("frame", frame)
    cv2.imshow("frame_canny", img_canny)
    cv2.imshow("frame_corners", img_thresh)
    cv2.imshow("frame_lines", blank)
    cv2.imshow("frame_K_means", img_cluster)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break