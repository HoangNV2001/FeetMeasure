import cv2
import numpy as np

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new,new)
    return new

def get_size(width, length):

    if 6 <= width and width <  8:
        return 32
    elif 8 <= width and width < 9:
        if 16 <= length and length < 21.6:
            return 33
        elif 21.6 <= length and length < 22.1:
            return 34
        elif 22.1 <= length and length < 22.5:
            return 35
        else:
            return 36
    elif 9 <= width and width <  10:
        if 18 <= length and length < 23.6:
            return 37
        elif 23.6 <= length and length < 24.1:
            return 38
        elif 24.1 <= length and length < 24.5:
            return 39
        else:
            return 40
    elif 10 <= width and width < 14:
        if 20 <= length and length < 25.6:
            return 41
        elif 25.6 <= length and length < 26.1:
            return 42
        elif 26.1 <= length and length < 26.5:
            return 43
        else:
            return 44
    else:
        raise Exception('Sum Ting Wong')

def full_pipeline(img_dir):
    WIDTH = 480
    HEIGHT = 320

    # these constants are tuned
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    '''
    SECTION 1: CROP & TRANSFORM A4 REGION
    '''
    ################################
    ###### EDGE DETECTION ##########
    ################################
    orig = cv2.imread(img_dir)
    orig = cv2.resize(orig, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)



    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3,3), 0, img)


    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.dilate(img, kernel)

    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    for line in lines[0]:
          cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                          (255,0,0), 2, 8)

    # finding contours
    # contours, _ = cv2.findContours(edges.copy(), cv2.CV_RETR_EXTERNAL,
    #                                 cv2.CV_CHAIN_APPROX_TC89_KCOS)


    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        rects.append(rect)


    # show only contours
    new = get_new(img)
    cv2.drawContours(new, rects,-1,(0,255,0),1)
    cv2.GaussianBlur(new, (9,9), 0, new)
    new = cv2.Canny(new, 0, CANNY, apertureSize=3)

    ################################
    ##### CORNER DETECTION #########
    ################################

    dst = cv2.cornerHarris(new, 2, 3, 0.04)
    candidates = np.argwhere(dst>0.06*dst.max())
    paper_corners = [None, None, None, None]
    min_corner_dist = [np.Inf, np.Inf, np.Inf, np.Inf]
    corners = [np.array([0,0]), np.array([HEIGHT,0]), np.array([0,WIDTH]), np.array([HEIGHT, WIDTH])]
    for point in candidates:
        for i in range(4):
            dist = np.linalg.norm(point - corners[i])
            if dist < min_corner_dist[i]:
                paper_corners[i] = point
                min_corner_dist[i] = dist

    for i in range(4):
        paper_corners[i] = (paper_corners[i][1],paper_corners[i][0])

    ################################
    ##### AFFINE TRANSFORM #########
    ################################
    img = orig.copy()

    rows, cols, ch = img.shape

    pts1 = np.float32([paper_corners[0],
                      paper_corners[1],
                      paper_corners[2]])

    pts2 = np.float32([(0,0),
                      (0,HEIGHT),
                      (WIDTH,0)])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    '''
    SECTION 2: FOOT SEGMENTATION
    '''
    ################################
    ##### Remove Background  #######
    ################################

    #Use Grab cut
    img = dst.copy()
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    ################################
    ######## Skin Masking  #########
    ################################
    min_YCrCb = np.array([0,120,50],np.uint8)
    max_YCrCb = np.array([255,200,200],np.uint8)

    # Get pointer to video frames from primary device
    image = img.copy()
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)



    '''
    SECTION 3: FOOT MEASURE
    '''
    feet_gray = cv2.cvtColor(skinYCrCb, cv2.COLOR_YCrCb2BGR)
    feet_gray = cv2.cvtColor(skinYCrCb, cv2.COLOR_BGR2GRAY)
    
    foot_width = 0

    for i in range(int(HEIGHT*0.1), int(HEIGHT*0.9)):

        width = sum(feet_gray[i,:]>0)
        if width > foot_width:
            foot_width = width
            max_width_i = i

    foot_length = 0
    for i in range(int(WIDTH*0.05), int(WIDTH*0.95)):

        length = sum(feet_gray[:,i]>0)
        if length > foot_length:
            foot_length = length
            max_length_i = length

    foot_width = foot_width / WIDTH * 21.0
    foot_length = foot_length / HEIGHT * 29.7

    return foot_width, foot_length
