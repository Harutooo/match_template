'''
hqauto: 
Description: 
version: 
Author: Xiaoxia Liu.
Date: 2022-09-09 09:55:42
LastEditors: Please set LastEditors
LastEditTime: 2022-09-10 19:31:45
'''

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sqlalchemy import true


def get_match_confidence(img1, img2, mask=None):
    if img1.shape != img2.shape:
        return False
    if mask is not None:
        img1 = img1.copy()
        img1[mask!=0] = 0
        img2 = img2.copy()
        img2[mask!=0] = 0
    ## using match
    match = cv.matchTemplate(img1, img2, cv.TM_CCOEFF_NORMED)
    _, confidence, _, _ = cv.minMaxLoc(match)
    print( confidence)
    return confidence

def _mini_match(rgb_img, temp, h, w):
    score = 0
    cmp_pixel = []
    array = np.zeros((3), int)
    for i in range(h):
        for j in range(w):
            if not np.array_equal(array,temp[i][j]):
                cmp_pixel.append((i,j))

    temp_b = temp[:,:,0]
    temp_g = temp[:,:,1]    
    temp_r = temp[:,:,2]

    img_b = rgb_img[:,:,0]
    img_g = rgb_img[:,:,1]    
    img_r = rgb_img[:,:,2]

    # for i in range(h):
    #     for j in range(w):
    #         if 






def _match(rgb_img, template, stride):
    template_h, template_w = template.shape[:2]
    rgb_img_h, rgb_img_w = rgb_img.shape[:2]
    h_edge, w_edge = rgb_img_h - template_h, rgb_img_w - template_w

    i = 0
    j = 0
    # while i < h_edge:
    #     while j < w_edge:

def _slide_match(rgb_img, kernel, stride, template):
    h, w = rgb_img.shape[:2]                                                    # 
    limit_h, limit_w = (h - kernel) // stride[0], (w - kernel) // stride[1]
    for x in range(limit_w):
        for y in range(limit_h):
            dst = rgb_img[y * stride[0] :y * stride[0] + kernel, x * stride[1] :x * stride[1] + kernel]
            # _plt_show(dst)
            matchTemplate(dst, template)
        dst = rgb_img[h - kernel : h, x * stride[1] :x * stride[1] + kernel]
        # _plt_show(dst)
        matchTemplate(dst, template)
    dst = rgb_img[h - kernel : h, w - kernel : w]
    # _plt_show(dst)
    matchTemplate(dst, template)

def _plt_show(img_list):
    img1, img2 = img_list
    plt.subplot(121)
    plt.imshow(img1[:,:,[2,1,0]])
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img2[:,:,[2,1,0]])
    plt.axis('off')
    plt.show()

def _resize(img_path, fx, fy):
    try:
        img = Image.open(img_path)
    except IOError:
        return
    width, height = img.size
    out_width  = round(width  * fx)
    out_height = round(height * fy)
    out_image  = img.resize((out_width, out_height))
    return out_image

def matchTemplate(rgb_img, template):
    th, tw = template.shape[:2]
    # manner = ['cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'CV.TM_CCORR', 'CV.TM_CCOEFF', 'CV.TM_SQDIFF_NORMED', 'CV_TM_CCOEFF_NORMED']
    # manner = ['cv.TM_CCORR_NORMED']
    manner = ['cv.TM_SQDIFF_NORMED']
    manner_num = [0] # CV_TM_CCOEFF_NORMED cv.TM_CCORR_NORMED cv.TM_SQDIFF
    for m in manner_num:
        print('manner : ' + manner[m])
        result = cv.matchTemplate(rgb_img, template, m)
        min, max, minLoc, maxLoc = cv.minMaxLoc(result)             # min = 26647042.0, max = 85258480.0
        print('min = {}, max = {}'.format(min, max))                # min = 22971724.0, max = 76488072.0
        # print(result)
        img_copy = rgb_img.copy()

        # loc = np.where(result >= 0.9)
        # print(loc)
        # for pt in zip(*loc[::-1]):
        #     img_copy = rgb_img.copy()
        #     cv.rectangle(img_copy, pt, (pt[0] + tw, pt[1] + th), (0, 255, 0), 3)

        #     show_list = [template, img_copy]
        #     _plt_show(show_list)

        # if min <= 22971724 and max >= 76488072:
        topLeft = minLoc
        bottomRight = (topLeft[0] + tw, topLeft[1] + th)
        
        cv.rectangle(img_copy, topLeft, bottomRight, 255, 2)

        show_list = [template, img_copy]
        _plt_show(show_list)

def _to_hsv(rgb_image):
    hsv = cv.cvtColor(rgb_image, cv.COLOR_BGR2HSV)             # 转到 HSV 空间
    red1 = (156, 180)                                       # 红色的 H 值范围
    red2 = (0, 25)                                          # 红色的 H 值范围
    blue1 = (100, 124)

    color_list = [red1, blue1]                         # 符合 H 值条件的颜色列表 
    s_min,s_max,v_min,v_max = 43, 255, 46, 255              # S、V 范围

    # height, width, _ = rgb_image.shape                      # 获取读入图片的尺寸
    # imgResult = np.zeros((height, width), np.uint8)      # 创建与读入图片大小一致的纯黑图片
    # 遍历颜色列表
    flag = true
    for color in color_list:
        h_min, h_max = color                                # 此时 H 范围
        lower = np.array([h_min,s_min,v_min])               # HSV 最小值
        upper = np.array([h_max,s_max,v_max])               # HSV 最大值
        mask =  cv.inRange(hsv, lower, upper)               # 原图hsv在此范围的蒙版
        plt.imshow(mask)
        plt.show()
        img = cv.bitwise_and(rgb_image,rgb_image,  mask = mask)
        if flag:
            imgResult = img
            flag = False
        imgResult = cv.bitwise_or(imgResult,img)
    return imgResult

if __name__ == '__main__':
    img_path = 'img3.jpg'
    template_path = 'c53babdf0_787.jpg'
    template = cv.imread(template_path)
    rgb_image = cv.imread(img_path)
    h,w = rgb_image.shape[:2]
    h = round(0.6 * h)
    rgb_image = cv.imread(img_path)[0:h,:,:]
    imgResult = _to_hsv(rgb_image=rgb_image)
    cv.imwrite('result.png', imgResult)
    print(imgResult.shape)
    plt.imshow(imgResult[:,:,[2,1,0]])
    plt.show()

    # resize_interval = np.linspace(float(0.9), float(1.1), 3)
    resize_interval = [0.7]
    for i in resize_interval:
        print('resize times = ' + str(i))
        rs_template = _resize(template_path, i, i)
        template = cv.cvtColor(np.array(rs_template), cv.COLOR_RGB2BGR)
        matchTemplate(rgb_img = imgResult, template = template)
        # _slide_match(imgResult, 200, [40,40], template )

    # template = cv.imread(template_path)
    # print(template.shape)
    # matchTemplate(rgb_img = imgResult, template = template)
