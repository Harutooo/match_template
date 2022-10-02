'''
hqauto: 
Description: 
version: 
Author: Xiaoxia Liu.
Date: 2022-09-10 17:08:11
LastEditors: Please set LastEditors
LastEditTime: 2022-09-27 10:13:59
'''
import logging
import os
import sys
from unittest import result
from rect import Rect
from utils import get_clusters
from time import time
import cv2
from matplotlib import pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

'''
功能：  将传入图片的红色、蓝色区域进行分割，得到分割后的图像
参数：  rgb_image - 待分割图像
返回值：原图中红、蓝区域
'''
def _to_hsv(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)        # 转到 HSV 空间
    red1 = (156, 180)                                       # 红色的 H 值范围
    red2 = (0, 25)                                          # 红色的 H 值范围
    blue1 = (100, 124)

    color_list = [red1]                                     # 符合 H 值条件的颜色列表 
    s_min,s_max,v_min,v_max = 43, 255, 46, 255              # S、V 范围

    flag = True
    for color in color_list:
        h_min, h_max = color                                # 此时 H 范围
        lower = np.array([h_min,s_min,v_min])               # HSV 最小值
        upper = np.array([h_max,s_max,v_max])               # HSV 最大值
        mask =  cv2.inRange(hsv, lower, upper)               # 原图hsv在此范围的蒙版

        if flag:
            new_mask = mask
            flag = False

        new_mask = cv2.bitwise_or(mask, new_mask)
    imgResult = cv2.bitwise_and(rgb_image, rgb_image, mask = new_mask)
    return imgResult

'''
类： 模板匹配
参数： scales       - 模板的缩放尺度
      max_distance - 最大像素距离
      criterion    - 匹配损失计算方法
      worst_match  - 匹配分数阈值
      debug        - 是否进行调试显示匹配图片
'''    
class TemplateMatcher:
    def __init__(
        self,
        scales = np.arange(0.5, 3, 0.1),
        # max_distance = 14,
        max_distance = 80,
        criterion = cv2.TM_CCOEFF_NORMED,
        worst_match = 0.75,
        debug = False,
    ):
        self.scales = scales
        self.max_distance = max_distance

        self.criterion = criterion
        self.worst_match = worst_match
        self.debug = debug

    # 进行 feature 在 scene 上的匹配
    def match(self, feature, scene, mask=None, scale=None, crop=True, cluster=True):
        if isinstance(mask, Rect) and not crop:
            scene_working = scene.copy()
            scene_working *= mask.to_mask()
        elif isinstance(mask, Rect):
            scene_working = scene[
                mask.top : (mask.top + mask.height),
                mask.left : (mask.left + mask.width),
            ].copy()
        else:
            scene_working = scene.copy()

        if scale is None:
            scale = self._find_best_scale(feature, scene_working)

        match_candidates = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            # Peaks in matchTemplate are good candidates.
            peak_map = cv2.matchTemplate(
                scene_working, scaled_feature, self.criterion
            )

            if self.criterion in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                best_val, _, best_loc, _ = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map <= self.worst_match)
            else:
                _, best_val, _, best_loc = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map >= self.worst_match)

            good_points = list(zip(*good_points))
            
            if self.debug:
                LOGGER.warning(
                    "%f %f %f %s %s",
                    scale,
                    self.worst_match,
                    best_val,
                    best_loc,
                    good_points,
                )
                h, w,_ = scaled_feature.shape
                cv2.rectangle(scene_working, best_loc, (best_loc[0] + w, best_loc[1] + h), 255, 2)
                plt.imshow(scene_working[:,:,[2,1,0]])
                plt.show()

            if cluster:
                clusters = get_clusters(
                    good_points, max_distance=self.max_distance
                )
            else:
                clusters = [(pt,) for pt in good_points]

            # TODO Break these down into more comprehensible comprehensions.
            match_candidates = [
                max(clust, key=lambda pt: peak_map[pt]) for clust in clusters
            ]
            # print('lenth of clusters :')
            # print(len(clusters[0]))
            # print('match_candidates1 : ')
            # print(match_candidates)

            if isinstance(mask, Rect):
                match_candidates = [
                    ((peak[0] + mask.top, peak[1] + mask.left), peak_map[peak])
                    for peak in match_candidates
                ]
            else:
                match_candidates = [
                    (peak, peak_map[peak]) for peak in match_candidates
                ]
            # print('match_candidates2 : ')
            # print(match_candidates)
        return (scale, match_candidates)

    def _find_best_scale(self, feature, scene):

        best_corr = 0
        best_scale = None

        for scale in self.scales:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            result = cv2.matchTemplate(scene, scaled_feature, self.criterion)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_corr:
                best_corr = max_val
                best_scale = scale

        if self.debug:
            LOGGER.warning("%f %f", best_scale, best_corr)

        if best_corr > self.worst_match:
            return best_scale
            
        return None

def match_picture(scene, future_path, save_path):
    save_name = os.path.join(save_path, scene)
    feature = cv2.imread(future_path)                                          # 读入模板图片
    rgb_img = cv2.imread(scene)                          # 读入待检测图片
    
    img_h, img_w = rgb_img.shape[:2]                                                # 待检测图片尺寸    
    img = rgb_img[0:round(0.6 * img_h),:,:]                                         # 剪裁下面部分
    scene = _to_hsv(img)                                                            # 制作蒙版
    cv2.imwrite(save_name, scene)

    matcher = TemplateMatcher(worst_match = 0.3,debug = True)
    result = matcher.match(feature=feature, scene=scene, scale=None)                      # , scale = None

    # print('result : {} '.format(result))
    data = result[1]
    if len(data) > 0:
        scale = result[0]
        best_loc = (data[0][0][1], data[0][0][0])
        scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)
        h, w,_ = scaled_feature.shape
        cv2.rectangle(scene, best_loc, (best_loc[0] + w, best_loc[1] + h), 255, 2)
        plt.imshow(scene[:,:,[2,1,0]])
        plt.show()

def main_worker(scene_path, future_path, mask_path):
    scenes = os.listdir(scene_path)

    for s in scenes:
        scene = os.path.join(scene_path, s)
        match_picture(scene, future_path, mask_path)



# def main_worker(scene_path, future_path, mask_path):
#     scenes = os.listdir(scene_path)

#     for s in scenes:
#         save_name = os.path.join(mask_path, s)
#         feature = cv2.imread(future_path)                                          # 读入模板图片
#         rgb_img = cv2.imread(os.path.join(scene_path, s))                          # 读入待检测图片
        
#         img_h, img_w = rgb_img.shape[:2]                                                # 待检测图片尺寸    
#         img = rgb_img[0:round(0.6 * img_h),:,:]                                         # 剪裁下面部分
#         scene = _to_hsv(img)                                                            # 制作蒙版
#         cv2.imwrite(save_name, scene)

#         matcher = TemplateMatcher(worst_match = 0.4,debug = True)
#         result = matcher.match(feature=feature, scene=scene, scale=None)                      # , scale = None

#         # print('result : {} '.format(result))
#         data = result[1]
#         if len(data) > 0:
#             scale = result[0]
#             best_loc = (data[0][0][1], data[0][0][0])
#             scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)
#             h, w,_ = scaled_feature.shape
#             cv2.rectangle(scene, best_loc, (best_loc[0] + w, best_loc[1] + h), 255, 2)
#             plt.imshow(scene[:,:,[2,1,0]])
#             plt.show()
            
if __name__ == '__main__':
    pre_time = time()
    # scene_path = 'imgs/scene/'
    feature_path = 'imgs/feature/max.jpg'
    mask_path = 'imgs/hsv_result/'

    if len(sys.argv) == 2:
        if os.path.isfile(sys.argv[1]):
            match_picture(sys.argv[1], feature_path, mask_path)
        elif os.path.isdir(sys.argv[1]):
            main_worker(sys.argv[1], feature_path, mask_path)
        else:
            print("it's not a dir and file,please input again")
    else:
        print("input again")
        
    print('use time : ' + str(time() - pre_time))

    # imgs = cv2.imread('min50.jpg')
    # mask = _to_hsv(imgs)
    # cv2.imwrite('mask50.jpg', mask)