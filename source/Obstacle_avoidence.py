"""
这个项目按照前进方向进行避障
1. YOLOX 进行框的识别
2， R factor 进行过滤
"""
import os

from yolox.GenshinObstcaleInterface import DetectorInterface
from MiDaS.Midas_Mondepth import Monodepth
from loguru import logger
from pathlib import Path
import numpy as np
import math
import cv2
from datetime import datetime
today = datetime.now().strftime("%m%d%H%M%S")

class ObstcaleAvoid:
    def __init__(self):

        self.count = 0
        exp = r"./Weights/exps/yolox_s_coco_genshin.py"  # yolox_s_coco_depth_genshin.py
        ckpt = r"./Weights/yolox/Co_Normal_best_ckpt0306.pth"
        self.detec = DetectorInterface(exp, ckpt)  # YOLOX 检测器

        model_type = "dpt_swin2_tiny_256"
        weight_path = r"../Weights/Midas"
        self.depth = Monodepth(model_type, weight_path)

    def cut_img(self,img):
        '''剪裁图片只保留有用用于识别的地方, 不包含小地图之类的
        :param img:
        :return:
        '''
        H, W = img.shape[:2]
        c_h, c_w = H // 2, W // 2  # 中心点
        l_coord = c_w - W // 3, c_h - H // 3 # x0 y0
        r_coord = c_w + W // 3 - 50, c_h + H // 3 #x1, y1
        return img[l_coord[1]:r_coord[1], l_coord[0]:r_coord[0]]  # [y0:y1,x0:x1]



#####################################
    def Rfactor(self, box, H, W):
        h, w = box[3] - box[1], box[2] - box[0]
        return max(w / W, h / H) #> 0.56  # 大于这个阈值为true，是有效深度

    @logger.catch()
    def CheckStdAvg(self,std,avg):
        '''
        满足Std Avg的有效深度，
        :param avg:
        :param std:
        '''

        if math.isfinite(std) or math.isnan(std) or math.isfinite(avg) or math.isnan(avg):
            return True
        elif avg >= 50 or std >= 60:#190 200
            logger.info(f"avg是{int(avg)} std是{int(std)}")
            return True
        else:
            return  False

    def CBbox(self,img):
        #r人物框的位置 bbox
        H, W = img.shape[:2]
        c_h, c_w = H // 2, W // 2  # 中心点
        x0,y0 = c_w - W // 35, c_h - H // 9  # 贴合人物
        x1,y1 = c_w + W // 15, c_h + H // 3  # 点

        return (int(x0),int(y0),int(x1),int(y1))

    @logger.catch()
    def ValidObs(self,img, save= False):
        self.Cx0, Cy0, self.Cx1, Cy1 = self.CBbox(img)  # 人物框位置
        #检测挡路的有效障碍物
        res = self.detec.predict(img[:, :, :3])
        #todo 删除
        base = "./Temp/img"
        if save:
            cv2.imwrite(f"{base}/orgImg/{today}_{self.count:0>5d}.jpg", img)
        BOx0, BOx1 = [], [] # BlockObstacle #有效深度 且挡住人的框
        if res is not None:
            bboxes, img_info = res
            depth_map = None# 减少重复计算深度图
            for bbox in bboxes:
                ###以下是为了找出有效的深度图
                H, W = img_info["height"], img_info["width"]
                R = self.Rfactor(bbox,H,W)
                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                if y0 < 0: y0 = 0  # 防止错误值
                if y1 == 0: y1 = H#
                if (self.Cx0 <= x0 <= self.Cx1 or  self.Cx0 <= x1 <= self.Cx1 or  self.Cx0 <= (x1-x0)/2+x0 <= self.Cx1) and R>=0.40:  #0.65#障碍物和人物框重合,完全包含人物框进去，及标注挡住人物前边的路 且是R有效深度图
                    if depth_map is None:  #没有算过深度图则计算一遍
                        depth_map = self.depth.processing_one(img)
                    if R >= 1.0: #有效深度图
                        BOx0.append(x0), BOx1.append(x1)
                    else:
                        crop = depth_map[y0:y1, x0:x1]  # 截取的深度图
                        avg  = np.average(crop)
                        std  = np.std(crop)
                        if self.CheckStdAvg(std,avg):
                            BOx0.append(x0), BOx1.append(x1) #记录有效的边框
                            #####################
                            if save:#查看实际结果
                                cv2.rectangle(img, (x0,y0), (x1, y1), (255, 255, 0), 2)
        if save:
            cv2.rectangle(img,  (self.Cx0, Cy0), (self.Cx1, Cy1), (255, 255, 255), 2)
            cv2.imwrite(f"{base}/bbox/{today}_{self.count:0>5d}.jpg", img)#(f"./Temp/BBox/{today}_{self.count:0>5d}.jpg", img)




            self.count += 1
        if len(BOx0) == 0 and len(BOx1) == 0:
            return None
        else:
            return  min(BOx0), max(BOx1)

#####################################

    @logger.catch()
    def run(self,Img, save=False):
        #return  是偏转的角度
        # cv2读取过
        #需要提前cut
        # while True:
        Img  = self.cut_img(Img) #只截取有用部分
        ans = self.ValidObs(Img,save)
        if ans is not None:#逻辑写在一张纸上了
            BOx0, BOx1 = ans
            if   BOx0 >= self.Cx0 and BOx1 >= self.Cx1: # 障碍物在人物框右侧  and 添加是为了避免出现最后一种情况
                direction = [BOx0 - self.Cx1, "12障碍物在人物框右侧, 向左走"] # 逆时针 负值
            elif BOx1 <= self.Cx1 and BOx0 <= self.Cx0: # 障碍物在人物框左侧
                direction = [BOx1 - self.Cx0, "34障碍物在人物框左侧，向右走"] #顺时针 正直
            elif BOx1 <= self.Cx1 and BOx0 >= self.Cx0: #障碍物完全在 人物框中
                if  BOx0 - self.Cx0 < self.Cx1 - BOx1:# 右边空间大
                    direction = [self.Cx1 - BOx1, "52障碍物完全在 人物框中 向右边走"] #顺时针 正直
                elif BOx0 - self.Cx0 > self.Cx1 - BOx1:# 左边空间大
                    direction = [self.Cx0 - BOx1, "51障碍物完全在 人物框中 向左边走"] # 逆时针 负值
                else:
                    direction = [0, '没记录障碍物在人物框中']
            else: #self.Cx0>BOx0 and self.Cx1<BOx1 #  障碍物完全超过人物框 这种情况很少，要放在最后
                if self.Cx0 - BOx0 < BOx1 - self.Cx1: #右边空间大
                    direction = [-(BOx1 - self.Cx1), "61障碍物完遮挡人物框 右边遮挡多 向左走 负值"]
                elif self.Cx0 - BOx0 > BOx1 - self.Cx1: #左边空间大
                    direction = [self.Cx0 - BOx0, "62障碍物完遮挡人物框 左边遮挡多 向右走 正直"]
                else:
                    direction = [0, '没记录障碍物完全遮挡人物框' ]
            if  abs(direction[0]) <= 25:
                direction[0] = direction[0] + 30
            elif 45 >= abs(direction[0]) >= 25:
                direction[0] = direction[0]*1.5
            return direction

        else: #表示前边都是畅通区域
            return None
    
            
            
            
            



if __name__ == "__main__":
    import pathlib
    print(pathlib.Path.cwd())
    OA = ObstcaleAvoid()
    for i in Path(r"\Genshin_try\Temp\JPEGImages").iterdir():
        Img = cv2.imread(str(i))
        a= OA.run(Img, save=True)
        if a is not None:
            print(i.name,"==>",a)
