
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger #展现日志
import cv2,os,time
from pathlib import Path

import torch


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names= ("Obstacle","Trunk"),
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = (640,640)
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            #创建tensor
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        '''
        推测
        :param img: 
        :return:
        '''
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        elif isinstance(img,list):
            img_info["file_name"] = img[1]
            img = img[0]
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        #事先处理图片
        img = torch.from_numpy(img).unsqueeze(0)
        #from_numpy()用来将数组array转换为张量Tensor
        #unsqueez 升维度，在0维度 升一维
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            #t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

class DetectorInterface:
    def __init__(self, exp=r"./exps/yolox_s_coco_genshin.py",
                 ckpt=r"./yolox_s_coco_genshin/best_ckpt.pth"):
        self.exp = get_exp(exp) #要改哦
        model = self.exp.get_model()
        logger.info("YOLOX模型加载: {}".format(get_model_info(model, self.exp.test_size)))
        model.cuda()
        model.eval()
        logger.info("加载权重....")
        ckpt = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        self.device = "gpu"
        self.legacy = False
        self.fp16 = False
        self.conf = 0.65#0.75
        self.nms = 0.45  #
        self.tsize = 640  # test img size
        self.cls_names = ("Obstacle","Trunk",)

        self.predictor = Predictor(
            model, self.exp,  self.cls_names, None, None,
            self.device, self.fp16, self.legacy,
        )
        logger.info("模型准备完毕...")
    @torch.no_grad()
    def predict(self, img):
        outputs, img_info = self.predictor.inference(img)
        ratio = img_info["ratio"]


        if outputs[0] is None:
            return None

        output = outputs[0].cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        output=torch.cat((bboxes, cls.view(-1,1), scores.view(-1,1)), dim=1)[scores>self.conf,:]
        #拼接两个 tensor
        return output,img_info
        #[[lx,ly,rx,ry,cls,s],[]} 左上x， y坐标，右下x，y坐标，分类 得分) =>  图片信息， 画框的图片




if __name__ == "__main__":
    pass

