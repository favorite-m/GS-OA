"""Compute depth maps for images in the input folder.
"""
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from loguru import logger #展现日志
from MiDaS import midas_utils
from MiDaS.midas.model_loader import load_model


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


class Monodepth:
    def __init__(self, model_type="dpt_beit_large_512", weight_path=None, optimize=False, height=None, square=None,
                 grayscale=True):
        '''

        :param model_type: 确定模型种类， 但需要下载对应的权重， 名字不知道可以随便输，会报错出来
        :param weight_path: 上述模型的权重
        :param optimize:  Optimize 加速用 一般关闭
        :param height: 自动设定高度参数
        :param square: 设定模型接受的横纵比
        :param grayscale: 是否为灰度图
        '''
        logger.info("初始化深度图模型中.....")
        self.model_type = model_type
        # select device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model
        if "swin" in model_type:
            self.optimize = False
            logger.info("因为是Swin模型, Optimize设置为关闭")
        elif not optimize:
            self.optimize = False
        elif optimize and self.device == torch.device("cuda"):
            self.optimize = optimize  # Use half-float optimization
            logger.info("Optimization已开启")
        else:
            self.optimize = False
            self.model_type = "openvino_midas_v21_small"
            self.device = torch.device("cpu")
            logger.info(self.device)
            logger.info(f"设置运行小模型{self.model_type}")

        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, weight_path, model_type, optimize,
                                                                        height, square)

        logger.info("模型在{}上运行 模型准备完毕...".format(self.device))

        ####无关紧要设定
        self.compare = False  # 是否和原图片进行对比
        self.grayscale = grayscale  # 是否用灰度显示还是显示彩色的

    def processing_one(self, img_path, output_path=None, Imgname=None, cut=False):
        """
        处理单个图片的深度图
        :param img_path:
        :param output_path:  深度图保存位置，没有输入则不保存
        :return:
        """
        start = time.time()
        original_image_rgb = midas_utils.read_image(img_path, cut=False)  # cut直接保留有用的 不包含任务栏
        img_input = self.transform({"image": original_image_rgb})["image"]
        with torch.no_grad():
            ###############################################################
            if "openvino" in self.model_type:
                sample = [np.reshape(img_input, (1, 3, *(self.net_w, self.net_h)))]
                prediction = self.model(sample)[self.model.output(0)][0]
                self.prediction = cv2.resize(prediction, dsize=original_image_rgb.shape[1::-1],
                                        interpolation=cv2.INTER_CUBIC)
            else:
                sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
                if self.optimize:
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()
                prediction = self.model.forward(sample)
                self.prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=original_image_rgb.shape[1::-1][::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

        if output_path is not None:
            # output
            # self.write(prediction, output_path)
            # print(prediction)
            if Imgname == None:
                output_path = str(Path().joinpath(output_path,  Path(img_path).stem))
            else: output_path = str(Path().joinpath(output_path, Imgname))#Path(img_path).stem))

            if not self.compare:
                midas_utils.write_depth(output_path, self.prediction, self.grayscale, bits=2)
            else:
                original_image_bgr = np.flip(original_image_rgb, 2)
                content = create_side_by_side(original_image_bgr * 255, self.prediction, self.grayscale)
                cv2.imwrite(output_path, content)
        # print(f"共用时{time.time() - start:.2f}s")

        return self.prediction





    def show_info(self, voc_file, img, depth_path, savepath1, show=False):  # 显示深度信息等东西
        box_results = midas_utils.VOC_imitate_yolo(voc_file)
        img = cv2.imread(img)
        depth = cv2.imread(depth_path)
        H, W = self.prediction.shape[:2]
        crops_depth_areas = []  # 识别出有效的物体位置的的深度图
        for box in box_results:
            R = self.Rfactor(box, H, W)  # 满足R 是有效深度
            l_coord, r_coord = box[0], box[1]
            crop = self.prediction[l_coord[1]:r_coord[1], l_coord[0]:r_coord[0]]  # 截取的深度图
            avg = np.average(crop)
            mean = np.mean(crop)
            std = np.std(crop)  # 方差的 开平方
            var = np.var(crop)  # 方差 The average of the squared differences from the Mean.越大表示数字离得越远
            # #########################显示相关信息
            x = box[0][0] + int((box[1][0] - box[0][0]) / 2)
            y = box[0][1] + int((box[1][1] - box[0][1]) / 2)
            color = (4, 250, 7)
            cv2.rectangle(img, box[0], box[1], color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"R:{R:.2f}", (x, y), font, 0.5, (0, 255, 0), 2)  # 绿色
            cv2.putText(img, f"Avg:{avg:.2f}", (x, y - 20), font, 0.5, (6, 230, 230), 2)
            color = (4, 250, 7)
            cv2.rectangle(depth, box[0], box[1], color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #  cv2.putText(depth, f"R:{R:.2f}", (x, y), font, 0.5, (0,255, 0), 2) #绿色
            cv2.putText(depth, f"Avg:{avg:.2f}", (x, y - 20), font, 0.5, (6, 230, 230), 2)
            # cv2.putText(depth, f"Mean:{mean:.2f}", (x, y - 40), font, 0.5, (6, 230, 230), 2)
            cv2.putText(depth, f"Std:{std:.2f}", (x, y - 60), font, 0.5, (6, 230, 230), 2)
            #  # cv2.putText(depth, f"Var:{var:.2f}", (x, y - 80), font, 0.5, (6, 230, 230), 2)
            #  ##########################人物
            c_h, c_w = H // 2, W // 2  # 中心点
            l_coord = c_w - W // 35, c_h - H // 9  # 贴合人物
            r_coord = c_w + W // 15, c_h + H // 3  # 点
            cv2.rectangle(depth, l_coord, r_coord, (255, 255, 0), 2)
            # 满足以下条件是符合要求的图的划分
            if R >= 0.52:
                if avg >= 1300:
                    cv2.line(depth, (box[0][0], 0), (box[0][0], H), (0, 0, 255), 2)
                    cv2.line(depth, (box[1][0], 0), (box[1][0], H), (0, 0, 255), 2)
                    c_h, c_w = H // 2, W // 2  # 中心点

        cv2.imwrite(savepath1, depth)

        # photo = np.hstack((img,depth))
        # cv2.imwrite(savepath1, photo)

        if show:
            cv2.imshow("RGB", img)
            cv2.imshow("Depth", depth)
            cv2.waitKey(0)
            cv2.destroyAllWindows()






if __name__ == "__main__":
    # dpt_swin2_tiny_256
    model_type = "dpt_swin2_tiny_256"  # "dpt_beit_large_512"#"dpt_swin2_large_384" #"dpt_swin2_tiny_256"# "openvino_midas_v21_small_256"#"dpt_swin2_large_384"#"openvino_midas_v21_small_256"#" "dpt_beit_large_512"  # "
    weight_path = r""
    output_path = r""  # 如果不设置则不保存
    Path(output_path).mkdir(parents=True, exist_ok=True)
    img_dir = r""
    model_ = Monodepth(model_type, weight_path, optimize=False, grayscale=True)
    a = model_.processing_one(r"")
    print(a)
    # for i in Path(img_dir).iterdir():
    #     model_.processing_one(str(i), output_path)

