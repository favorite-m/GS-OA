from source.interaction_background import InteractionBGD
itt = InteractionBGD()
from source.teyvat_move_controller import TeyvatMoveController
TM = TeyvatMoveController()

from source.Obstacle_avoidence import ObstcaleAvoid
OA = ObstcaleAvoid()
import source.movement as mov

from pathlib import Path
from source.setup_logger import setup_logger

walk_flag = True
import source.img_manager as imgM

from source.util import *


def f():
    return False

class Walk:
    def __init__(self):
        log_file = r"./Logs"
        if not Path(log_file).exists(): Path(log_file).mkdir(parents=True, exist_ok=True)
        setup_logger(
            log_file,
            distributed_rank=0,
            filename="run_log.txt",
            mode="a",
        )

    def motion_state(self):
        check = False
        if TM.check_climbing() or TM.check_swimming():
            check = True
        if TM.check_flying():
            check = True
            itt.key_press("space")
        return check
    def catch_img(self):
        while True:
            img = itt.capture()
            base = "./Temp/img"

    @logger.catch()
    def zou(self):
        mov.reset_view()
        flag = 0 #在多少次没有检测到 障碍物 重置视角
        mov.view_to_quest_angle() #朝向日常任务
        time.sleep(0.1)
        WkeyPress = False # W键是否按下
        while walk_flag:
            if WkeyPress == False:
                itt.key_down('w', is_log=False)
                WkeyPress = True
            img = itt.capture()
            res = OA.run(img,save=False)
            if self.motion_state():
                flag = 0 #是否存在特殊状态，如飞行。步行的时间就加长
            if res != None: #说明有障碍物
                itt.key_up('w', is_log=False)##############################
                WkeyPress = False
                move, strategy = res[0], res[1]
                if move != 0:
                    itt.move_to(int(move),0,relative=True)
                    logger.info(f"策略是{strategy},向{move}移动")
                    if itt.get_img_existence(imgM.motion_climbing):
                        itt.key_press('x')
                        time.sleep(0.065)
                        itt.move_to(int(move), 0, relative=True)
                else:
                    logger.info(f"策略{strategy}是错误。向{move}移动")
                flag += 10
            else:#没有障碍物增加一次记录
                flag += 1
                if flag > 70:
                    itt.key_up('w', is_log=False)  ########################
                    WkeyPress = False
                    logger.info("重置视角")
                    mov.reset_view()
                    mov.cview()
                    if itt.get_img_existence(imgM.motion_climbing):
                        logger.info("特殊状态")
                        itt.key_press('x')
                        time.sleep(0.065)
                        itt.move_to(20, 0, relative=True)
                    mov.view_to_quest_angle()
                    flag = 0

    @logger.catch()
    def test(self):
        mov.view_to_quest_angle()



if __name__ == "__main__":
    Walk().zou()

