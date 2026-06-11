import random

class ALOHA:
    # 静态方法调用参数需要使用类的方法
    act_prob = 0.5 # ALOHA的发送概率
    ALOHA_action_list = [0, 0, 1, 1, 0]

    ALOHA_delay = 6

    @staticmethod
    def action(index):
        if ALOHA.ALOHA_action_list[index] == 1:
            # random.random()会生成一个新的随机数，这些随机数在 [0.0, 1.0) 区间内均匀分布。
            if ALOHA.act_prob < random.random():
                return 1
            else:
                return 0
        else:
            return 0
