from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import BaseDataElement

from mmcls.models.classifiers import BaseClassifier
from mmcls.registry import MODELS

@MODELS.register_module()
class Distiller(BaseClassifier):
    """
    我的图像分类backbone蒸馏抽象类
    """
    def __init__(self, init_cfg: Optional[dict] = None, data_preprocessor: Optional[dict] = None, teacher=None, student=None, head=None):
        super().__init__(init_cfg, data_preprocessor)
        if not isinstance(teacher, nn.Module):
            self.teacher = MODELS.build(teacher)
        if not isinstance(student, nn.Module):
            self.student = None
        if not isinstance(head, nn.Module):
            self.teacher.head = MODELS.build(head)
            # self.student.head = MODELS.build(head)

        # self.load_teacher_ckpt()
        # self.load_student_classifier()

    def forward(self, inputs: torch.Tensor, data_samples: Optional[List[BaseDataElement]] = None, mode: str = 'tensor'):
        # x_s = self.student.features(inputs)
        x_t = self.teacher.features(inputs)
        if mode == "tensor":
            print("真的用到了这个模式吗？？？")
        elif mode == "loss":
            # loss = self.distill(output_s=x_s, output_t=x_t)
            # return {"distill_loss": loss}
            return self.teacher.head.loss(self.teacher(inputs), data_samples)
            
        elif mode == "predict":
            return self.teacher.head.predict(self.teacher(inputs), data_samples)
        
    def distill(self,output_s, output_t):

        return F.mse_loss(output_s[0], output_t[0])
    
    def load_teacher_ckpt(self, ckpt_path=None):
        "给教师模型load权重, 并且把教师模型的权重冻住不能被优化"
        state =  torch.load("/data/limingxuan/new_mmlab/mmclassification/ckpt/epoch_5.pth")

        new = {}
        for key in state["state_dict"].keys():
            ckpt = state["state_dict"].copy()
            tmp = key.replace("teacher.", "")
            new.update({tmp:ckpt.pop(key)})
        self.teacher.load_state_dict(new)
        self.teacher.eval()
        
    def load_student_classifier(self, ckpt_path=None):
        "我猜想如果只做features层面的蒸馏loss, 分类头的参数随机初始化正确率是10%对于十分类"
        state =  torch.load("/data/limingxuan/new_mmlab/mmclassification/ckpt/epoch_5.pth")

        new = {}
        for key in state["state_dict"].keys():
            ckpt = state["state_dict"].copy()
            tmp = key.replace("teacher.", "")
            if tmp.startswith("classifier") == True:
                new.update({tmp:ckpt.pop(key)})
        self.student.load_state_dict(new)
        # self.teacher.eval()   
        
    
