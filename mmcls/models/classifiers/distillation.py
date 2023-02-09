from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import BaseDataElement

from mmcls.models.classifiers import BaseClassifier
from mmcls.registry import MODELS

import clip
from PIL import Image
from data.imagenet_constant import IMAGENET_CLASSES

@MODELS.register_module()
class Distiller(BaseClassifier):
    """
    我的图像分类backbone蒸馏抽象类
    """
    def __init__(self, init_cfg: Optional[dict] = None, 
                 data_preprocessor: Optional[dict] = None, 
                 train_cfg=None,
                 teacher=None, 
                 student=None, 
                 head=None):
        super().__init__(init_cfg, data_preprocessor)
        
        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg
            
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
        # x_t = self.teacher.features(inputs)
        if mode == "tensor":
            print("真的用到了这个模式吗？？？")
        elif mode == "loss":
            # label = self.gen_text(inputs=inputs, data_samples=data_samples)
            # pre_txt = self.teacher.head(self.teacher(inputs))
            
            return self.teacher.head.loss(self.teacher(inputs), data_samples)
            
        elif mode == "predict":
            return self.teacher.head.predict(self.teacher(inputs), data_samples)
        
    def distill(self, label_txt, pre_txt):

        return F.mse_loss(label_txt, pre_txt)
    
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
    
    def gen_text(self, data_samples, inputs):
        
        label_text_dict = torch.load("./data/class.pth")
        label_text = []
        for i in range(len(data_samples)):
            label_text.append(label_text_dict[i])
        
        label = torch.concat(label_text).to(inputs.device)
        
        return label
