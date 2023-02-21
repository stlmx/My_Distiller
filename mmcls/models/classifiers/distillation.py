from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import BaseDataElement

from mmcls.models.classifiers import BaseClassifier
from mmcls.registry import MODELS


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
        
        self.label_text_dict = torch.load("./data/class_promt.pth")
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)

        # self.load_teacher_ckpt()
        # self.load_student_classifier()

    def forward(self, inputs: torch.Tensor, data_samples: Optional[List[BaseDataElement]] = None, mode: str = 'tensor'):
        # x_s = self.student.features(inputs)
        # x_t = self.teacher.features(inputs)
        if mode == "tensor":
            print("真的用到了这个模式吗？？？")
        elif mode == "loss":
            label = self.gen_text(inputs=inputs, data_samples=data_samples)
            # pre_txt的维度为[bs, 512]
            pre_txt = self.teacher.head(self.teacher(inputs))[-1]
            loss_dis = self.distill(label, pre_txt, data_samples=data_samples, inputs=inputs)
            
            loss_ori = self.teacher.head.loss(self.teacher(inputs), data_samples)
            
            loss = {}
            loss_ori.update({"loss_dis":loss_dis})
            return loss_ori
            
        elif mode == "predict":
            return self.teacher.head.predict(self.teacher(inputs), data_samples)
        
    def distill(self, label_txt, pre_txt, data_samples, inputs):
        one_hot_label = self.gen_one_hot_label(data_samples, inputs)
        # text_label是[1000, 512]
        text_label = torch.stack(self.label_text_dict)
        text_label = text_label.squeeze(1).to(one_hot_label.device)
        
        pre_txt = self.norm1(pre_txt)
        text_label = self.norm2(text_label)
        
        matrix = pre_txt @ text_label.T
        
        score = F.cross_entropy(matrix, one_hot_label)

        return score
    
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
        
        label_text = []
        for i in range(len(data_samples)):
            label_text.append(self.label_text_dict[data_samples[i]._gt_label.label])
        
        label = torch.concat(label_text)
        label = label.to(inputs.device)
        
        return label
    
    def gen_one_hot_label(self, data_samples, inputs):
        one_hot_label = torch.zeros((len(data_samples), 1000), device=inputs.device)
        
        for i in range(len(data_samples)):
            one_hot_label[i][data_samples[i]._gt_label.label] = 1
            
        return one_hot_label
