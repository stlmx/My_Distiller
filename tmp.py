import torch
import torch.nn as nn
import torch.nn.functional as F
from data.imagenet_constant import IMAGENET_CLASSES

for i  in range(1000):
    if IMAGENET_CLASSES[i] == "monarch butterfly":
        print("the index of monarch butterfly is:", i)
    if IMAGENET_CLASSES[i] == "gossamer-winged butterfly":
        print("the index of gossamer-winged butterfly is:", i)
    if IMAGENET_CLASSES[i] == "howler monkey":
        print("the index of howler monkey is:", i)
    if IMAGENET_CLASSES[i] == "fly":
        print("the index of fly is:", i)

# emb = torch.load("data/class_promt.pth")
# emb = torch.stack(emb).squeeze(1)

# print(emb.min())

print(IMAGENET_CLASSES[874])