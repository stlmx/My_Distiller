import torch
from data.imagenet_constant import IMAGENET_CLASSES
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device="cpu")

# lt = [torch.ones(1), torch.ones(1)]
# at = torch.concat(lt)

# label_text = []
# for i in range(len(lt)):
#     label_text.append(model.encode_text(clip.tokenize(IMAGENET_CLASSES[i]).cuda()))

result = [] 
for i in range(len(IMAGENET_CLASSES)):
    result.append(model.encode_text(clip.tokenize(IMAGENET_CLASSES[i])))


torch.save(result, "./data/class.pth")
    
    


