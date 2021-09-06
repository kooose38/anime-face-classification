import os 
from glob import glob 
import numpy as np 
from PIL import Image 
import torch, torchvision 
import torch.nn as nn 
from torchvision import transforms 
from torchvision.models import vgg19 
from torch.utils.data import DataLoader 
from sklearn.metrics import classification_report
from torch.utils.tensorboard import summary 
import matplotlib.pyplot as plt
import json 
from tqdm import tqdm 
import argparse

from networks.vgg import Classifier

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

class Transform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                                         transforms.RandomResizedCrop(resize, scale=(.5, 1.0)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
            ]), 
            "val": transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, phase, img):
        return self.data_transform[phase](img)
    
    
def load_model(labels):
    RESIZE = 224 
    MEAN = (.485, .456, .406)
    STD = (.229, .224, .225)
    net = Classifier(len(labels))
    # gitリポジトリにサイズ関係で置けないので別途用意すること
    net.load_state_dict(torch.load("./weights/classifier4.pth", map_location={"cuda:0": "cpu"}))
    net.eval()
    net.to(device)
    transform = Transform(RESIZE, MEAN, STD)
    return net, transform 

def load_label():
    with open("labels.json", "r") as f:
        label2index = json.load(f)
        f.close()
    index2label = {v: k for k, v in label2index.items()}
    return index2label  

def show_img(img):
    plt.imshow(img.resize((1080, 1080)))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def main(img_path: str):
    # モデルの読み込み
    index2label = load_label()
    net, transform = load_model(index2label)
    # 画像の前処理
    img = Image.open(img_path)
    show_img(img)
    img_tensor = transform("val", img).unsqueeze(0).to(device)
    # 推論
    with torch.no_grad():
        output = net(img_tensor)
        output = nn.Softmax(dim=1)(output)
        pred = output.topk(3)[0][0].detach().cpu().numpy().tolist() # 確率
        pred_id = output.topk(3)[1][0].detach().cpu().numpy().tolist() # index 
    # ラベル名と確率
    results = {}
    for i, (p, idx) in enumerate(zip(pred, pred_id)):
        pred_name = index2label[int(idx)]
        result = {}
        result["score"] = p 
        result["predict"] = pred_name 
        results[str(i+1)] = result
    return results  

parser.add_argument("arg1", help="image path name", type=str, default="1.jpg")
args = parser.parse_args()

if __name__ = "__main__":
    root_path = "img/"
    main(root_path+args.arg1)