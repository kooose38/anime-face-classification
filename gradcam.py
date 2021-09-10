from lime.wrappers.scikit_image import SegmentationAlgorithm 
from skimage.segmentation import mark_boundaries
from lime import lime_image
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 
import uuid 
import os 
import argparse

from predict import load_model, load_label 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
index2label = load_label()
net, transform = load_model(index2label)

def predict_fn(img):
    '''モデルの予測を確立で返す'''
    img_tensor = torch.stack(tuple(transform("val", i) for i in img), dim=0)
    img_tensor = img_tensor.to(device)
    net.to(device)
    output = net(img_tensor)
    proba = torch.nn.functional.softmax(output, dim=1)
    return proba.detach().cpu().numpy()

def show_boundary(img_path: str, num_samples: int=400, show_idx: int=0):
    """指定されたラベルにおける寄与を出力する"""
    img = Image.open(img_path).resize((224, 224))
    segmentaion_fn = SegmentationAlgorithm("quickshift", kernel_size=4, 
                                           max_dist=200, ratio=.2, random_state=0)
    segments = segmentaion_fn(img)
    explainer = lime_image.LimeImageExplainer(random_state=0)
    exp = explainer.explain_instance(np.array(img), predict_fn, top_labels=2, hide_color=0, 
                                    num_samples=num_samples, segmentation_fn=segmentaion_fn)
    class_index: int = exp.top_labels[show_idx]
    index2label: Dict[int, str] = load_label()
    image, mask = exp.get_image_and_mask(class_index, positive_only=False, num_features=5, hide_rest=False)
    img_boundary = mark_boundaries(image, mask)

    id = uuid.uuid4()
    os.makedirs("results", exist_ok=True)
    fig = plt.figure()
    plt.imshow(img_boundary)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{index2label[int(class_index)]}")
    plt.show()
    fig.savefig(f"results/img{str(id)[:4]}.png")

def show_ratio(img_path: str, num_samples: int=200, show_idx: int=0):
    """指定されたラベルの降順に寄与を出力する"""
    img = Image.open(img_path).resize((224, 224))
    segmentaion_fn = SegmentationAlgorithm("quickshift", kernel_size=4, 
                                           max_dist=200, ratio=.2, random_state=0)
    segments = segmentaion_fn(img)
    explainer = lime_image.LimeImageExplainer(random_state=0)
    exp = explainer.explain_instance(np.array(img), predict_fn, top_labels=2, hide_color=0, 
                                    num_samples=num_samples, segmentation_fn=segmentaion_fn)
    class_index = exp.top_labels[show_idx]
    index2label: Dict[int, str] = load_label()

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    ax = axes.ravel()
    plt.subplots_adjust(top=1.1, wspace=0, hspace=0)
    for i in range(4):
        area_index, value = exp.local_exp[class_index][i]
        img = exp.image.copy()
        c = 0 if value < 0 else 1
        img[segments == area_index, c] = np.max(img)
        ax[i].imshow(img)
        ax[i].set_title(f"{index2label[class_index]} ratio: {value:.5f}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
    os.makedirs("results", exist_ok=True)
    id = uuid.uuid4()
    fig.savefig(f"results/img{str(id)[:4]}.png")

    
parser.add_argument("--image", help="image path name", type=str, default="1.jpg")
parser.add_argument("--types", help="show ratio or boundary", type=bool, default=True)
args = parser.parse_args()
image = str(args.image)
type = args.types 
if type:
    show_boundary(image)
else:
    show_ratio(image)
