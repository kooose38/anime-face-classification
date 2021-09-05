import matplotlib.pyplot as plt 
import glob 
from PIL import Image 

def main():
    filename = "./data/*.png"
    for f in glob.glob(filename):
        img = Image.open(f)
        plt.imshow(img)
        
if __name__=="__main__":
    main()