import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
#Metadata
IMG_DIR = "/home/ubuntu/Deep_Learning/final/plant/images/"
LABEL_DIR = "/home/ubuntu/Deep_Learning/final/plant/"
#targets
train = pd.read_csv(LABEL_DIR+"train.csv")
print(train.head(6))

sizes = train[['healthy','multiple_diseases','rust','scab']].sum().tolist()

#Targets Distribution
labels =['healthy','multiple_diseases','rust','scab']
explode = [0.1,0,0,0]
plt.pie(sizes,explode=explode,labels=labels,startangle=90,autopct = '%3.1f%%')
plt.title('Targets Distribution')
plt.axis('equal')
plt.savefig('Targets.png')
plt.show()

def visual_img(label):
    sample_list = train[train[label]==1]['image_id'].head(4).tolist()
    sample_img = []
    for i in range(len(sample_list)):
        sample_img.append(cv2.cvtColor(cv2.imread(IMG_DIR + sample_list[i] + ".jpg"),cv2.COLOR_BGR2RGB))
    sample_img = np.array(sample_img)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(label,fontsize=16)
    axs[0,0].imshow(sample_img[0])
    axs[0,0].set_title(sample_list[0])
    axs[0,0].axis('off')
    axs[0,1].imshow(sample_img[1])
    axs[0,1].set_title(sample_list[1])
    axs[0,1].axis('off')   
    axs[1,0].imshow(sample_img[2])
    axs[1,0].set_title(sample_list[2])
    axs[1,0].axis('off')
    axs[1,1].imshow(sample_img[3])
    axs[1,1].set_title(sample_list[3])
    axs[1,1].axis('off')
    plt.savefig(label+'.png')
    plt.show()
    
    

for i in range(len(labels)):
    visual_img(labels[i])