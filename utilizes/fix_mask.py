import cv2 
import numpy as np
import os
from tqdm import tqdm
import shutil

for i in range(5):
    path = f'/home/kc/hungpv/polyps/dataset/KCECE/Clinic_fold_new/fold_{i}/masks'
    for image_id in tqdm(os.listdir(path)):
        image = cv2.imread(os.path.join(path, image_id))[:,:,::-1]
        image = image * 255
        cv2.imwrite(os.path.join(path, image_id), image)