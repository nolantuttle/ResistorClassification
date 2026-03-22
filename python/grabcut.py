import numpy as np
import cv2 as cv
import os
import shutil as sh
from matplotlib import pyplot as plt

def grabcut_image(filepath):
    image = cv.imread(filepath)
    assert image is not None, "Invalid filepath, try again!"
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (200,200,100,100)
    cv.grabCut(image,mask,rect,bgdModel,fgdModel,25,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]
    return image

def grabcut_dir(directory):
    new_directory = "archive_clean"
    sh.copytree(directory, new_directory, dirs_exist_ok=True)
    count = 0
    num_files = 0

    for root, dirs, files in os.walk(new_directory):
        for file in files:
            num_files = num_files + 1

    for root, dirs, files in os.walk(new_directory):
        for file in files:
            if (file.endswith('.jpg')):
                image = cv.imread(os.path.join(root, file))
                assert image is not None, "Invalid filepath, try again!"
                mask = np.zeros(image.shape[:2],np.uint8)
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (175,175,350,350)
                cv.grabCut(image,mask,rect,bgdModel,fgdModel,15,cv.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                image = image*mask2[:,:,np.newaxis]
                cv.imwrite(os.path.join(root, file), image)
                count = count + 1
                if count % 10 == 0:
                    print('Files Processed:', count)
                    print('Percent Complete:', f"{count/num_files * 100:.2f}", '%')


cv.imshow('image', grabcut_image('archive/1K_1-4W/1K_1-4W_(2).jpg'))
cv.waitKey(0)
#cut_image = grabcut_dir('archive')