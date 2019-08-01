import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import glob
import cv2
import math
import scipy
import os
import random

random.seed(233)

def addHaze(CleanImgName, OutputFolder):
    output_path = 'E:\\Users\\Haze\\' + OutputFolder
    
    A = np.zeros(10)
    t = np.zeros(10)
    HazeRegionX = np.zeros(10)
    HazeRegionY = np.zeros(10)
    HazePointX = np.zeros(10)
    HazePointY = np.zeros(10)

    for i in range(0,10):
        CleanImg = misc.imread(CleanImgName)
        CleanImg = np.array(CleanImg, dtype=float)/255
        ImgSize = CleanImg.shape
        
        A[i] = random.uniform(0.6,0.9)
        t[i] = random.uniform(0.1,0.9)
        HazeRegionY[i] = random.randint(ImgSize[0]/4, ImgSize[0]/2)
        HazeRegionX[i] = random.randint(ImgSize[1]/4, ImgSize[1]/2)
        HazePointX[i] = random.randint(HazeRegionX[i], (ImgSize[1]-HazeRegionX[i]))
        HazePointY[i] = random.randint(HazeRegionY[i], (ImgSize[0]-HazeRegionY[i]))
        HazeImg = CleanImg
        
        HazeImg[int(HazePointY[i]-HazeRegionY[i]):int(HazePointY[i]+HazeRegionY[i]),
                int(HazePointX[i]-HazeRegionX[i]):int(HazePointX[i]+HazeRegionX[i]),
                :] = CleanImg[int(HazePointY[i]-HazeRegionY[i]):int(HazePointY[i]+HazeRegionY[i]),
                              int(HazePointX[i]-HazeRegionX[i]):int(HazePointX[i]+HazeRegionX[i]),
                              :]*t[i] + A[i]*(1-t[i])
        
        misc.imsave(output_path + CleanImgName.split('\\')[-1].split('.')[0]+'_'+str(i)+'_'+'.png',HazeImg)
        
    return 0 

if __name__ == '__main__':

    imgL_path = 'E:\\Users\\original\\'

    CleanList = glob.glob(imgL_path + '*')
    numImg = len(CleanList)
    print('total num of image is ' + str(numImg))
    
    for i in range(0,numImg):
        CleanImgName = CleanList[i]
        addHaze(CleanImgName, 'test_haze\\')
        print('working on adding haze ' + str(i+1) + '/' + str(numImg) + ' finished')
    
    
