"""Module for bulk generation of rainy images following the algorithm in
https://www.photoshopessentials.com/photo-effects/photoshop-weather-effects-rain/,
which appeared in:

X. Fu, Q. Qi, Z. -J. Zha, X. Ding, F. Wu, and J. Paisley, ``Successive graph
convolutional network for image de-raining,'' International Journal of Computer
Vision, vol. 129, pp. 1691--1711, May 2021, doi: 10.1007/s11263-020-01428-6.

H. Yin, F. Zheng, H. -F. Duan, D. Savic, and Z. Kapelan, ``Estimating Rainfall
Intensity Using an Image-Based Deep Learning Model,'' Engineering, vol. 21,
pp. 162--174, Feb. 2023, doi: 10.1016/j.eng.2021.11.021.


L. Wang, H. Qin, X. Zhou, X. Lu and F. Zhang, ``R-YOLO: A Robust Object
Detector in Adverse Weather,'' IEEE Transactions on Instrumentation and
Measurement, vol. 72, pp. 1--11, Dec. 2022, doi: 10.1109/TIM.2022.3229717.
"""
### Written by Michael Yuhas

import cv2
import numpy
from typing import Tuple
import argparse
from glob import glob
import cv2
import numpy
import os, random
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide path to training and test data')
    parser.add_argument('--path', default='/', type=str, help='Input folder name only ')
    parser.add_argument('--percentage', default=25, type=int, help='Percentage of labelled data ')
    args = parser.parse_args()
    
path=args.path
p=args.percentage     ###percentage of label data

angle=90
drop_length=75
drop_size=4
black_point=40
white_point=200
def get_rain_mask(size: Tuple[int, int], amount: float = 25) -> numpy.ndarray:
    mask = numpy.random.normal(loc=amount, scale=255 / 3, size=size)
    #print(mask)
    mask = cv2.resize(mask, (size[1] * drop_size + 2 * drop_length, size[0] * drop_size + 2 * drop_length))
    mask = mask[:size[0] + 2 * drop_length, :size[1] + 2 * drop_length]
    kernel = numpy.zeros((drop_length, drop_length))
    kernel[drop_length // 2, :] = numpy.ones(drop_length, dtype=numpy.float32)
    rotation_matrix = cv2.getRotationMatrix2D((drop_length / 2, drop_length /2), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, kernel.shape[::-1])
    kernel = kernel * (1.0 / numpy.sum(kernel))
    mask = cv2.filter2D(mask, -1, kernel)
    mask = mask[drop_length:size[0]+drop_length, drop_length:size[1]+drop_length]
    mask = 255 * (mask - black_point) / (white_point - black_point)
    mask = numpy.clip(mask, 0, 255)
    mask = numpy.repeat(mask[:, :, numpy.newaxis], 3, axis=2)
    return mask

def apply_mask(img, mask):
    img = img.astype(numpy.float32) / 255
    mask = mask / 255
    img = 1 - (1 - img) * (1 - mask)
    return (img * 255).astype(numpy.uint8)
    
    
def process_folder(folder: str, rain_level: float,  output_d: Tuple[int]) -> None:
    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    currrnt_dir = os.getcwd()
    new_dir_name = f"{folder}_rain{rain_level}"    
    new_dir = os.path.join(currrnt_dir, new_dir_name)
    os.mkdir(new_dir)    
    for f in imagelist:
        img = cv2.imread(os.path.join(folder, f))
        mask = get_rain_mask(size=img.shape[0:2], amount=rain_level)
        img = apply_mask(img, mask)
        image_name = os.path.split(f)[-1]
        new_file = new_dir + '/' + image_name    
        cv2.imwrite(new_file, img)

        
##############################################
def adjust_lightness(img: numpy.ndarray, level: float) -> numpy.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    level=level/10
    if level<0:
        img[:, :, 1] = cv2.multiply(img[:, :, 1], 1+level)
    else:
        img[:, :, 1] = cv2.multiply(img[:, :, 1], 1-level)
        img[:, :, 1] = cv2.add(img[:, :, 1], 255*level)
    return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    
def process_lightness(folder: str, lightness_level: float,  output_d: Tuple[int]) -> None:

    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    currrnt_dir = os.getcwd()
    new_dir_name = f"{folder}_lightness{lightness_level}"    
    new_dir = os.path.join(currrnt_dir, new_dir_name)
    os.mkdir(new_dir)
    for file in imagelist:
        mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)#matrix of zeros with 480*640
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, output_d)
        img = adjust_lightness(img, lightness_level)
        image_name = os.path.split(file)[-1]
        new_file = new_dir + '/' + image_name    
        cv2.imwrite(new_file, img)


list0 = ['TrainData','TestData','Train','Test_Lightness','Test_Rain']
list1 = ['a-5', 'b-4', 'c-3', 'd-2', 'e-1', 'f0', 'g1', 'h2', 'i3', 'j4', 'k5', 'l0', 'm10','n20','o30','p40','q50']
list2 = ['a-10', 'b-9', 'c-8', 'd-7', 'e-6', 'f-5', 'g-4', 'h-3', 'i-2', 'j-1', 'k0', 'l1', 'm2', 'n3', 'o4', 'p5', 'q6',
        'r7', 's8', 't9', 'u10']
list3=['a0','b10','c20','d30','e40','f50','g60','h70','i80','j90','k100']


#Augmentation of Training data
for i in range(len(list1)):
    if i<=10:
        dic=path+'/'+list0[2]+'/'+list1[i]
        j=i-5
        process_lightness(dic, j, (640, 480))
    elif i>11:
        dic=path+ '/'+list0[2]+'/'+list1[i]
        j=i-11
        process_folder(dic, j*10, (640, 480))


#Deletion of non-augmentated folder

for i in range(len(list1)):
    if i!=11:
        shutil.rmtree(path+ '/'+list0[2]+'/'+list1[i]+'/')  


#Adding a unlabelled folder to training set---for weakly supervision

path1 = os.path.join(path+'/'+list0[2]+'/', 'unlabelled')  
os.mkdir(path1)

#Transfering certaing percentage of data to the unlabelled folder

L=os.listdir(path=path+'/'+list0[2])
L=sorted(L)
for i in range(len(L)-1):
    src_dir =path+'/'+list0[2]+'/'+L[i]+'/'
    dst_dir =path+'/'+list0[2]+'/'+'unlabelled'+'/'
    file_list = os.listdir(src_dir)
    num=int(len(file_list)*(100-p)/100)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)

#Lightness augmentation of test data
for i in range(len(list2)):
    dic=path+'/'+list0[3]+'/'+list2[i]
    j=i-10
    process_lightness(dic, j, (640, 480))
    
#Deleting non-augmented data

for i in range(len(list2)):
    shutil.rmtree(path+'/'+list0[3]+'/'+list2[i]+'/')

    
#Rain augmentation of test data
for i in range(len(list3)):
    if i>0:
        dic=path+'/'+list0[4]+'/'+list3[i]
        process_folder(dic, i*10, (640, 480))
        
#Deleting non-augmented data
for i in range(1,len(list3)):
    shutil.rmtree(path+'/'+list0[4]+'/'+list3[i]+'/')
    
##AUROC folder
os.mkdir(args.path+'/'+'AUROC_Rain')
os.mkdir(args.path+'/'+'AUROC_Lightness')
os.mkdir(args.path+'/'+'AUROC_Rain'+'/'+'ID')
os.mkdir(args.path+'/'+'AUROC_Rain'+'/'+'OOD')
os.mkdir(args.path+'/'+'AUROC_Lightness'+'/'+'ID')
os.mkdir(args.path+'/'+'AUROC_Lightness'+'/'+'OOD')
L2 = sorted(os.listdir(path+'/'+'Test_Lightness'))
L3=sorted(os.listdir(path+'/'+'Test_Rain'))
for i in range(11):
    if i<=5:
        shutil.move(path+'/'+'Test_Rain'+'/'+L3[i], path+'/'+'AUROC_Rain'+'/'+'ID'+'/')
    else:
        shutil.move(path+'/'+'Test_Rain'+'/'+L3[i], path+'/'+'AUROC_Rain'+'/'+'OOD'+'/')
        
for i in range(21):
    if i<=4 or i>=16:
        shutil.move(path+'/'+'Test_Lightness'+'/'+L2[i], path+'/'+'AUROC_Lightness'+'/'+'OOD'+'/')
    else:
        shutil.move(path+'/'+'Test_Lightness'+'/'+L2[i], path+'/'+'AUROC_Lightness'+'/'+'ID'+'/')
 

