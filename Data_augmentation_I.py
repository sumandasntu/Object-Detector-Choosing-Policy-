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
### BIG Kernel 22*11
RAIN_KERNEL = numpy.array([
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0]])

### BIG Kernel 15*10
RAIN_KERNEL_SMALL = numpy.array([
    [0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.015, 0, 0, 0, 0]])

### BIG Kernel 36*13
RAIN_KERNEL_BIG = numpy.array([
    [0, 0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.18, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.18, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.16, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.16, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.14, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.14, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.12, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.12, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.1, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.1, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.08, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.08, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.06, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.06, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.04, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.04, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0]])




def update_mask(mask: numpy.ndarray, rain_level: float) -> numpy.ndarray:
    """Generate a new rain mask given the previous one.    
    Args: mask - previous rain mask image.
        rain_level - amount of rain at this time instance.
        rain_speed - how fast the rain should fall.        
    Returns:        The new rain mask.    """
    mask_small=mask
    mask_big=mask
    noise = numpy.random.rand(mask.shape[0], mask.shape[1])#noise of imge size 480*640
    new_drops = numpy.zeros((mask.shape[0], mask.shape[1]), dtype=numpy.uint8)
    new_drops_small = numpy.zeros((mask.shape[0], mask.shape[1]), dtype=numpy.uint8)
    new_drops_big = numpy.zeros((mask.shape[0], mask.shape[1]), dtype=numpy.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if noise[i,j] <= rain_level/3:
                new_drops[i,j] = 255
                new_drops_small[i,j]=0
                new_drops_big[i,j]=0
            elif noise[i,j] > rain_level/3 and noise[i,j] <= 2*rain_level/3:
                new_drops[i,j] = 0
                new_drops_small[i,j]=255
                new_drops_big[i,j]=0
            elif noise[i,j] > 2*rain_level/3 and noise[i,j] <= rain_level:
                new_drops[i,j] = 0
                new_drops_small[i,j]=0
                new_drops_big[i,j]=255
            else:
                new_drops[i,j] = 0
                new_drops_small[i,j]=0
                new_drops_big[i,j]=0                
    mask = new_drops
    mask_small = new_drops_small
    mask_big= new_drops_big
    return mask, mask_small, mask_big
    
    
    
def apply_mask(img: numpy.ndarray, mask: numpy.ndarray, type: str) -> numpy.ndarray:
    """Given a rain or snow mask, apply it to the image.    
    Args: img - image to add the mask to.
        mask - single channel mask of rain or snow.
        type - the type of mask to apply        
    Returns: 3-channel color image with the mask applied.    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    if type == "rain":
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL)##mult.of mask & rain_kernel,-1 represent same output size as input
        #c=numpy.count_nonzero(mask)
    elif type == 'small_rain':
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL_SMALL)
        #c=numpy.count_nonzero(mask)
    elif type == 'big_rain':
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL_BIG)
        #c=numpy.count_nonzero(mask)
    mask = cv2.merge((mask, mask, mask))## same mask is merge for 3 color channel, 480*640 will convert to 480*640*3
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)##A stands for opacisty (transparency) of color
    mask[:, :, 3] = 128  ## fixing the opacity to half (range is 0 to 255)
    img = cv2.add(img, mask)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)



def process_folder(folder: str, rain_level: float,  output_d: Tuple[int]) -> None:
    """Add OOD with a given rain level and brightness level to every image
    file in a folder of images.
    Args: folder - path to folder of images.
        mode - divide the files into 4 segments of equal length:
            5: Ramp intensity for all OOD frames
        output_d - (width x height) desired dimensions of output images. """
    #rain_level_mod=[0, .22, .45, .88, 1.2, 1.7, 2.35, 3, 3.8, 5.5, 20]
    rain_level_mod=[0, .27, .55,  .9, 1.3, 1.7, 2.3, 3, 4, 5.75, 15]# three drops
    #rain_level_mod=[0, .25, .75,  1.25, 1.75, 2.25, 3.25, 4.5, 6, 8.5, 25]# Cheating only with Big rain drops
    rain_level1=rain_level_mod[int(rain_level)]*.005
    #rain_level1=rain_level

    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    currrnt_dir = os.getcwd()
    new_dir_name = f"{folder}_rain{rain_level}"    
    new_dir = os.path.join(currrnt_dir, new_dir_name)
    os.mkdir(new_dir)
    if rain_level1 > 0.0:    
        for file in imagelist:
            mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)#matrix of zeros with 480*640
            rain_mask, rain_mask_small, rain_mask_big = update_mask(mask, rain_level1)
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            img = cv2.resize(img, output_d)
            #print(numpy.count_nonzero(rain_mask)+numpy.count_nonzero(rain_mask_small)+ numpy.count_nonzero(rain_mask_big))
            img= apply_mask(img, rain_mask, 'rain')
            img = apply_mask(img, rain_mask_small, 'small_rain')
            img= apply_mask(img, rain_mask_big, 'big_rain')
            image_name = os.path.split(file)[-1]
            new_file = new_dir + '/' + image_name    
            cv2.imwrite(new_file, img)
            
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
        process_folder(dic, j, (640, 480))


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
        process_folder(dic, i, (640, 480))
        
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

