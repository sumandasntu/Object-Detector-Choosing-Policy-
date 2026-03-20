import os, random
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide a Data folder')
    parser.add_argument('--path', default=' ', type=str, help='path to the data')
    args = parser.parse_args()

head_tail = os.path.split(args.path)
path=head_tail[0]
data=head_tail[1]

#Creating Train and Test Folder

list = ['TrainData','TestData','Train','Test_Lightness','Test_Rain']
for items in list: 
    os.mkdir(path+'/'+items)    

#Spliting total data into TrainData and TestData by 80:20 ration

for i in range(2):
    src_dir =path+'/'+data+'/'
    dst_dir =path+'/'+list[i]+'/'
    file_list = os.listdir(src_dir)
    print(len(file_list))
    if i==0:
        num=int(len(file_list)*8/10)
    else:
        num=len(file_list)
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    print(len(file_list))
    file_list = os.listdir(dst_dir)
    print(len(file_list))
shutil.rmtree(path+'/'+data+'/')


# Directory
list1 = ['a-5', 'b-4', 'c-3', 'd-2', 'e-1', 'f0', 'g1', 'h2', 'i3', 'j4', 'k5', 'l0','m10','n20','o30','p40','q50']
list2 = ['a-10', 'b-9', 'c-8', 'd-7', 'e-6', 'f-5', 'g-4', 'h-3', 'i-2', 'j-1', 'k0', 'l1', 'm2', 'n3', 'o4', 'p5', 'q6',
        'r7', 's8', 't9', 'u10']
list3=['a0','b10','c20','d30','e40','f50','g60','h70','i80','j90','k100']


#Creating directories in Train and test folders for weakly supervision and testing

parent_dir =path+'/'+list[2]+'/'
for items in list1:
    path1 = os.path.join(parent_dir, items)  
    os.mkdir(path1)
parent_dir =path+'/'+list[3]+'/'
for items in list2:
    path1 = os.path.join(parent_dir, items)  
    os.mkdir(path1)
parent_dir =path+'/'+list[4]+'/'
for items in list3:
    path1 = os.path.join(parent_dir, items)  
    os.mkdir(path1)
    
    
#Moving training data into different folders of train data

src_dir =path+'/'+list[0]+'/'
N=len(os.listdir(src_dir))
for i in range(len(list1)):
    dst_dir =path+'/'+list[2]+'/'+list1[i]+'/'
    file_list = os.listdir(src_dir)
    if i<11:
        num=int(N/(2*11))
    else:
        num=int(N/(2*6))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
shutil.rmtree(path+'/'+list[0]+'/')

#Moving half of testing data into different folders of test lightness

src_dir =path+'/'+list[1]+'/'
N=int(len(os.listdir(src_dir))/2)
for i in range(len(list2)):
    dst_dir = path+'/'+list[3]+'/'+list2[i]+'/'
    file_list = os.listdir(src_dir)
    num=int(N/(21))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
    
#Moving remaining testing data into different folders of test rain

src_dir =path+'/'+list[1]+'/'
N=int(len(os.listdir(src_dir)))
for i in range(len(list3)):
    dst_dir = path+'/'+list[4]+'/'+list3[i]+'/'
    file_list = os.listdir(src_dir)
    num=int(N/(11))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
shutil.rmtree(path+'/'+list[1]+'/')
