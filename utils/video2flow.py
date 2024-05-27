import imageio
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from multiprocessing import Pool
import csv
from skimage.transform import resize

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().rsplit(' ', 1) for line in lines]
    paths, labels = zip(*data)
    return paths, labels


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--base_root',default='/path/to/Kinetics-600/',type=str)
    parser.add_argument('--save_root_appen',default='Kinetics-600-flow/',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--start_index',default=0,type=int,help='start_index')
    parser.add_argument('--end_index',default=57205,type=int,help='end_index')
    parser.add_argument('--seed',default=0,type=int,help='seed')
    parser.add_argument('--appen',default='',type=str)
    args = parser.parse_args()
    return args

args=parse_args()
np.random.seed(args.seed)

bound = 20

video_list, _ = load_txt_file('HMDB-rgb-flow/splits/Kinetics_all.txt')

args.data_root = args.base_root + 'Kinetics-600-train/'
args.save_root = args.base_root + args.save_root_appen

filenames = video_list[args.start_index:args.end_index]
print(filenames)

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

def dense_flow(filenames):
    for file in filenames:
        file = file.replace('\n', '')
        print(file)
        split_parts = file.split('/')
        subfolder = split_parts[0]  
        file_name = split_parts[1]
        save_mp4_folder = args.save_root + subfolder + '/'
        if not os.path.exists(save_mp4_folder):
            os.makedirs(save_mp4_folder)

        flow_name = args.save_root + file[:-4] + '_flow_x.mp4'
        flow_name2 = args.save_root + file[:-4] + '_flow_y.mp4'
        if os.path.exists(flow_name) and os.path.exists(flow_name2):
            continue

        check_root = args.base_root + 'tmp/'
        directories = os.listdir(check_root)

        directory_exist = False
        for directory in directories:
            if directory.endswith(file_name[:-4]) and os.path.isdir(os.path.join(check_root, directory)):
                files_exist = os.listdir(os.path.join(check_root, directory))
                start_frame = int(len(files_exist)/2)-1
                tmp_root = check_root + directory + '/'
                directory_exist = True
                
        if not directory_exist:
            rnd = np.random.rand()
            tmp_root = args.base_root + 'tmp/tmp_'+str(rnd)+ '_'+args.appen + '_'+ file_name[:-4]+ '/'
            start_frame = 0

        if not os.path.exists(tmp_root):
            os.makedirs(tmp_root)

        filename = args.data_root + file
        vid = imageio.get_reader(filename,  'ffmpeg', fps=24)

        #print(len(list(enumerate(vid))))
        frame_num = len(list(enumerate(vid)))
        end_frame = frame_num-1

        for i in range(start_frame, end_frame):
            #print(i)
            try:
                prev_image = vid.get_data(i)
                image = vid.get_data(i+1)
            except:
                continue

            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)

            dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
            flowDTVL1=dtvl1.calc(prev_gray,gray,None)

            flow_x=ToImg(flowDTVL1[...,0],bound)
            flow_y=ToImg(flowDTVL1[...,1],bound)

            save_x=os.path.join(tmp_root,'flow_x_{:05d}.jpg'.format(i))
            save_y=os.path.join(tmp_root,'flow_y_{:05d}.jpg'.format(i))
            flow_x_img=Image.fromarray(flow_x)
            flow_y_img=Image.fromarray(flow_y)
            imageio.imwrite(save_x,flow_x_img)
            imageio.imwrite(save_y,flow_y_img)


        name = 'flow_x'
        files = os.listdir(tmp_root)
        files.sort()
        fileList = []
        for f in files:
            if f.startswith(name):
                complete_path = tmp_root + f
                fileList.append(complete_path)
        flow_save = args.save_root + file[:-4] + '_flow_x.mp4'
        writer = imageio.get_writer(flow_save, fps=24)

        img_0 = imageio.imread(fileList[0])
        h_ = img_0.shape[0]
        w_ = img_0.shape[1]

        for im in fileList:
            img = imageio.imread(im)
            if img.shape[0] != h_ or img.shape[1] != w_:
                img = resize(img, (h_, w_))
            writer.append_data(img)
        writer.close()


        fileList = []
        name = 'flow_y'
        for f in files:
            if f.startswith(name):
                complete_path = tmp_root + f
                fileList.append(complete_path)
        flow_save = args.save_root + file[:-4] + '_flow_y.mp4'
        writer = imageio.get_writer(flow_save, fps=24)

        for im in fileList:
            img = imageio.imread(im)
            if img.shape[0] != h_ or img.shape[1] != w_:
                img = resize(img, (h_, w_))
            writer.append_data(img)
        writer.close()

        cmd = ['rm -rf', tmp_root]
        os.system(' '.join(cmd))


pool=Pool(args.num_workers)
pool.map(dense_flow, zip(filenames))
