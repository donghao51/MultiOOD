import imageio.v2 as imageio
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm 

def images_to_video(input_folder, output_folder, flow_x=True):
    for subfolder in tqdm(os.listdir(input_folder), desc="Converting to videos"):
        subfolder_path = os.path.join(input_folder, subfolder)

        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            files.sort()
            fileList = []
            for f in files:
                complete_path = subfolder_path + '/' + f
                fileList.append(complete_path)
            if flow_x:
                video_name = f"{subfolder}_flow_x.mp4"
            else:
                video_name = f"{subfolder}_flow_y.mp4"
            video_path = os.path.join(output_folder, video_name)
            writer = imageio.get_writer(video_path, fps=24)

            for im in fileList:
                writer.append_data(imageio.imread(im))
            writer.close()


if __name__ == "__main__":
    input_folder = 'hmdb51_tvl1_flow/tvl1_flow/u/' # change to hmdb51_tvl1_flow
    input_folder2 = 'hmdb51_tvl1_flow/tvl1_flow/v/' # change to hmdb51_tvl1_flow
    output_folder = 'path/to/hmdb51/flow/' # change to saved path

    os.makedirs(output_folder, exist_ok=True)
    images_to_video(input_folder, output_folder, flow_x=True)
    images_to_video(input_folder2, output_folder, flow_x=False)


