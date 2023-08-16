import enum
import cv2
import os
import numpy as np
import face_alignment
from skimage import io
from tqdm import tqdm
import glob
import random
import threading
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import subprocess
def extract_images(vid_file,ori_imgs_dir,id=None,frame_total=None):
    # print('--- Step1: extract images from vids ---')
    cap = cv2.VideoCapture(vid_file)
    frame_num = 0
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        if id:
            cv2.imwrite(os.path.join(ori_imgs_dir, id + '_' + str(frame_num) + '.jpg'), frame)
        else:
            cv2.imwrite(os.path.join(ori_imgs_dir, id + '_' + str(frame_num) + '.jpg'), frame)
        frame_num = frame_num + 1
        if frame_total:
            if frame_num >= frame_total:
                break
    cap.release()
    # exit()
    
# Step 2: detect lands
def detect_lands(ori_imgs_dir, lmd_dir):
    print('--- Step 2: detect landmarks ---')
    fa = face_alignment.FaceAlignment( face_alignment.LandmarksType.TWO_D, flip_input=False)
    image_list = os.listdir(ori_imgs_dir)
    image_list.sort(key = lambda x: int(x.split('_')[0])*1500 + int(x.split('_')[1][:-4]))
    # print(image_list)
    with tqdm(total = len(image_list)) as pbar:
        for image_path in image_list:
            if image_path.endswith('.jpg') and not os.path.exists(os.path.join(lmd_dir, image_path[:-3] + 'lms')):
                input = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
                preds = fa.get_landmarks(input)
                # print(image_path)
                pbar.set_description(f"Processing '{image_path}'")
                if len(preds) > 0:
                    lands = preds[0].reshape(-1, 2)[:,:2]
                    np.savetxt(os.path.join(lmd_dir, image_path[:-3] + 'lms'), lands, '%f')
            pbar.update(1)  # 更新进度条的进度
                    
def extract_deepspeech_feature(vid_file, audio_dir, id): 
    wav_file = os.path.join(audio_dir, f'{str(id)}.wav')
    if not os.path.exists(wav_file):
        extract_wav_cmd = 'ffmpeg -loglevel quiet -i ' + vid_file + f' -ss 0 -t {1500//25} -f wav -ar 16000 ' + wav_file
        os.system(extract_wav_cmd)
    # extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + audio_dir
    # os.system(extract_ds_cmd)
    # exit()

def get_mp4_files_in_folder(folder_path):
    mp4_files = []
    print(os.path.join(folder_path, '*.mp4'))
    for file in glob.glob(os.path.join(folder_path, '*.mp4')):
        if os.path.isfile(file):
            mp4_files.append(file)
    
    return mp4_files

def step1(ori_imgs_dir  = './data/HDTF/images', start = 1):
    # 得到对应的视频文件
    # all_videos= get_mp4_files_in_folder('/data1/dengkaijun/workdirs/DiffTalk/data/HDTF')
    # all_videos_list = [os.path.basename(vid).split(".")[0] for vid in all_videos]
    # with open('train_name.txt', 'r') as f:
    #     video_list = f.readlines()
    #     video_list = [os.path.basename(vid).split(".")[0] for vid in video_list]
    # test_video = list(set(all_videos_list) - set(video_list))
    # random.shuffle(test_video)
    # # 生成对应的测试name
    # with open('test_name.txt', 'w') as f:
    #     for vid in test_video[:150]:
    #         f.write(vid+'\n')
      
    with open('test_name.txt', 'r') as f:
        video_list = f.read().splitlines()
    # print(video_list)
    # 提取图片
    with tqdm(total = len(video_list)) as pbar:
        for i,vid in enumerate(video_list, start):
            pbar.set_description(f"Processing '{vid}'")
            vid_file = f'/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/{vid}.mp4'
            extract_images(vid_file,ori_imgs_dir,i,1500)
            pbar.update(1)
    return video_list

def step1_multithreaded(video_dir = None, ori_imgs_dir='./data/HDTF/images',  num_threads=4):
    video_list = get_mp4_files_in_folder(video_dir)
    def process_video(vid):
        extract_images(vid, ori_imgs_dir, os.path.basename(vid).split('.')[0], 1500)
        return vid

    with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=len(video_list)) as pbar:
        for result in executor.map(process_video, video_list):
            pbar.set_description(f"Processing '{result}'")
            pbar.update(1)

    return video_list

def process_video(vid, ori_imgs_dir, index):
    vid_file = f'/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/{vid}.mp4'
    extract_images(vid_file, ori_imgs_dir, index, 1500)

def process_image(args):
    image_path, ori_imgs_dir, lmd_dir, fa = args
    if image_path.endswith('.jpg') and not os.path.exists(os.path.join(lmd_dir, image_path[:-3] + 'lms')):
        input = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(os.path.join(lmd_dir, image_path[:-3] + 'lms'), lands, '%f')
    return os.path.basename(image_path)

# 多线程
def detect_lands_multithreaded(ori_imgs_dir, lmd_dir, num_threads = 4, num_gpu = 4):
    print('--- Step 2: detect landmarks ---')
    image_list = os.listdir(ori_imgs_dir)
    image_list.sort(key=lambda x: int(x.split('_')[0]) * 1500 + int(x.split('_')[1][:-4]))

    fa_list = [face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device = f'cuda:{gpu}') for gpu in range(num_gpu)]
    args_list = [(image_path, ori_imgs_dir, lmd_dir, fa_list[i % num_gpu]) for i, image_path in enumerate(image_list)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=len(image_list)) as pbar:
        for result in executor.map(process_image, args_list):
            pbar.set_description(f"Processing '{result}'")
            pbar.update(1)

def step2(ori_imgs_dir  = './data/HDTF/images',
          lmd_dir = './data/HDTF/landmarks',
          ):
    # 提取关键点
    detect_lands(ori_imgs_dir, lmd_dir)
    
def step2_multithreaded(ori_imgs_dir  = './data/HDTF/images',
                        lmd_dir = './data/HDTF/landmarks',
                        num_processes = 4,
                        num_gpu = 1):
    detect_lands_multithreaded(ori_imgs_dir, lmd_dir, num_processes, num_gpu)
      
def step3_multithreaded(video_dir=None, audio_dir='./data/HDTF/audio_smooth', start=1, end =1, num_threads=4):
    print('--- Step3: extract deepspeech feature ---')
    def process_video(id):

        vid_file = os.path.join(video_dir,f'{str(id)}.mp4')
        extract_deepspeech_feature(vid_file, audio_dir, id)
        # extract_ds_cmd = [
        #     'CUDA_VISIBLE_DEVICES=' + str(gpu),
        #     '/data2/dengkaijun/anaconda3/envs/tf/bin/python',
        #     'data_util/deepspeech_features/extract_ds_features.py',
        #     '--input=' + os.path.join(audio_dir, vid_file)
        #     ]

        # subprocess.run(' '.join(extract_ds_cmd), shell=True)
        return str(id) + '.mp4'

    # args_list = [(vid, i) for i, vid in enumerate(video_list, start)]
    args_list = [i for i in range(start,end+1)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=len(args_list)) as pbar:
        for result in executor.map(process_video, args_list):
            pbar.set_description(f"Processing '{result}'")
            pbar.update(1)
    
    extract_ds_cmd = f'CUDA_VISIBLE_DEVICES=1 /data2/dengkaijun/anaconda3/envs/tf/bin/python data_util/deepspeech_features/extract_ds_features.py --input=' + audio_dir
    os.system(extract_ds_cmd)


if __name__ == '__main__':
    root = '/data2/dengkaijun/workdirs/DiffTalk/data/HDTF'
    video_dir = os.path.join(root,'videos')
    ori_imgs_dir  = os.path.join(root,'images')
    lmd_dir = os.path.join(root,'landmarks')
    audio_dir = os.path.join(root,'audio_smooth')
    num_workers = 8
    num_gpu = 8
    start = 1
    end = 248
    video_list = step1_multithreaded(video_dir, ori_imgs_dir, num_workers*2)
    step2_multithreaded(ori_imgs_dir, lmd_dir, num_workers, num_gpu)
    step3_multithreaded(video_dir, audio_dir, start, end, num_workers)