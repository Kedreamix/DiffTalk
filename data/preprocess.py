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
def extract_images(vid_file,ori_imgs_dir,id=None,frame_total=None):
    # print('--- Step1: extract images from vids ---')
    cap = cv2.VideoCapture(vid_file)
    frame_num = 0
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        if id:
            cv2.imwrite(os.path.join(ori_imgs_dir, str(id) + '_' + str(frame_num) + '.jpg'), frame)
        else:
            cv2.imwrite(os.path.join(ori_imgs_dir, str(id) + '_' + str(frame_num) + '.jpg'), frame)
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
                    
def extract_deepspeech_feature(vid_file,audio_dir,id): 
    print('--- Step0: extract deepspeech feature ---')
    wav_file = os.path.join(audio_dir, f'{id}.wav')
    extract_wav_cmd = 'ffmpeg -i ' + vid_file + ' -ss 0 -t 50 -f wav -ar 16000 ' + wav_file
    os.system(extract_wav_cmd)
    # extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + audio_dir
    # os.system(extract_ds_cmd)
    # exit()


def get_mp4_files_in_folder(folder_path):
    mp4_files = []
    
    for file in glob.glob(os.path.join(folder_path, '*.mp4')):
        if os.path.isfile(file):
            mp4_files.append(file)
    
    return mp4_files


def step1(ori_imgs_dir  = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/images', start = 1):
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

def step1_multithreaded(ori_imgs_dir='/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/images', start=1, num_threads=4):
    with open('test_name.txt', 'r') as f:
        video_list = f.read().splitlines()

    def process_range(start_idx, end_idx):
        with tqdm(total=end_idx - start_idx) as pbar:
            for i in range(start_idx, end_idx):
                pbar.set_description(f"Processing '{video_list[i]}'")
                process_video(video_list[i], ori_imgs_dir, i + start)
                pbar.update(1)

    thread_list = []
    videos_per_thread = len(video_list) // num_threads

    for i in range(num_threads):
        start_idx = i * videos_per_thread
        end_idx = start_idx + videos_per_thread if i != num_threads - 1 else len(video_list)
        thread = threading.Thread(target=process_range, args=(start_idx, end_idx))
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    return video_list

def process_video(vid, ori_imgs_dir, index):
    vid_file = f'/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/{vid}.mp4'
    extract_images(vid_file, ori_imgs_dir, index, 1500)


def process_image(args):
    image_path, ori_imgs_dir, lmd_dir = args
    if image_path.endswith('.jpg') and not os.path.exists(os.path.join(lmd_dir, image_path[:-3] + 'lms')):
        
        input = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(os.path.join(lmd_dir, image_path[:-3] + 'lms'), lands, '%f')


def detect_lands_multithreaded(ori_imgs_dir, lmd_dir, num_processes = 4):
    print('--- Step 2: detect landmarks ---')
    image_list = os.listdir(ori_imgs_dir)
    image_list.sort(key=lambda x: int(x.split('_')[0]) * 1500 + int(x.split('_')[1][:-4]))

    # num_processes = 4  # Define the number of processes you want to use
    args_list = [(image_path, ori_imgs_dir, lmd_dir) for image_path in image_list]

    with Pool(num_processes) as pool, tqdm(total=len(image_list)) as pbar:
        for _ in pool.imap_unordered(process_image, args_list):
            pbar.update(1)

def step2(lmd_dir = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/landmarks',
          ori_imgs_dir  = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/images'):
    # 提取关键点
    detect_lands(ori_imgs_dir, lmd_dir)
    
def step2_multithreaded(lmd_dir = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/landmarks',
                        ori_imgs_dir  = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/images',
                        num_processes = 4):
    detect_lands_multithreaded(ori_imgs_dir, lmd_dir, num_processes)
    
    
def step3(video_list = None, audio_dir = '/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/audio_smooth', start=1):
    for i,vid in tqdm(enumerate(video_list,start)):
        vid_file = f'/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/{vid}.mp4'
        extract_deepspeech_feature(vid_file,audio_dir,i)
    extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + audio_dir
    os.system(extract_ds_cmd)

if __name__ == '__main__':
    root = r'/data1/dengkaijun/workdirs/DiffTalk/data/HDTF/'
    ori_imgs_dir  = os.path.join(root,'images')
    lmd_dir = os.path.join(root,'landmarks')
    audio_dir = os.path.join(root,'audio_smooth')
    start = 99
    # test_video_list = step1(start=99)
    # step2()
    
    # test_video_list = step1_multithreaded(start=99)
    step2_multithreaded()
    # step3(video_list = test_video_list, start = 99)