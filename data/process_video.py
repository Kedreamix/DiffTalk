import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob

def get_mp4_files(directory):
    mp4_files = glob.glob(directory + '/*.mp4')
    return mp4_files
def change_mp4_fps_threaded(vid_folder, target_folder, num_threads=4):
    # with open('train_name.txt', 'r') as f:
    #     train_video_list = f.read().splitlines()

    # with open('test_name.txt', 'r') as f:
    #     test_video_list = f.read().splitlines()
    # mp4_names = train_video_list + test_video_list
    mp4_names = get_mp4_files(vid_folder)
    print(mp4_names)
    def process_mp4(arg):
        mp4_name, idx = arg
        mp4_path = os.path.join(mp4_name)
        new_mp4_name = f"{idx}.mp4"
        new_mp4_path = os.path.join(target_folder, new_mp4_name)
        # print(mp4_path, new_mp4_path)
        # if os.path.exists(new_mp4_path):
        #     return new_mp4_name

        # 复制并重命名mp4文件到目标文件夹
        # shutil.copy(mp4_path, new_mp4_path)
    
        command = f'ffmpeg -loglevel quiet -i {mp4_path} -vf fps=fps=25 -c:a copy {new_mp4_path}'
        os.system(command)
        print(f"Moved and renamed {mp4_path} to {new_mp4_path}")
        return new_mp4_name
    
    args_list = [(vid, i) for i, vid in enumerate(mp4_names, start=1)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=len(args_list)) as pbar:
        for result in executor.map(process_mp4, args_list):
            pbar.set_description(f"Processing '{result}'")
            pbar.update(1)
    

if __name__ == "__main__":
    vid_folder_path = "./HDTF/train_videos"
    target_folder_path = "./videos"
    os.makedirs(target_folder_path, exist_ok=True)
    num_threads = 8  # Number of threads to use
    
    change_mp4_fps_threaded(vid_folder_path, target_folder_path, num_threads)
