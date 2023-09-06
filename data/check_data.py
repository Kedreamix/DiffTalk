import os
from tqdm import tqdm
root = '../data/HDTF'
video_dir = os.path.join(root, 'videos')
ori_imgs_dir = os.path.join(root, 'images')
lmd_dir = os.path.join(root, 'landmarks')
audio_dir = os.path.join(root, 'audio_smooth')

# 获取原始图像文件列表
ori_imgs_files = os.listdir(ori_imgs_dir)

for img_file in tqdm(ori_imgs_files):
    # 提取文件名字
    img_filename = os.path.splitext(img_file)[0]
    if int(img_filename.split('_')[0]) > 75:
        continue
    # 在landmarks目录中查找对应的.lms文件
    lms_file = img_filename + '.lms'
    lms_path = os.path.join(lmd_dir, lms_file)
    if os.path.exists(lms_path):
        # 找到了对应的.lms文件
        # print(f"Found {lms_file} in landmarks directory.")
        pass
    else:
        print(f"{lms_file} not found in landmarks directory.")
    
    # # 在audio_smooth目录中查找对应的.npy文件
    # npy_file = img_filename + '.npy'
    # npy_path = os.path.join(audio_dir, npy_file)
    # if os.path.exists(npy_path):
    #     # 找到了对应的.npy文件
    #     print(f"Found {npy_file} in audio_smooth directory.")
    # else:
    #     print(f"{npy_file} not found in audio_smooth directory.")