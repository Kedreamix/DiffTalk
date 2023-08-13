import os
from tqdm import tqdm

# 指定文件夹路径和输出文件名
root = '/data2/dengkaijun/workdirs/DiffTalk/data/HDTF'
folder_path = os.path.join(root,'images')
train_output_file = 'data_train.txt'
test_output_file = 'data_test.txt'

# 获取文件夹中的图片文件列表
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 对文件名进行排序
sorted_image_files = sorted(image_files, key = lambda x: int(x.split('_')[0])*1500 + int(x.split('_')[1][:-4]))

max_numbers = {}  # 用于存储每个前缀的最大数字
for file in tqdm(sorted_image_files):
    file = os.path.basename(file).split('.')[0]
    prefix, number = file.split('_')
    number = int(number)
    
    if prefix not in max_numbers or number > max_numbers[prefix]:
        max_numbers[prefix] = number

print(max_numbers)
# 将每个前缀的最大数字写入文件
with open('data.txt', 'w') as f:
    for prefix, max_number in max_numbers.items():
        f.write(f"{prefix} {max_number}\n")

print("Max numbers saved to data.txt")

# DATA_LEN = len(max_numbers)
# split = 0.5
# train_len = (DATA_LEN*split)

# 将排序后的文件名写入输出文件
with open(train_output_file, 'w') as f:
    with tqdm(total = len(sorted_image_files)) as pbar:
        for file_name in sorted_image_files:
            pbar.set_description(os.path.basename(file_name).split('.')[0])
            f.write(os.path.basename(file_name).split('.')[0] + '\n')
            pbar.update(1)

print("Sorted image names written to", train_output_file)