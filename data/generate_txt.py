import os
from tqdm import tqdm

# 指定文件夹路径和输出文件名
root = '../data/HDTF'
folder_path = os.path.join(root,'audio_smooth')
train_output_file = 'my_data_train.txt'
test_output_file = 'my_data_test.txt'

# 获取文件夹中的文件列表
files = [file for file in tqdm(os.listdir(folder_path)) if file.lower().endswith(('npy')) and '_' in file]

# 对文件名进行排序
sorted_files = sorted(files, key = lambda x: int(x.split('_')[0])*1500 + int(x.split('_')[1][:-4]))

max_numbers = {}  # 用于存储每个前缀的最大数字
for file in tqdm(sorted_files):
    file = os.path.basename(file).split('.')[0]
    prefix, number = file.split('_')
    number = int(number)
    
    if prefix not in max_numbers or number > max_numbers[prefix]:
        max_numbers[prefix] = number

print(max_numbers)
# 将每个前缀的最大数字写入文件
with open(os.path.join(root, 'my_data.txt') , 'w') as f:
    for prefix, max_number in max_numbers.items():
        f.write(f"{prefix} {max_number}\n")

print("Max numbers saved to data.txt")

train_prefixes = [prefix for prefix, max_number in tqdm(max_numbers.items()) if max_number == 1499]
test_prefixes = [prefix for prefix, max_number in tqdm(max_numbers.items()) if max_number != 1499]
# print(train_prefixes)
# 将排序后的文件名写入输出文件
f_train = open(train_output_file, 'w')
f_test = open(test_output_file, 'w')

with tqdm(total = len(sorted_files)) as pbar:
    for file in sorted_files:
        pbar.set_description(os.path.basename(file).split('.')[0])
        pbar.update(1)
        file_name = os.path.basename(file).split('.')[0]
        # print(file_name.split('_'))
        if file_name.split('_')[0] in train_prefixes:
            f_train.write(file_name + '\n')
        else:
            f_test.write(file_name + '\n')
        
f_train.close()
f_test.close()

print("Sorted image names written to", train_output_file, test_output_file)