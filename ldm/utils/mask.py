import PIL
from PIL import Image
import numpy as np
import face_alignment
from skimage import io
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import cv2
import torch
from torchvision import transforms
mp_face_mesh = mp.solutions.face_mesh

def face_mask(img_shape, landmark_list, dtype='uint8'):
    height, width = img_shape[:2]
    mask = np.zeros((height, width, 1), dtype=dtype)
    cv2.drawContours(mask, np.int32([landmark_list[2:15]]), -1, color=(1), thickness=cv2.FILLED)
    # cv2.polylines(mask, np.int32([landmark_list[48:60]]), isClosed=False, color=(0), thickness=1)  # 'Outer Mouth'
    # cv2.polylines(mask, np.int32([landmark_list[60:68]]), isClosed=False, color=(0), thickness=1)  # 'Inner Mouth'
    # cv2.line(mask, np.int32(landmark_list[48]), np.int32(landmark_list[59]), color=(0), thickness=1)
    # cv2.line(mask, np.int32(landmark_list[60]), np.int32(landmark_list[67]), color=(0), thickness=1)
    return mask

def face_mask_square(img_shape, landmark_list, dtype='uint8'):

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)

    cv2.drawContours(mask, np.int32([[landmark_list[1],[landmark_list[1][0], height],landmark_list[15],[landmark_list[15][0], height], landmark_list[1],landmark_list[15],[landmark_list[1][0], height],[landmark_list[15][0], height]]]), -1, color=(1), thickness=cv2.FILLED)

    return mask

def diff_face_mask(image_path_, size = 256):
    image = Image.open(image_path_)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    image = Image.fromarray(img)
    h, w = image.size
    if size is not None:
        image = image.resize((size, size), resample=PIL.Image.BICUBIC)

    image = np.array(image).astype(np.uint8)
    # image = (image / 127.5 - 1.0).astype(np.float32)

    # landmarks = np.loadtxt(example["landmark_path_"], dtype=np.float32)
    landmarks = get_landmarks(image_path_)
    landmarks_img = landmarks[13:48]
    landmarks_img2 = landmarks[0:4]
    landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
    scaler = h / size
    
    #mask
    mask = np.ones((size, size))
    mask[(landmarks[30][1]).astype(int):, :] = 0.
    mask = mask[..., None]
    print(image.shape, mask.shape)
    image_mask = (image * mask).astype(np.uint8)
    # image_mask = (image_mask / 127.5 - 1.0).astype(np.float32)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(16, 16))

    # Display the image
    ax.imshow(image_mask)  # Convert back to [0, 1] range for display

    # Plot landmarks
    for i, landmark in enumerate(landmarks_img):
        x, y = landmark[0], landmark[1]
        ax.scatter(x, y, c='red', s=10)
        ax.text(x, y, str(i), fontsize=8, color='red', ha='center', va='bottom')

    # # Save the images
    base_name = os.path.basename(image_path_).split('.')[0]
    lm_filename = f"{base_name}_mask.jpg"
    alm_filename = f"{base_name}_lm.jpg"

    # # Save the image with landmarks
    plt.savefig(lm_filename, bbox_inches='tight', pad_inches=0)
    plt.clf()  # Clear the figure for the next plot
    
    fig, ax = plt.subplots(1, figsize=(16, 16))
    # Plot landmarks
    ax.imshow(image)
    for i, landmark in enumerate(landmarks):
        x, y = landmark[0], landmark[1]
        ax.scatter(x, y, c='red', s=10)
        ax.text(x, y, str(i), fontsize=8, color='red', ha='center', va='bottom')
    plt.savefig(alm_filename, bbox_inches='tight', pad_inches=0)
    plt.clf()  # Clear the figure for the next plot
    
    
def get_landmarks(image_path_):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    # 读取图像
    input = cv2.imread(image_path_)
    
    # 调整图像大小
    desired_size = (256, 256)  # 设置目标尺寸为256x256
    input = cv2.resize(input, desired_size)
    
    # 执行人脸检测
    preds = fa.get_landmarks(input)
    
    if len(preds) > 0:
        lands = preds[0].reshape(-1, 2)[:,:2]
        return lands


def contour_extractor(path_to_img):
    # 68个关键点的索引
    landmark_points_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71,
                          63, 105, 66, 107, 336,
                          296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144,
                          362, 385, 387, 263, 373,
                          380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317,
                          14, 87]
    # 图片路径列表
    IMAGE_LIST = [path_to_img]
    # 使用FaceMesh模型
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1) as face_mesh:
        # 遍历图片路径列表
        for file in (IMAGE_LIST):
            # 读取图片
            image = cv2.imread(file)
            # 将BGR图像转换为RGB进行处理
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 在图像上打印和绘制面部关键点
            if not results.multi_face_landmarks:
                # 如果没有检测到面部关键点，则创建一个空的关键点列表
                frame_landmark_list = np.zeros((468,3))
            else:
                for face_landmarks in results.multi_face_landmarks:
                    # 创建一个空的关键点列表
                    frame_landmark_list = []
                    # 遍历468个关键点
                    for i in range(0, 468):
                        pt1 = face_landmarks.landmark[i]
                        x = pt1.x
                        y = pt1.y
                        z = pt1.z
                        # 将关键点的坐标添加到列表中
                        frame_landmark_list.append([x, y, z])
                    frame_landmark_list = np.asarray(frame_landmark_list)

            # 提取68个关键点
            landmarks_extracted = frame_landmark_list[landmark_points_68]
            landmarks_extracted = np.asarray(landmarks_extracted)

            # 对关键点进行缩放和归一化
            landmarks_extracted[:, 0] = landmarks_extracted[:, 0] * 256
            landmarks_extracted[:, 1] = landmarks_extracted[:, 1] * 256
        

            # 只保留前两个维度的坐标
            landmarks_extracted = landmarks_extracted[:, :2]

            # 创建一个空的关键点列表
            landmark_list = []
            # 将关键点坐标转换为整数，并添加到列表中
            for items in landmarks_extracted:
                tuple = [int(items[0]), int(items[1])]
                landmark_list.append(tuple)

    return landmark_list




def diff_face_mask2(image_path, image_size = [256,256]):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.uint8)
    img = Image.fromarray(img)
    # print(img.size)
    if image_size is not None:
        img = img.resize((image_size[0], image_size[1]), resample=PIL.Image.BICUBIC)
    
    img = np.array(img).astype(np.uint8)
    
    # fig, ax = plt.subplots(1, figsize=(16, 16))
    # ax.imshow(img)
    # plt.savefig('test.jpg', bbox_inches='tight', pad_inches=0)
    
    landmark_list = contour_extractor(image_path)
    mask = face_mask(image_size, landmark_list)
    
    # show_image(mask, 'mask.jpg')
    # cond_image = img*(1. - mask) + mask*torch.randn_like(img)
    mask_img = img*(np.ones_like(mask) - mask) + mask
    # print(max(mask_img.), min(mask_img.all()))
    
    base_name = os.path.basename(image_path).split('.')[0]
    output_path = f"{base_name}_mask2.jpg"
    show_image(mask_img, output_path)
    # show_image(img, 'img.jpg')

def show_image(img, output_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.imshow(img, cmap = 'gray')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    
# Example usage
image_path = '/data/dengkaijun/workdirs/DiffTalk/data/HDTF/images/1_432.jpg'
diff_face_mask(image_path)
diff_face_mask2(image_path)
