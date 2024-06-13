import zipfile
import cv2
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# defind the path of the zip file
zip_file = '/home/jianglongyu/mydrive/vimeo_septuplet.zip'
extract_to = '/home/jianglongyu/mydrive/vimeo_data'
train_txt_path = '/home/jianglongyu/mydrive/vimeo_data/vimeo_septuplet/sep_trainlist.txt'
test_txt_path = '/home/jianglongyu/mydrive/vimeo_data/vimeo_septuplet/sep_testlist.txt'
train_path = '/home/jianglongyu/mydrive/vimeo_data/train.h5'
test_path = '/home/jianglongyu/mydrive/vimeo_data/test.h5'

# extract the train_list and test_list
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extract('vimeo_septuplet/sep_trainlist.txt', extract_to)
    zip_ref.extract('vimeo_septuplet/sep_testlist.txt', extract_to)

# read the train_list and test_list
def read_file_list(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines[:100]]

train_dirs = read_file_list(train_txt_path)
test_dirs = read_file_list(test_txt_path)

# create a function to extract the frames, create the index, and save them as h5 file
def process_files(dir_list, h5_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        with h5py.File(h5_path, 'w') as h5_file:
            # in this dataset, it has 7 frames in each folder
            total_imgs = len(dir_list) * 7
            dset = h5_file.create_dataset('images', (total_imgs, 256, 448, 3), dtype='uint8')
            index_dset = h5py.string_dtype(encoding='utf-8')
            index_dset = h5_file.create_dataset('index', (total_imgs,), dtype=index_dset)

            index = 0
            for dir_name in tqdm(dir_list, desc=f'Processing {h5_path}'):
                for i in range(1, 8):
                    img_file = f'vimeo_septuplet/sequences/{dir_name}/im{i}.png'
                    try:
                        img_data = zip_ref.read(img_file)
                        # convert the image data to numpy array
                        img_array = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            resizied_img = cv2.resize(img, (448, 256))
                            dset[index] = resizied_img
                            index_dset[index] = f'{dir_name}/im{i}.png'
                            index += 1
                        else:
                            print(f"Error reading image: {img_file}")
                    except KeyError as e:
                        print(f"File not found in zip archive: {img_file}")
                        continue

process_files(train_dirs, train_path)
process_files(test_dirs, test_path)

print("Done!")

# with open(test_txt_path, 'r') as f:
#     lines = f.readlines()
#     print(f"Total lines in {train_txt_path}: {len(lines)}")

# with h5py.File(train_path, 'r') as h5_file:
#     index_dset = h5_file['index']
#     target_path = b'00001/0001/im1.png'
#     indices = np.where(index_dset[:] == target_path)[0]
    
#     if len(indices) > 0:
#         img_index = indices[0]
#         img_dset = h5_file['images']
#         img_data = img_dset[img_index]
#         img = Image.fromarray(img_data, 'RGB')
#         print(type(img))
#     else:
#         print(f"Image {target_path} not found.")