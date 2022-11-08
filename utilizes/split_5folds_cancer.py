import os
import shutil
from tqdm import tqdm

FOLDS = 5
data_path = '/mnt/hungpv/polyps/dataset'
data_names = os.listdir(data_path)

for data_name in tqdm(data_names):
    new_data_name = data_name + '_5folds'
    sub_data_path = os.path.join(data_path, data_name)
    new_data_path = os.path.join(data_path, new_data_name)
    os.makedirs(new_data_path, exist_ok=True)
    print(sub_data_path)
    files_name = os.listdir(f'{sub_data_path}/images')
    files_name = sorted(files_name)
    img_per_fold = int(len(files_name) / 5)
    for i in tqdm(range(FOLDS)):
        new_data_image = f'{new_data_path}/fold{i}/images'
        new_data_mask = f'{new_data_path}/fold{i}/masks'
        os.makedirs(new_data_image, exist_ok=True)
        os.makedirs(new_data_mask, exist_ok=True)

        names = files_name[img_per_fold * i : img_per_fold * (i + 1)]
        old_image_paths = [os.path.join(data_path, data_name) + '/images/' + p.split('.')[0] + '.jpeg' for p in names]
        old_mask_paths = [os.path.join(data_path, data_name) + '/masks/' + p.split('.')[0] + '.png' for p in names]
        for j in range(len(old_image_paths)):
            shutil.copy(old_image_paths[j], new_data_image)
            shutil.copy(old_mask_paths[j], new_data_mask)


