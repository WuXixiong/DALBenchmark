import os
import shutil

def load_class_mapping(wnids_path):
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}
    return wnid_to_label

def process_train_set(train_dir, class_mapping):
    for wnid in os.listdir(train_dir):
        wnid_path = os.path.join(train_dir, wnid)
        if not os.path.isdir(wnid_path):
            continue
        class_label = class_mapping[wnid]
        images_dir = os.path.join(wnid_path, 'images')
        
        for img_file in os.listdir(images_dir):
            if img_file.endswith('.JPEG'):
                img_path = os.path.join(images_dir, img_file)
                new_img_path = f'tiny-imagenet/new_train/{class_label}/{img_file}'
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                shutil.copy(img_path, new_img_path)

def process_val_set(val_dir, val_annotations_path, class_mapping):
    with open(val_annotations_path, 'r') as f:
        annotations = [line.strip().split('\t') for line in f.readlines()]

    for annotation in annotations:
        img_file, wnid = annotation[0], annotation[1]
        class_label = class_mapping[wnid]
        img_path = os.path.join(val_dir, 'images', img_file)
        
        new_img_path = f'tiny-imagenet/new_val/{class_label}/{img_file}'
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        shutil.copy(img_path, new_img_path)

wnids_path = 'tiny-imagenet-200/wnids.txt'
train_dir = 'tiny-imagenet-200/train/'
val_dir = 'tiny-imagenet-200/val/'
val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

class_mapping = load_class_mapping(wnids_path)
process_train_set(train_dir, class_mapping)
process_val_set(val_dir, val_annotations_path, class_mapping)
