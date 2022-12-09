import os
from shutil import copyfile
from utils import make_dir


def _preprocess_dataset(src, dst):
    make_dir(dst)
    for _, _, files in os.walk(src, topdown=True):
        for name in files:
            if name[-3:] != 'jpg':
                continue
            label = name.split('_')[0]
            if int(label) == -1:
                continue
            img_src = os.path.join(src, name)
            img_dir_dst = os.path.join(dst, label)
            img_dst = os.path.join(img_dir_dst, name)
            make_dir(img_dir_dst)
            copyfile(img_src, img_dst)


def preprocess_dataset(src, dst):
    make_dir(dst)

    _preprocess_dataset(os.path.join(src, 'bounding_box_train'), os.path.join(dst, 'train'))
    _preprocess_dataset(os.path.join(src, 'bounding_box_test'), os.path.join(dst, 'gallery'))
    _preprocess_dataset(os.path.join(src, 'query'), os.path.join(dst, 'query'))


if __name__ == '__main__':
    preprocess_dataset('../Market-1501-v15.09.15', './dataset')
