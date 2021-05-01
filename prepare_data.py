import os
import shutil
import random

root_dir = 'database'
classes = ['Normal', 'COVID']

random.seed(0)

def make_dirs(root_dir: str, sub_dir: str, create_new=False):
    path = os.path.join(root_dir, sub_dir)

    if create_new or not (os.path.exists(path) and os.path.isdir(path)):
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
        for c in classes:
            os.makedirs(os.path.join(path, c))



make_dirs(root_dir, 'training', create_new=True)
make_dirs(root_dir, 'testing', create_new=True)


for c in classes:
    imgs = [_ for _ in os.listdir(os.path.join(root_dir, c)) if _.lower().endswith('png')]
    imgs_selected = set(random.sample(imgs, 200))
    for img in imgs:
        if img in imgs_selected:
            shutil.copy(os.path.join(root_dir, c, img), os.path.join(root_dir, 'testing', c, img))
        else:
            shutil.copy(os.path.join(root_dir, c, img), os.path.join(root_dir, 'training', c, img))

print('finished')