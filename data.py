import scipy.misc
import random
import numpy as np
import cv2 as cv
import os

train_set = []
test_set = []
batch_index = 0

def load_dataset(data_dir, img_size):
    global train_set
    global test_set
    imgs = []

    files = os.listdir(data_dir)
    print('loading data...')
    for i in range(len(files)):
        try:
            tmp = scipy.misc.imread(data_dir+'/'+files[i])
            print(data_dir+'/'+files[i])
            x,y,z = tmp.shape
            n_x = x//img_size
            n_y = y//img_size
            coords = [ (q,r) for q in range(n_x) for r in range(n_y) ]
            for q, r in coords:
                imgs.append(tmp[q*img_size:(q+1)*img_size, r*img_size:(r+1)*img_size, :])
        except:
            print("oops! file:"+data_dir+'/'+files[i])

    test_size = min(20, int(len(imgs)*0.2))
    random.shuffle(imgs)
    test_set = imgs[:test_size]
    train_set = imgs[test_size:]
    print('imgs count:', len(train_set))
    return

def get_test_set(original_size, shrunk_size):
    imgs = test_set
    x = [scipy.misc.imresize(imgs[i], (shrunk_size,shrunk_size)) for i in range(len(imgs))]
    y = test_set
    return x,y


def get_batch(batch_size, original_size, shrunk_size):
    global batch_index
    max_counter = len(train_set)//batch_size
    counter = batch_index % max_counter
    if counter == 0:
        random.shuffle(train_set)
    x = [scipy.misc.imresize(train_set[counter*batch_size+n],(shrunk_size,shrunk_size)) for n in range(batch_size)]
    batch_index = (batch_index+1)%max_counter
    y = [train_set[counter*batch_size+n] for n in range(batch_size)]
    return x, y

