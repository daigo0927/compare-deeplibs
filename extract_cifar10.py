import os
import pickle
import numpy as np
from PIL import Image

def extract(d, pkl, mode):
    subd = '/'.join([d, mode])
    if not os.path.exists(subd):
        os.mkdir(subd)
        
    # Extract pickle files and convert them into png format
    print('--------- Extracting {} -----------'.format(pkl))
    pklpath = '/'.join([d, pkl])
    with open(pklpath, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')

    images = data['data']
    images = np.reshape(images, (len(images), 3, 32, 32))
    images = np.transpose(images, axes=(0,2,3,1))
    labels = data['labels']
    filenames = data['filenames']
    
    for image, label, filename in zip(images, labels, filenames):
        
        subsubd = '/'.join([subd, str(label)])
        if not os.path.exists(subsubd):
            os.mkdir(subsubd)
            
        with Image.fromarray(image) as img:
            img.save('{}/{}'.format(subsubd, filename))

                
if __name__ == '__main__':
    d = 'cifar-10-batches-py'
    
    train_pkls = ['data_batch_1',
                  'data_batch_2',
                  'data_batch_3',
                  'data_batch_4',
                  'data_batch_5']
    for pkl in train_pkls:
        extract(d, pkl, mode='train')
        
    test_pkl = 'test_batch'
    extract(d, test_pkl, mode='test')

    print('--------- Completed ------------')
        
