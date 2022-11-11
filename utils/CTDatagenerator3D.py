import tensorflow.keras
import numpy as np
import skimage
import SimpleITK as sitk
import os
import json
import glob

class DataGenerator(tensorflow.keras.utils.Sequence):
    """Generates data for Keras"""
    """This structure guarantees that the network will only train once on each sample per epoch"""

    def __init__(self, im_path, label_path, batch_size=1, dim=(128, 128, 64),
                 n_classes=1, shuffle=True, augment=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = self.createIDs(im_path, label_path)
        self.im_path = im_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
    def createIDs(self, im_path, label_path):
        partition = {}
        images = glob.glob(os.path.join(im_path, "*.nii.gz")) + glob.glob(os.path.join(im_path, "*.nrrd"))
        images_IDs = [name.split("/")[-1] for name in images]
        partition['list'] = images_IDs
        return partition['list']
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #print(list_IDs_temp)
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        if self.augment:
            print(list_IDs_temp)
            pass

        if not self.augment:
            
            X = np.empty([self.batch_size, *self.dim])
            Y = np.empty([self.batch_size, *self.dim, self.n_classes])

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                img_X = sitk.ReadImage(os.path.join(self.im_path, ID))
                img_arr = sitk.GetArrayFromImage(img_X)
                X[i,] = img_arr#np.moveaxis(img_arr,[0],[2])
                
                #maskID = str(ID).replace('.nrrd','') + '.seg.nrrd'
                img_Y = sitk.ReadImage(os.path.join(self.label_path, ID))
                msk_arr = sitk.GetArrayFromImage(img_Y)
                Y[i,] = np.expand_dims(msk_arr, axis=-1)
                #Y[i,] = keras.utils.to_categorical(msk_arr, num_classes=self.n_classes)

            X = X.reshape(self.batch_size, *self.dim, 1)
            #print(X.shape)
            #print(Y.shape)
            return X, Y
        
def create_folder(name):
    newpath = os.path.join(os.getcwd(), name) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def push_notification(message):
    payload = '{"text":"%s"}' %message
    response = requests.post('https://hooks.slack.com/services/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', data=payload)
    
