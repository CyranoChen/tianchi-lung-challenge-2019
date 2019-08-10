import os
import time
import numpy as np
import pandas as pd
import random
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence

import settings
import helper
import visual

def get_classify_file_list(wtype=None):
    dataset = []
    path = wtype + '/' if wtype is not None else '*/'

    # 添加正样本
    dataset.extend(glob(settings.PREPROCESS_POS_DIR + path + '*.png'))

    # 按正样本的数量，添加假阳性和负样本
    pos_count = len(dataset)
    list_fpos = glob(settings.PREPROCESS_FPOS_DIR + path + '*.png')
    random.shuffle(list_fpos) # 随机抽取

    list_neg = glob(settings.PREPROCESS_NEG_DIR + path + '*.png')
    random.shuffle(list_neg) # 随机抽取

    # 添加假阳性
    # dataset.extend(list_fpos[:min(len(list_fpos), pos_count//settings.FPOS_RATE_BY_POS)])
    dataset.extend(list_fpos)

    # 添加负样本
    # dataset.extend(list_neg[:min(len(list_neg), pos_count//settings.NEG_RATE_BY_POS)])

    random.shuffle(dataset)
    print('dataset length:', len(dataset))

    return dataset


def split_dataset(dataset, train_by_valid=0.05, wtype=None, show_labels=True):
    
    trainset, validset = train_test_split(dataset, test_size=train_by_valid, random_state=0) # 划分训练和验证集
    testset = []
#         validset, testset = train_test_split(testset, test_size=valid_by_test, random_state=0) # 划分验证和测试集

    print('train, valid, test:', len(trainset), len(validset), len(testset))
    
    if show_labels:
        labels = get_labels(wtype)
    #         y_test_list = [helper.get_label_from_cube_name(fn, self.labels) for fn in testset]
        y_valid_list = [helper.get_label_from_cube_name(fn, labels) for fn in validset]
        y_train_list = [helper.get_label_from_cube_name(fn, labels) for fn in trainset]

        # 验证标注中的同一分布情况
    #         print('y_test:', [y_test_list.count(l) for l in range(len(self.labels))])
        print('y_valid:', [y_valid_list.count(l) for l in range(len(labels))])
        print('y_train:', [y_train_list.count(l) for l in range(len(labels))])

    return trainset, validset, testset


def get_labels(wtype=None):
    if wtype == 'lung':
        return [0, 1, 5]
    elif wtype == 'medi':
        return [0, 31, 32]
    else:
        return [0, 1, 5, 31, 32]


class ClassifySequence(Sequence):
    def __init__(self, dataset, wtype=None, batch_size=64):
        self.wtype = wtype
        self.labels = get_labels(wtype)
        self.dataset = dataset   
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        X, Y = [], []
        files = self.dataset[idx*self.batch_size:(idx + 1)*self.batch_size]
        for file in files:    
            cube = helper.load_cube_img(file, rows=8, cols=8)

            size_old = settings.CUBE_SIZE
            size_new = 32
            cs = (size_old - size_new) // 2
            ce = (size_old - size_new) // 2 + size_new
            cube = cube[cs:ce, cs:ce, cs:ce]
            assert cube.shape == (size_new, size_new, size_new)

            X.append(cube.astype('float'))
            Y.append(helper.get_label_from_cube_name(file, self.labels))

        X = helper.preprocess_input(np.asarray(X)) # 预处理，归一化，添加channel维度
        Y = keras.utils.to_categorical(np.asarray(Y), len(self.labels)) # one-hot

        return X, Y


def get_segment_file_list(wtype=None):
    dataset = []
    path = wtype + '/' if wtype is not None else '*/'

    # 添加正样本
    dataset.extend(glob(settings.PREPROCESS_POS_DIR + path + '*.png'))

    random.shuffle(dataset)
    print('dataset length:', len(dataset))

    return dataset

            
class SegmentSequence(Sequence):
    def __init__(self, dataset, wtype=None, batch_size=64):
        self.wtype = wtype
        self.labels = get_labels(wtype)
        self.dataset = dataset   
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        X, Y = [], []
        files = self.dataset[idx*self.batch_size:(idx + 1)*self.batch_size]
        for file in files:
            cube_file = file
            seg_file = cube_file.replace(settings.PREPROCESS_POS_DIR, settings.PREPROCESS_SEG_DIR)
#             seg_file = seg_file.replace(self.wtype, self.wtype + '_label')
            
            cube = helper.load_cube_img(cube_file, rows=8, cols=8)
            seg = helper.load_cube_img(seg_file, rows=8, cols=8)
            
            assert cube.shape == seg.shape == (64, 64, 64)
                
            X.append(cube.astype('float'))
            Y.append(seg.astype('float'))

        X = helper.preprocess_input(np.asarray(X)) # 预处理，归一化，添加channel维度
        Y = helper.preprocess_input(np.asarray(Y)) # 预处理，归一化，添加channel维度

        return X, Y


# class ClassifyGenerator(object):
#     def __init__(self, wtype=None):
#         if wtype == 'lung':
#             self.labels = [0, 1, 5]
#         elif wtype == 'medi':
#             self.labels = [0, 31, 32]
#         else:
#             self.labels = [0, 1, 5, 31, 32]
#         self.wtype = wtype
#         self.dataset = self.get_file_list()    
#         self.trainset, self.validset, self.testset = self.split_dataset()
#         time.sleep(1) # get testset in advance
# #         self.X_test, self.Y_test = self.get_dataset(mode='test')

        
#     def get_file_list(self):
#         dataset = []
        
#         path = self.wtype + '/' if self.wtype is not None else '*/'
        
#         # 添加正样本
#         dataset.extend(glob(settings.PREPROCESS_POS_DIR + path + '*.png'))

#         # 按正样本的数量，添加假阳性和负样本
#         pos_count = len(dataset)
#         list_fpos = glob(settings.PREPROCESS_FPOS_DIR + path + '*.png')
#         random.shuffle(list_fpos) # 随机抽取

#         list_neg = glob(settings.PREPROCESS_NEG_DIR + path + '*.png')
#         random.shuffle(list_neg) # 随机抽取
        
#         # 添加假阳性
#         dataset.extend(list_fpos[:min(len(list_fpos), pos_count//settings.FPOS_RATE_BY_POS)])
                            
#         # 添加负样本
#         dataset.extend(list_neg[:min(len(list_neg), pos_count//settings.NEG_RATE_BY_POS)])
        
#         random.shuffle(dataset)
#         print('dataset length:', len(dataset))
              
#         return dataset
              
    
#     def split_dataset(self, train_by_valid=0.05):
#         trainset, validset = train_test_split(self.dataset, test_size=train_by_valid, random_state=0) # 划分训练和验证集
#         testset = []
# #         validset, testset = train_test_split(testset, test_size=valid_by_test, random_state=0) # 划分验证和测试集
              
#         print('train, valid, test:', len(trainset), len(validset), len(testset))
              
# #         y_test_list = [helper.get_label_from_cube_name(fn, self.labels) for fn in testset]
#         y_valid_list = [helper.get_label_from_cube_name(fn, self.labels) for fn in validset]
#         y_train_list = [helper.get_label_from_cube_name(fn, self.labels) for fn in trainset]
              
#         # 验证标注中的同一分布情况
# #         print('y_test:', [y_test_list.count(l) for l in range(len(self.labels))])
#         print('y_valid:', [y_valid_list.count(l) for l in range(len(self.labels))])
#         print('y_train:', [y_train_list.count(l) for l in range(len(self.labels))])
              
#         return trainset, validset, testset
    
    
#     def get_records(self, mode='train'):
#         if mode == 'train':
#             return self.trainset
#         elif mode == 'valid':
#             return self.validset
#         elif mode == 'test':
#             return self.testset

        
#     def get_dataset(self, mode='test'):
#         records = self.get_records(mode)
        
#         X, Y = [], []
#         for file in tqdm(records):
#             cube = helper.load_cube_img(file, rows=8, cols=8)
#             assert cube.shape == (64, 64, 64)
            
#             X.append(cube.astype('float'))
#             Y.append(helper.get_label_from_cube_name(file, self.labels))
            
#         X = self.preprocess_input(np.asarray(X)) # 预处理，归一化，添加channel维度
#         Y = np.asarray(Y)
        
#         print(mode, X.shape, Y.shape, '(no one-hot)')
        
#         return X, Y
            
        
#     def flow_classfication(self, mode='train', batch_size=64):
#         records = self.get_records(mode)

#         while True:
#             X, Y, count = [], [], 0
#             while count < batch_size:
#                 file = random.sample(records, 1)[0] 
#                 cube = helper.load_cube_img(file, rows=8, cols=8)
                
#                 size_old = settings.CUBE_SIZE
#                 size_new = 32
#                 cs = (size_old - size_new) // 2
#                 ce = (size_old - size_new) // 2 + size_new
#                 cube = cube[cs:ce, cs:ce, cs:ce]
#                 assert cube.shape == (size_new, size_new, size_new)
                
#                 if np.sum(cube) > settings.THRESHOLD_VALID_CUBE:
#                     X.append(cube.astype('float'))
#                     Y.append(helper.get_label_from_cube_name(file, self.labels))

#                     count += 1
                
#             X = helper.preprocess_input(np.asarray(X)) # 预处理，归一化，添加channel维度
#             Y = keras.utils.to_categorical(np.asarray(Y), len(self.labels)) # one-hot
                       
#             yield X, Y

            
# class SegmentGenerator2d(object):
#     def __init__(self, wtype=None):
#         if wtype == 'lung':
#             self.labels = [0, 1, 5]
#         elif wtype == 'medi':
#             self.labels = [0, 31, 32]
#         else:
#             self.labels = [0, 1, 5, 31, 32]
#         self.wtype = wtype
#         self.dataset = self.get_file_list()    # DataFrame
#         self.trainset, self.validset, self.testset = self.split_dataset() # DataFrame
#         time.sleep(1) # get testset in advance
# #         self.X_test, self.Y_test = self.get_dataset(mode='test')

        
#     def get_file_list(self):
#         dataset = []
#         df_anno = pd.read_csv(settings.PREPROCESS_ANNOTATION_FILE)
#         df_anno['seriesuid'] = df_anno['seriesuid'].astype(str)
#         df_anno = df_anno.set_index('seriesuid')
        
#         for idx, item in df_anno.iterrows():
#             if item.label not in self.labels:
#                 continue
                
#             for z in range(int(item.vcoordZ - item.diameterZ / item.spacingZ // 2), 
#                            int(item.vcoordZ + float(item.diameterZ / item.spacingZ) // 2) + 1):
#                 record = {}
#                 record['seriesuid'] = str(idx)
#                 record['width'] = item.width
#                 record['height'] = item.height
#                 record['vcoordX'] = item.vcoordX
#                 record['vcoordY'] = item.vcoordY
#                 record['vcoordZ'] = int(z)
#                 record['diameterX'] = item.diameterX
#                 record['diameterY'] = item.diameterY
#                 record['label'] = int(item.label)
                
#                 dataset.append(record)
        
#         return dataset
              
    
#     def split_dataset(self, train_by_test=0.1, valid_by_test=0.5):
#         trainset, testset = train_test_split(self.dataset, test_size=0.1, random_state=0) # 划分训练和测试集
#         validset, testset = train_test_split(testset, test_size=0.5, random_state=0) # 划分验证和测试集
        
#         columns = ['seriesuid', 'width', 'height', 'vcoordX', 'vcoordY', 'vcoordZ', 'diameterX', 'diameterY', 'label']
              
#         df_trainset = pd.DataFrame(trainset, columns=columns)
#         df_trainset = df_trainset.set_index('seriesuid')
#         df_validset = pd.DataFrame(validset, columns=columns)
#         df_validset = df_validset.set_index('seriesuid')
#         df_testset = pd.DataFrame(testset, columns=columns)
#         df_testset = df_testset.set_index('seriesuid')
        
#         print('train, valid, test:', len(df_trainset), len(df_validset), len(df_testset))
              
#         # 验证标注中的同一分布情况
#         print('y_test:', [list(df_trainset['label']).count(l) for l in self.labels])
#         print('y_valid:', [list(df_validset['label']).count(l) for l in self.labels])
#         print('y_train:', [list(df_testset['label']).count(l) for l in self.labels])
              
#         return df_trainset, df_validset, df_testset
    
    
#     def preprocess_input(self, input_tensor, data_format=None):
#         """Preprocesses a tensor encoding a batch of cubes.

#         # Arguments
#             x: input Numpy tensor, 4D. (m, h, w, c)
#             data_format: data format of the cube tensor.

#         # Returns
#             Preprocessed tensor.
#         """
#         if data_format is None:
#             data_format = K.image_data_format()
#         assert data_format in {'channels_last', 'channels_first'}

#         if data_format == 'channels_first':
#             if input_tensor.ndim == 3:
#                 input_tensor = np.expand_dims(input_tensor, axis=0)
#                 input_tensor[0, :, :] /= 255.
#             else:
#                 input_tensor = np.expand_dims(input_tensor, axis=1)
#                 input_tensor[:, 0, :, :] /= 255.           
#         else:
#             input_tensor = np.expand_dims(input_tensor, axis=-1)
#             input_tensor[..., 0] /= 255.

#         return input_tensor
    
    
#     def get_records(self, mode='train'):
#         if mode == 'train':
#             random.shuffle(self.trainset)
#             return self.trainset
#         elif mode == 'valid':
#             random.shuffle(self.validset)
#             return self.validset
#         elif mode == 'test':
#             random.shuffle(self.testset)
#             return self.testset

        
# #     def get_dataset(self, mode='test'):
# #         records = self.get_records(mode)
        
# #         X, Y = [], []
# #         for file in tqdm(records):
# #             cube = helper.load_cube_img(file, rows=8, cols=8)
# #             assert cube.shape == (64, 64, 64)
            
# #             X.append(cube.astype('float'))
# #             Y.append(helper.get_label_from_cube_name(file, self.labels))
            
# #         X = self.preprocess_input(np.asarray(X)) # 预处理，归一化，添加channel维度
# #         Y = np.asarray(Y)
        
# #         print(mode, X.shape, Y.shape, '(no one-hot)')
        
# #         return X, Y
            
        
#     def flow_segmentation(self, mode='train', batch_size=64, plot=False):
#         df_records = self.get_records(mode)

#         while True:
#             X, Y, count = [], [], 0
#             while count < batch_size:
#                 rec = df_records.sample(1).iloc[0]
#                 z = int(rec.vcoordZ)
#                 if self.wtype == 'lung' or rec.label in [1,5]:
#                     img_file_src = settings.PREPROCESS_TRAIN_DIR + rec.name + f'/{str(z).zfill(4)}.png'
#                     mask_file_src = settings.PREPROCESS_TRAIN_DIR + rec.name + f'/{str(z).zfill(4)}_maskl.png'
#                 elif self.wtype == 'medi' or rec.label in [31,32]:
#                     img_file_src = settings.PREPROCESS_TRAIN_DIR + rec.name + f'/{str(z).zfill(4)}_medi.png'
#                     mask_file_src = settings.PREPROCESS_TRAIN_DIR + rec.name + f'/{str(z).zfill(4)}_maskm.png'

#                 img, mask = helper.load_slice_img(img_file_src, mask_file_src, pad_shape=settings.SEGMENT_INPUT_SHAPE)
#                 label = helper.build_slice_label(rec, pad_shape=settings.SEGMENT_INPUT_SHAPE)

#                 assert img.shape == mask.shape == label.shape == settings.SEGMENT_INPUT_SHAPE

#                 if plot:
#                     visual.plot_segment_sample(img, mask, label, title=f'{rec.name}-w{rec.width}-h{rec.height}-x{rec.vcoordX}-y{rec.vcoordY}-z{rec.vcoordZ}-dx{rec.diameterX}-dy{rec.diameterY}-l{rec.label}')
                
#                 if np.sum(img*(mask>0)*(label>0)) == 0:
#                     if plot:
#                         print('nothing in label zone', '\n', rec)
#                     continue

#                 X.append((img*(mask>0)).astype('float')) # ignore the mask
#                 Y.append(label.astype('float'))
#                 count += 1
            
#             # 预处理，归一化，添加channel维度
#             X = self.preprocess_input(np.asarray(X)) 
#             Y = self.preprocess_input(np.asarray(Y)) 
                    
#             yield X, Y

            
# class SegmentGenerator3d(object):
#     def __init__(self, wtype=None):
#         if wtype == 'lung':
#             self.labels = [0, 1, 5]
#         elif wtype == 'medi':
#             self.labels = [0, 31, 32]
#         else:
#             self.labels = [0, 1, 5, 31, 32]
#         self.wtype = wtype
# #         self.dataset = self.get_file_list()    # DataFrame
#         self.trainset, self.validset, self.testset = self.split_dataset() # DataFrame
#         time.sleep(1) # get testset in advance
# #         self.X_test, self.Y_test = self.get_dataset(mode='test')


#     def split_dataset(self):
#         path = self.wtype + '/' if self.wtype is not None else '*/'
        
#         trainset = glob(settings.PREPROCESS_SEG_DIR + path + '*_x_train.npy')
#         validset = glob(settings.PREPROCESS_SEG_DIR + path + '*_x_val.npy')
        
#         validset, testset = train_test_split(validset, test_size=0.5, random_state=0) # 划分验证和测试集
     
#         return trainset, validset, testset

    
#     def get_records(self, mode='train'):
#         if mode == 'train':
#             random.shuffle(self.trainset)
#             return self.trainset
#         elif mode == 'valid':
#             random.shuffle(self.validset)
#             return self.validset
#         elif mode == 'test':
#             random.shuffle(self.testset)
#             return self.testset
        
    
#     def flow_segmentation(self, mode='train', batch_size=16):
#         records = self.get_records(mode)

#         while True:
#             i = random.randint(0, len(records)-1)
#             wtype = records[i].split('/')[-2]
#             file_name = records[i].split('/')[-1]
            
#             X = np.load(records[i]).astype('float')
#             Y = np.load(settings.PREPROCESS_SEG_DIR + wtype + '/' + file_name.replace('x', 'y')).astype('float')
            
#             assert X.shape == Y.shape == (16, 64, 64, 64, 1)
                       
#             yield X, Y


        