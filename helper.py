import os
from glob import glob, iglob
import matplotlib.pyplot as plt

import numpy as np
import cv2
import scipy.ndimage
from skimage import morphology, measure, segmentation, filters

import settings


# 获取CT的窗宽width和窗位center
def get_window_size(window_type):
    if window_type == 'lung': # 肺窗 WW 1500 ~ 2000HU, WL -450 ~ —600HU
        ww, wl = settings.WINDOW_LUNG
    elif window_type == 'mediastinal': # 纵膈窗 WW 250—350HU、WL 30—50HU
        ww, wl = settings.WINDOW_MEDI
    return ww, wl


# 按窗宽和窗位实现归一化
def normalize(lung_hu, ww=2000, wl=-500):
    ''' normalize pixel value to [0, 255] based on ww and wl '''
#     threshold_upper = (wl + ww // 2)
#     threshold_low = (wl - ww // 2)
       
    win_min = wl - ww // 2. + 0.5
    win_max = wl + ww // 2. + 0.5
    factor = 255. / ww
    
    lung_norm = (lung_hu.copy() - win_min) * factor  
    lung_norm = np.clip(lung_norm, 0, 255)  
       
    return np.uint8(lung_norm)


# 提取每张CT片的肺部区域，返回为布尔矩阵和所占面积
def get_segmented_lung(img, mode='lung', plot=False):
    origin_img = img.copy()
    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    
    # Step 1: Convert into a binary image.
    if mode == 'lung':
        binary = img < settings.BINARY_THRESHOLD_LUNG
    elif mode == 'medi':
        binary = img > settings.BINARY_THRESHOLD_BONE

    if plot:
        axs[0,0].set(title='binary')
        axs[0,0].imshow(binary, cmap='gray')
        
    if mode == 'lung':
        # Step 2: Remove the blobs connected to the border of the image.
        cleared = segmentation.clear_border(binary)
        
        if plot:
            axs[0,1].set(title='cleared')
            axs[0,1].imshow(cleared, cmap='gray')
        
        # Step 3: Label the image.
        label_image = measure.label(cleared)

        # Step 4: Keep the labels with 2 largest areas.
        areas = [r.area for r in measure.regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(label_image):
                if region.area < areas[-2]:
                    for c in region.coords:
                           label_image[c[0], c[1]] = 0

        mask = label_image > 0
        
        if plot:
            axs[0,2].set(title='2 largest areas')
            axs[0,2].imshow(mask, cmap='gray')
        
        # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
        mask = morphology.binary_erosion(mask, morphology.disk(2))
        if plot:
            axs[0,3].set(title='erosion')
            axs[0,3].imshow(mask, cmap='gray')

        # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
        mask = morphology.binary_closing(mask, morphology.disk(10))
        if plot:
            axs[1,0].set(title='closing')
            axs[1,0].imshow(mask, cmap='gray')

        # Step 7: Fill in the small holes inside the binary mask of lungs.
        edges = filters.roberts(mask)
        mask = scipy.ndimage.binary_fill_holes(edges)
        # dilation
        mask = morphology.dilation(mask, selem=morphology.disk(5))
        if plot:
            axs[1,1].set(title='dilation')
            axs[1,1].imshow(mask, cmap='gray')
            
            
        # Step 8: Remove bone
#         mask *= img < settings.BINARY_THRESHOLD_BONE
        
    elif mode == 'medi':
        # Step 2: Erosion operation with a disk of radius 1.
        mask = morphology.binary_erosion(binary, morphology.disk(1))
        if plot:
            axs[0,1].set(title='erosion')
            axs[0,1].imshow(mask, cmap='gray')

        # Step 3 dilation
        mask = morphology.dilation(mask, selem=morphology.disk(6))
#         mask = morphology.dilation(mask, selem=morphology.disk(4))
        if plot:
            axs[0,2].set(title='dilation')
            axs[0,2].imshow(mask, cmap='gray')
                  
        # Step 4: Label the image.
        label_image = measure.label(mask)

        # Step 5: remove bone area
        regions = measure.regionprops(label_image)
        for r in regions:
            if 5 < r.area > 10000:
                for c in r.coords:
                    label_image[c[0], c[1]] = 0
                    
        mask = label_image > 0 
        if plot:
            axs[0,3].set(title='remove bone area')
            axs[0,3].imshow(mask, cmap='gray')
        
       
        # Step 6: Closure operation with a disk of radius 8.
        mask = morphology.binary_closing(mask, morphology.disk(8))
        if plot:
            axs[1,0].set(title='closing')
            axs[1,0].imshow(mask, cmap='gray')
            
            
#         # Step 7: Fill in the small holes inside the binary mask of lungs.
#         edges = filters.roberts(mask)
#         mask = scipy.ndimage.binary_fill_holes(edges)
#         if plot:
#             axs[1,1].set(title='fill_holes')
#             axs[1,1].imshow(mask, cmap='gray')
            
        mask = mask < 1
        
    
    # Step 8: Superimpose the binary mask on the input image.
    if plot:
        axs[1,2].set(title='orgin')
        axs[1,2].imshow(origin_img, cmap='gray')
        axs[1,3].set(title='final')
        axs[1,3].imshow(mask*origin_img, cmap='gray')
        plt.show()
    
    return mask, np.sum(mask != 0)


# 读取肺CT的三维重建为ndarray
def load_lung_array(seriesuid, width, height, num_z, mode='train', wtype='lung'):
    if mode == 'test':
        preprocess_dir = settings.PREPROCESS_TEST_DIR
    else:
        preprocess_dir = settings.PREPROCESS_TRAIN_DIR
        
    if wtype == 'lung': 
        list_imgs = glob(preprocess_dir + seriesuid + '/????.png')
        list_masks = glob(preprocess_dir + seriesuid + '/????_maskl.png')
    elif wtype == 'medi':
        list_imgs = glob(preprocess_dir + seriesuid + '/????_medi.png')
        list_masks = glob(preprocess_dir + seriesuid + '/????_maskm.png')

    list_imgs.sort()
    list_masks.sort()
    
    assert num_z == len(list_imgs) == len(list_masks)
    
    lung_array = np.zeros(shape=(num_z, height, width)) #z,y,x
    mask_array = np.zeros(shape=(num_z, height, width))
    
    for i in range(num_z):
        lung_array[i] = cv2.imread(list_imgs[i], cv2.IMREAD_GRAYSCALE)
        mask_array[i] = cv2.imread(list_masks[i], cv2.IMREAD_GRAYSCALE)
        
    assert lung_array.shape == mask_array.shape == (num_z, height, width)
    
    return lung_array, mask_array


# 通过肺CT的三维重建提取指定坐标为中心的立方体
def get_cube_from_lung_array(lung_array, cx, cy, cz, block_size=64):
    num_z, height, width = lung_array.shape # z,y,x
    d = int(block_size)
    
    sz = max(cz - d / 2, 0)
    if sz + d > num_z:
        sz = max(num_z - d, 0)
        
    sy = max(cy - d / 2, 0)
    if sy + d > height:
        sy = max(height - d, 0)
    
    sx = max(cx - d / 2, 0)
    if sx + d > width:
        sx = max(width - d, 0)
        
    sz = int(sz)
    sy = int(sy)
    sx = int(sx)

    cube = lung_array[sz:(sz+d), sy:(sy+d), sx:(sx+d)]
    assert cube.shape[0] == cube.shape[1] == cube.shape[2] == d
    return cube


# 根据标注坐标，制作分割的掩码
def create_seg_label(diameter, offset=np.array([0,0,0]), block_size=64):
    seg = np.zeros((block_size, block_size, block_size), dtype=np.uint8)
    dz, dy, dx = diameter # z,y,x
    oz, oy, ox = offset
    
    sz = int(max(block_size // 2 - oz - dz / 2, 0))  
    ez = int(min(block_size // 2 - oz + dz / 2, block_size))
                     
    sy = int(max(block_size // 2 - oy - dy / 2, 0))   
    ey = int(min(block_size // 2 - oy + dy / 2, block_size))
                     
    sx = int(max(block_size // 2 - ox - dx / 2, 0))   
    ex = int(min(block_size // 2 - ox + dx / 2, block_size))

    seg[sz:ez, sy:ey, sx:ex] = 1*255
    return seg


# 保存立方体为一张序列图片
def save_cube_img(target_path, cube, rows, cols):
    num_z, h, w = cube.shape # z,y,x
    assert rows * cols == num_z
    
    img = np.zeros((rows * h, cols * w), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            y = row * h
            x = col * w
            img[y:(y+h), x:(x+w)] = cube[row * cols + col,:,:]
    
    if target_path is not None:
        cv2.imwrite(target_path, img)
        
    return img

    
# 提取序列图片为一个立方体
def load_cube_img(src_path, rows, cols, block_size=64):
    d = block_size
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)   
    assert img.shape == (rows*d, cols*d)

    cube = np.zeros((rows * cols, d, d),  dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            y = row * d
            x = col * d
            cube[row * cols + col,:,:] = img[y:(y+d), x:(x+d)]

    return cube


# 提取单张图片为一个np.array, 指定shape
def load_slice_img(img_src_path, mask_src_path, pad_shape=(512, 512)):
    img = cv2.imread(img_src_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_src_path, cv2.IMREAD_GRAYSCALE)
                   
    assert img.shape == mask.shape
                   
    pad_before = (pad_shape[0] - img.shape[0]) // 2
    pad_after = (pad_shape[1] - img.shape[1]) // 2
                   
    if img.shape[0] % 2 == img.shape[1] % 2 == 1:
        pad_after += 1
                                 
    return np.pad(img, pad_width=(pad_before, pad_after), mode='constant'), np.pad(mask, pad_width=(pad_before, pad_after), mode='constant')
                   

# 制作分割网络样本的标注
def build_slice_label(record, pad_shape=(512, 512)):
    label = np.zeros((int(record.height), int(record.width)), dtype=np.uint8)
    label[int(record.vcoordY - np.ceil(record.diameterY/2)):int(record.vcoordY + np.ceil(record.diameterY/2)), 
          int(record.vcoordX - np.ceil(record.diameterX/2)):int(record.vcoordX + np.ceil(record.diameterX/2))] = 1*255
    
    pad_before = (pad_shape[0] - label.shape[0]) // 2
    pad_after = (pad_shape[1] - label.shape[1]) // 2
                   
    if label.shape[0] % 2 == label.shape[1] % 2 == 1:
        pad_after += 1
    
    return np.pad(label, pad_width=(pad_before, pad_after), mode='constant')


# 获取指定立方体文件名中的标注ID
def get_label_from_cube_name(file_name, labels=[0, 1, 5, 31, 32]):
    try:
        label = int(file_name.split('/')[-1].strip('.png').split('_')[-1].strip('l'))
        return labels.index(label)
    except:
        return 0
    
    
# 获取对应预测结果的分类ID
def get_origin_class_by_predicted_value(value, labels=[0, 1, 5, 31, 32]):
    return labels[value]
        
# 获取预测敏感区域，提取疑似坐标
def get_regions_detail(pred_cube, item_cube):
#     class_boundary = np.array([settings.CUBE_SIZE, settings.CUBE_SIZE, settings.CUBE_SIZE])
#     pred_cube = morphology.dilation(pred_cube, np.ones([3, 3, 3]))
    pred_cube = morphology.binary_closing(pred_cube, np.ones([3, 3, 3]))
    labels = measure.label(pred_cube)
    regions = measure.regionprops(labels)
    
    bboxes= []
    vcoords = []
    diameters = []
#     crops= []
#     crop_centers = []
    for r in regions:
        box = r.bbox
        if box[3] - box[0] > 2 and box[4] - box[1] > 2 and box[5] - box[2] > 2: # ignore too small focus
            cz = (box[3] + box[0]) // 2
            cy = (box[4] + box[1]) // 2
            cx = (box[5] + box[2]) // 2
                        
            c = np.array([cz + item_cube.vcoordZ - settings.CUBE_SIZE // 2, 
                          cy + item_cube.vcoordY - settings.CUBE_SIZE // 2, 
                          cx + item_cube.vcoordX - settings.CUBE_SIZE // 2])
            d = np.array([int(box[3] - box[0]), int(box[4] - box[1]), int(box[5] - box[2])])
            
            vcoords.append(c)
            diameters.append(d)
            bboxes.append(box)
            
#     for idx, bbox in enumerate(bboxes):
#         crop = np.zeros(class_boundary, dtype=np.float32)
#         crop_center = centers[idx]
#         crop_center = crop_center + orign
#         half = class_boundary / 2
#         crop_center = check_center(class_boundary, crop_center, img_arr.shape)
#         crop = img_arr[int(crop_center[0] - half[0]):int(crop_center[0] + half[0]), \
#                int(crop_center[1] - half[1]):int(crop_center[1] + half[1]), \
#                int(crop_center[2] - half[2]):int(crop_center[2] + half[2])]
#         crops.append(crop)
#         crop_centers.append(crop_center)
    return vcoords, diameters, bboxes

# 非极大值抑制
def nms(dets, thresh):
    cx = dets[:, 0]
    cy = dets[:, 1]
    cz = dets[:, 2]
    dx = dets[:, 3]
    dy = dets[:, 4]
    dz = dets[:, 5]
    scores = dets[:, -1]

    areas = dx * dy * dz
    order = scores.argsort()[::-1]
       
    x1,x2 = cx - dx // 2, cx + dx // 2
    y1,y2 = cy - dy // 2, cy + dy // 2
    z1,z2 = cz - dz // 2, cz + dz // 2

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])       
        
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.maximum(z2[i], z2[order[1:]])   

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        d = np.maximum(0.0, zz2 - zz1 + 1)
        
        inter = w * h * d
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# 处理网络输入张量，增加channel维度，归一化
def preprocess_input(inputs, data_format='channels_last'):
    """Preprocesses a tensor encoding a batch of cubes.

    # Arguments
        x: input Numpy tensor, 5D. (m, d, h, w, c)
        data_format: data format of the cube tensor.

    # Returns
        Preprocessed tensor.
    """

    if data_format == 'channels_first':
        if inputs.ndim == 4:
            inputs = np.expand_dims(inputs, axis=0)
            inputs[0, :, :, :] /= 255.
        else:
            inputs = np.expand_dims(inputs, axis=1)
            inputs[:, 0, :, :, :] /= 255.           
    else:
        inputs = np.expand_dims(inputs, axis=-1)
        inputs[..., 0] /= 255.

    return inputs