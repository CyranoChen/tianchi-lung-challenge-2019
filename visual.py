import time
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

import settings


def plot_meta(df_meta, title=''):
    print(title, df_meta.shape)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    axs[0,0].boxplot(df_meta['width'])
    axs[0,0].set(title='width')
    axs[0,1].boxplot(df_meta['height'])
    axs[0,1].set(title='height')
    axs[0,2].boxplot(df_meta['slice'])
    axs[0,2].set(title='slice')

    axs[1,0].boxplot(df_meta['spacingX'])
    axs[1,0].set(title='spacingX/Y')
    axs[1,1].boxplot(df_meta['spacingZ'])
    axs[1,1].set(title='spacingZ')
#     axs[1,2].boxplot(df_meta['segmented'])
#     axs[1,2].set(title='segmented')

    plt.show()
    
    
def plot_annotation(df_annotation, title='annotation'):
    print(title, df_annotation.shape)
    
    fig, axs = plt.subplots(5, 3, figsize=(15, 25))

    axs[0,0].boxplot(df_annotation['vcoordX'])
    axs[0,0].set(title='vcoordX')
    axs[0,1].boxplot(df_annotation['vcoordY'])
    axs[0,1].set(title='vcoordY')
    axs[0,2].boxplot(df_annotation['vcoordZ'])
    axs[0,2].set(title='vcoordZ')

    axs[1,0].boxplot(df_annotation['diameterX'])
    axs[1,0].set(title='diameterX')
    axs[1,1].boxplot(df_annotation['diameterY'])
    axs[1,1].set(title='diameterY')
    axs[1,2].boxplot(df_annotation['diameterZ'])
    axs[1,2].set(title='diameterZ')
    
    df_annotation['startX'] = df_annotation['vcoordX'] - df_annotation['diameterX'] / 2
    df_annotation['endX'] = df_annotation['vcoordX'] + df_annotation['diameterX'] / 2
    
    axs[2,0].boxplot(df_annotation['startX'])
    axs[2,0].set(title='startX')
    axs[3,0].boxplot(df_annotation['endX'])
    axs[3,0].set(title='endX')
    
    df_annotation['startY'] = df_annotation['vcoordY'] - df_annotation['diameterY'] / 2
    df_annotation['endY'] = df_annotation['vcoordY'] + df_annotation['diameterY'] / 2
    
    axs[2,1].boxplot(df_annotation['startY'])
    axs[2,1].set(title='startY')
    axs[3,1].boxplot(df_annotation['endY'])
    axs[3,1].set(title='endY')
    
    df_annotation['startZ'] = df_annotation['vcoordZ'] - df_annotation['diameterZ'] / 2
    df_annotation['endZ'] = df_annotation['vcoordZ'] + df_annotation['diameterZ'] / 2
     
    axs[2,2].boxplot(df_annotation['startZ'])
    axs[2,2].set(title='startZ')
    axs[3,2].boxplot(df_annotation['endZ'])
    axs[3,2].set(title='endZ')
    
    s_width = []
    s_height = []
    s_slice = []
    for idx, item in df_annotation.iterrows():
        s_width.append(item.width - 2 * min(item.startX, item.width - item.endX))
        s_height.append(item.height - 2 * min(item.startY, item.height - item.endY))
        s_slice.append(item.slice - 2 * min(item.startZ, item.slice - item.endZ))
    
    box_w = axs[4,0].boxplot(s_width)
    axs[4,0].set(title='safe width')
    box_h = axs[4,1].boxplot(s_height)
    axs[4,1].set(title='safe height')
    axs[4,2].boxplot(s_slice)
    axs[4,2].set(title='safe slice')
    
    print('box_w:', box_w['whiskers'][1].get_ydata())
    print('box_h:', box_h['whiskers'][1].get_ydata())

    plt.show()

    
def plot_slices(lung_array, title='', box=None): # z,y,x
    print(title, lung_array.shape)
    
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    num_z = lung_array.shape[0]
    c, c_step = 0, num_z // settings.PLOT_SLICES_NUM

    c = num_z // 5
    c_step = c_step // 2
    
    for ax in axs.flat:
        ax.imshow(lung_array[c,:,:], cmap='gray')
        ax.set(title='slice ' + str(c))

        if box:
            ax.add_patch(patches.Rectangle((box['x'], box['y']),box['w'] * 4,box['h'] * 4, linewidth=1, edgecolor='r', facecolor='none'))
        c += c_step

    plt.show()

    
def anim_slices(lung_array, title='', interval=100): # z,y,x
    print(title, lung_array.shape)
    
    start = time.time()
    fig = plt.figure(figsize=(12, 12))

    ims = []
    for i in range(lung_array.shape[0]):
        ims.append([plt.imshow(lung_array[i], cmap="gray")])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    print('time cost:', time.time() - start)
    
    return ani
    
    
def plot_3d_lung(lung_array, title=''):
    print(title, lung_array.shape)
    
    start = time.time()
    imgs = lung_array.transpose(2,1,0) # z,y,x -> x,y,z
    verts, faces = measure.marching_cubes_classic(imgs, 130)
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    mesh.set_facecolor([1, 1, 1])

    ax.add_collection3d(mesh)
    ax.set_xlim(0, lung_array.shape[0])
    ax.set_ylim(0, lung_array.shape[1])
    ax.set_zlim(0, lung_array.shape[2])

    #ax.view_init(10,270) #旋转角度
    plt.rcParams['axes.facecolor'] = 'black' #背景颜色
    plt.axis('off')

    plt.show()
    print('time cost:', time.time() - start)
    
    
def plot_labels(lung_array, mask_array, labels, title=''):
    print(title, lung_array.shape)
    
    num_labels = len(labels)
    fig, axs = plt.subplots(num_labels, 2, figsize=(24, 12*num_labels), sharex=True, sharey=True)
    
    i = 0
    for idx, item in labels.iterrows():  
        num_z = int(item.vcoordZ)
        
#         axs[i,0].imshow(lung_array[num_z,:,:], cmap='gray')
#         axs[i,0].set(title='origin ' + str(num_z))
        
#         axs[i,1].imshow(lung_array[num_z,:,:]*mask_array[num_z,:,:], cmap='gray')
#         axs[i,1].set(title='masked ' + str(num_z))
        
        box = { 'x': item.vcoordX - item.diameterX/2 , 'y': item.vcoordY - item.diameterY/2, 'w': item.diameterX, 'h': item.diameterY }
        axs[i,0].imshow(lung_array[num_z,:,:], cmap='gray')
        axs[i,0].add_patch(patches.Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=2, edgecolor='r', facecolor='none'))
        axs[i,0].set(title=f'label z:{str(item.vcoordZ)} dz:{str(item.diameterZ//item.spacingZ)} l:{str(item.label)}')  
        
        axs[i,1].imshow(lung_array[num_z,:,:]*mask_array[num_z,:,:], cmap='gray')
        axs[i,1].add_patch(patches.Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=1, edgecolor='r', facecolor='none'))
        axs[i,1].set(title=f'masked z:{str(item.vcoordZ)} dz:{str(item.diameterZ//item.spacingZ)} l:{str(item.label)}')     
        
        i += 1

    plt.show()

    
def plot_segment_sample(img, mask, label, title=''):
    print(title)
    
    fig, axs = plt.subplots(2, 2, figsize=(20,20), sharex=True, sharey=True)
    
    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set(title='image')
    axs[0,1].imshow(img*(mask>0), cmap='gray')
    axs[0,1].set(title='image*mask')
    axs[1,0].imshow(label, cmap='gray')
    axs[1,0].set(title='label')
    axs[1,1].imshow(img*(mask>0)*(label>0), cmap='gray')
    axs[1,1].set(title='image*mask*label')
    
    plt.show()