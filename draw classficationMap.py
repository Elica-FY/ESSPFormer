import scipy.io as sio
import Functions
import numpy as np

# 画预测图并保存
path1 = '../classification_maps/ViTFormer_FC+ROPE_IN_iter0_matrix.mat'
mat_data = sio.loadmat(path1)
prediction_matrix = mat_data['P']
#gt_matrix = mat_data['label']
path2 = '../datasets/Indian_pines_gt.mat'
mat_data = sio.loadmat(path2)
gt_hsi = mat_data['indian_pines_gt']
height, width =gt_hsi.shape[1],gt_hsi.shape[0]
Functions.classification_map(prediction_matrix, height, width, 300, '../classification_maps/3DCNN_IN.png')



# 画标签图并保存

# path = '../datasets/Indian_pines_gt.mat'
# mat_data = sio.loadmat(path)
# gt_hsi = mat_data['indian_pines_gt']
#
# #gt_re = np.reshape(gt_hsi, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
# height, width =gt_hsi.shape[1],gt_hsi.shape[0]
# Functions.classification_map( gt_hsi,height, width, 300, '../classification_maps/test5.png')