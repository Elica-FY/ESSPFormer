import scipy.io as sio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from itertools import combinations
import os
# -------------------------------------------------------------------------------
def load_dataset(Dataset, VALIDATION_SPLIT):
    if Dataset == 'IN':
        mat_data = sio.loadmat('datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249  # 0.95  # 0.5

    elif Dataset == 'PU':
        uPavia = sio.loadmat('datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776  # 0.99  # 0.01

    elif Dataset == 'PC':
        uPavia = sio.loadmat('/datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('/datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152

    elif Dataset == 'SV':
        SV = sio.loadmat('datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129  # 0.995  # 0.005

    elif Dataset == 'KSC':
        KSC = sio.loadmat('../datasets/KSC.mat')
        gt_KSC = sio.loadmat('../datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211

    elif Dataset == 'BS':
        BS = sio.loadmat('../datasets/Botswana.mat')
        gt_BS = sio.loadmat('../datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248

    elif Dataset == 'LK':
        LK = sio.loadmat('../datasets/WHU_Hi_LongKou.mat')
        gt_LK = sio.loadmat('../datasets/WHU_Hi_LongKou_gt.mat')
        data_hsi = LK['WHU_Hi_LongKou']
        gt_hsi = gt_LK['WHU_Hi_LongKou_gt']
        TOTAL_SIZE = 2200000 #[500,400,270]

    elif Dataset == 'HC':
        HC = sio.loadmat('datasets/WHU_Hi_HanChuan.mat')
        gt_HC = sio.loadmat('datasets/WHU_Hi_HanChuan_gt.mat')
        data_hsi = HC['WHU_Hi_HanChuan']
        gt_hsi = gt_HC['WHU_Hi_HanChuan_gt']
        TOTAL_SIZE = 2200000 #[500,400,270]
    else:
        raise ValueError("Unknown dataset")
    num_classes = np.max(gt_hsi)
    print("num_classes={0}".format(num_classes))

    input_normalize = np.zeros(data_hsi.shape)
    for i in range(data_hsi.shape[2]):
        input_max = np.max(data_hsi[:, :, i])
        input_min = np.min(data_hsi[:, :, i])
        input_normalize[:, :, i] = (data_hsi[:, :, i] - input_min) / (input_max - input_min)
    # data size
    height, width, band = input_normalize.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    number_gt = []
    pos_gt = {}
    number_test = []
    pos_test = {}
    number_train = []
    pos_train = {}

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(gt_hsi==(i+1))
        each_class_num = each_class.shape[0]
        number_gt.append(each_class_num)
        np.random.shuffle(each_class)
        pos_gt[i] = each_class

        number_train_i = np.ceil(each_class_num*(1-VALIDATION_SPLIT)).astype(int)
        number_test_i = each_class_num - number_train_i

        number_test.append(number_test_i)
        number_train.append(number_train_i)
        pos_train[i] = each_class[0:number_train_i,:]
        pos_test[i] = each_class[number_train_i:,:]
        pos_gt[i] = each_class

    total_pos_gt = pos_gt[0]
    total_pos_train = pos_train[0]
    total_pos_test = pos_test[0]
    # print(f'number_train :{number_train}')
    for i in range(1, num_classes):
        total_pos_gt = np.r_[total_pos_gt, pos_gt[i]] #(695,2)
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_gt = total_pos_gt.astype(int)
    total_pos_train = total_pos_train.astype(int)
    total_pos_test = total_pos_test.astype(int)

    return input_normalize,gt_hsi,num_classes, total_pos_train, total_pos_test, total_pos_gt, number_train, number_test, number_gt
#-------------------------------------------------------------------------------
def mirror_hsi(ds,height,width,band,input_normalize,patch):
    padding = 4 + patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    zero_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)

    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    zero_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize

    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    return mirror_hsi, zero_hsi

#-------------------------------------------------------------------------------
def SCHPFP(dataset,patch,hpfpatch,n_perGroup,pixelNum,band,oa_bi,total_pos_test,total_pos_train,total_pos_true,zero_image,mirror_image):
    # Pixel Selection in SCHPFP
    file_path = f'./saved_corr_pos/{dataset}true_patch{patch}_ns{n_perGroup}.npy'
    if os.path.exists(file_path):
        print("read")
        total_corr_pos_true = np.load(file_path)[:, -n_perGroup - 1:, :]
        total_corr_pos_test = read_pixelSelection(total_corr_pos_true, total_pos_test, patch, n_perGroup + 1)
        total_corr_pos_train = pixelSelection(zero_image, total_pos_train, patch, pixelNum)
        print(f"total_corr_pos_train  shape: {total_corr_pos_train.shape}")
    else:
        print("calculate")
        total_corr_pos_true = pixelSelection(zero_image, total_pos_true, patch, n_perGroup + 1)
        total_corr_pos_test = pixelSelection(zero_image, total_pos_test, patch, n_perGroup + 1)
        total_corr_pos_train = pixelSelection(zero_image, total_pos_train, patch, pixelNum)
        print(f"total_corr_pos_train  shape: {total_corr_pos_train.shape}")
        np.save(file_path, total_corr_pos_true)
    # Patch Extraction and Feature Fusion in SCHPFP
    n_train, n_corr, _ = total_corr_pos_train.shape
    x_train = np.zeros((n_train, n_corr, band), dtype=float)
    x_test = np.zeros((total_corr_pos_test.shape[0], n_perGroup+1, band), dtype=float)
    x_true = np.zeros((total_corr_pos_true.shape[0], n_perGroup+1, band), dtype=float)
    for i in range(n_train):
        for j in range(n_corr):
            # Patch Extraction and Feature Fusion
            vc = gain_neighborhood_pixel(mirror_image, total_corr_pos_train[i], j, hpfpatch, oa_bi)
            x_train[i, j, :] = vc
    for i in range(total_corr_pos_test.shape[0]):
        for j in range(n_perGroup+1):
            x_test[i, j, :] = gain_neighborhood_pixel(mirror_image, total_corr_pos_test[i], j, hpfpatch, oa_bi)
    x_test[:, :-1, :] = x_test[:, 1:, :]
    for i in range(total_corr_pos_true.shape[0]):
        for j in range(n_perGroup+1):
            x_true[i, j, :] = gain_neighborhood_pixel(mirror_image, total_corr_pos_true[i], j, hpfpatch, oa_bi)
    x_true[:, :-1, :] = x_true[:, 1:, :]
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_true.dtype))
    print("**************************************************")
    return x_train, x_test, x_true
#-------------------------------------------------------------------------------
def SSGPCF(x_train,x_test,x_ture, pN, npG):
    # Vector Grouping in SSGPCF
    x_train_band, nn_ex = combs_exp(x_train, pN, npG)
    # Pooling Aggregation and Cross Fusion in SSGPCF
    x_train_band_f = cross_fuse(x_train_band)
    x_test_band_f = cross_fuse(x_test)
    x_ture_band_f = cross_fuse(x_ture)
    return x_train_band_f,x_test_band_f,x_ture_band_f, nn_ex
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=7, oa_bi=0):
    # Patch Extraction and Feature Fusion
    x = point[i,0]
    y = point[i,1]
    temp_image = np.mean(mirror_image[x+oa_bi:(x+patch+oa_bi), y+oa_bi:(y+patch+oa_bi), :], axis=(0, 1), keepdims=False)
    return temp_image
#-------------------------------------------------------------------------------
def get_combs(a, b):
    elements = list(range(a))
    comb_array = list(combinations(elements, b))
    result = np.append(comb_array, np.full((np.shape(comb_array)[0], 1), a - 1), axis=1)
    return np.array(result), np.shape(result)[0]
def combs_exp(x, nc, ns):
    # Vector Grouping
    combs, n_ex = get_combs(nc, ns)
    x0, x1, x4 = x.shape
    X = np.empty((x0 * n_ex, ns+1, x4))
    for i in range(x0):
        X[i * n_ex:(i + 1) * n_ex] = x[i, combs]
    return X, n_ex
#-------------------------------------------------------------------------------
def cross_fuse(cube):
    # Pooling Aggregation and Cross Fusion
    batch_size, n_sp, spectral_dim = cube.shape
    selected_spectra = cube
    cross_fused_spectra = np.zeros((batch_size, 4 * spectral_dim))

    cross_fused_spectra[:, 0::4] = np.min(selected_spectra[:, :-1, :], axis=1)
    cross_fused_spectra[:, 1::4] = np.mean(selected_spectra[:, :-1, :], axis=1)
    cross_fused_spectra[:, 2::4] = selected_spectra[:, -1, :]
    cross_fused_spectra[:, 3::4] = np.max(selected_spectra[:, :-1, :], axis=1)
    print(f'cfs.shape:{cross_fused_spectra.shape}')

    return cross_fused_spectra

#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, number_true, num_classes, nex):
    corr_y_train = []
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
            for ij in range(nex):
                corr_y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes):
        for j in range(number_true[i]):
            y_true.append(i)
    corr_y_train = np.array(corr_y_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    return corr_y_train, y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0
  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)  # batch_pre(64,16)
        batch_target0 = batch_target.long()
        loss = criterion(batch_pred, batch_target0)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data)
        batch_target0 = batch_target.long()
        loss = criterion(batch_pred, batch_target0)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    # 存储计算结果的对象及数组
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data)
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, matrix
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float16)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def list_to_colormap(num_classes):
    y = np.zeros((19, 3))
    y[0] = np.array([1,1,1])#
    y[1] = np.array([1, 1, 0.5])#
    y[2] = np.array([0, 0, 1])#
    y[3] = np.array([1, 0, 0])#
    y[4] = np.array([0, 1, 0.25])#
    y[5] = np.array([1, 0, 1])#
    y[6] = np.array([0.5, 0, 1])#
    y[7] = np.array([0, 0.5, 1])#
    y[8] = np.array([0, 1, 0]) #
    y[9] = np.array([0.5, 0.5, 0.25])#
    y[10] = np.array([0.5, 0, 0.5])#
    y[11] = np.array([0, 0.5, 1])#
    y[12] = np.array([0, 0.25, 0.5])#
    y[13] = np.array([0, 0.5, 0.25])#
    y[14] = np.array([0.5, 0.25, 0])#
    y[15] = np.array([0, 1, 0.5])#
    y[16] = np.array([1, 1, 0])#
    y[17] = np.array([0.25,0.5,1])#
    y[18] = np.array([1,1,0.75])

    return y[:num_classes+1,:]

def classification_map(map, height, width, dpi, save_path = './classification_maps/test'):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    color_matrix = list_to_colormap(np.max(map).astype(int))
    ax.imshow(map, cmap=colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig(save_path, dpi=dpi,bbox_inches='tight', pad_inches=0)#bbox_inches='tight', pad_inches=0
    return 0

def record_output(settings, oa_ae, aa_ae, kappa_ae, conf_matrices_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')
    setrecord = 'model and training settings: ' + settings + '\n'
    f.write(setrecord)
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    combined_lists = zip(*aa_ae)
    combined_lists= list(combined_lists)
    AA_mean = [sum(values) / len(values) for values in combined_lists]
    sentence10 = "AA_mean for each class are: " + str(AA_mean) + '\n'
    f.write(sentence10)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) +'\n'#
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) +' ± ' + str(np.std(aa_ae)) + '\n'#
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) +' ± ' + str(np.std(kappa_ae)) +'\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)
    f.close()

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class
    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

# Pixel Selection in SCHPFP
def pixelSelection(mir_gt_hsi, pos_train, patches, n):
    half_patch = 4 + patches//2
    n_samples = pos_train.shape[0]
    corr_pos_train = np.zeros((n_samples, n, 2), dtype=int)

    for i in range(n_samples):
        x, y = pos_train[i]
        x_min = x + 4
        x_max = x + 4 + patches
        y_min = y + 4
        y_max = y + 4 + patches
        patch = mir_gt_hsi[x_min:x_max, y_min:y_max, :]
        patch_shape = patch.shape[:2]
        center_spectrum = mir_gt_hsi[x + half_patch, y + half_patch, :]
        patch_reshaped = patch.reshape(-1, mir_gt_hsi.shape[2])
        correlations = np.array([np.corrcoef(center_spectrum, p)[0, 1] for p in patch_reshaped])
        top_indices = np.argsort(np.nan_to_num(correlations, nan=0))[-n:]
        top_coords = np.array([np.unravel_index(idx, patch_shape) for idx in top_indices])
        top_global_coords = top_coords + np.array([x_min, y_min]) - half_patch
        corr_pos_train[i] = top_global_coords

    return corr_pos_train

def read_pixelSelection(pos_true, pos_train, patches, n):
    last_column_true = pos_true[:, -1, :]
    indices = np.array([np.where((last_column_true == row).all(axis=1))[0][0] for row in pos_train])
    corr_pos_train = pos_true[indices][:, -n:, :]
    return corr_pos_train
