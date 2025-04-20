import scipy.io as sio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from itertools import combinations

# -------------------------------------------------------------------------------
# 定位训练和测试样本
# 以IndianPine.mat为例，train_data为训练集标记类别145*145，test_data为测试集标记类别145*145，
# true_data为所有标记145*145，true_data为原始影像数据145*145*200
# 在标签中，没有选中的像素点标记值为0
# train/test/true坐标集按照类别由小到大顺序进行排列
# 进行小样本训练则smallSampling = Ture
# 直接看调试部分

def load_dataset(Dataset, VALIDATION_SPLIT, np_random_seed=6):
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
    #------------------获取每一类别的像素位置和像素个数，并按照类别序号由小到大的顺序存储---------------
    for i in range(num_classes):
        each_class = []
        # 获取类别值等于i+1的下标（像素2D位置，即x,y），其中下标为0代表没有类型标记的像素点
        each_class = np.argwhere(gt_hsi==(i+1))
        # 获取该类别的所有像素点个数，追加到number_gt序列中
        each_class_num = each_class.shape[0]
        number_gt.append(each_class_num)
        # pos_gt[i]存储第i+1类别的所有像素坐标信息
        # 使用随机种子打乱坐标信息顺序，以划分train和test
        np.random.shuffle(each_class)
        pos_gt[i] = each_class

        number_train_i = np.ceil(each_class_num*(1-VALIDATION_SPLIT)).astype(int)#根据VALIDATION_SPLIT获取第i个类别用于训练的样本个数
        number_test_i = each_class_num - number_train_i

        number_test.append(number_test_i)
        number_train.append(number_train_i)
        pos_train[i] = each_class[0:number_train_i,:]#将 each_class 中的前 number_train_i 行作为第 i 个类别的训练集，然后将结果赋值给 pos_train[i]
        pos_test[i] = each_class[number_train_i:,:]
        pos_gt[i] = each_class

    # 将pos_gt数组转换成一个大矩阵total_pos_gt，且已经实现了坐标集按照类别由小到大顺序进行排列
    total_pos_gt = pos_gt[0]
    total_pos_train = pos_train[0]
    total_pos_test = pos_test[0]
    # print(f'number_train :{number_train}')
    for i in range(1, num_classes):
        # 合并矩阵，保持列数不变
        total_pos_gt = np.r_[total_pos_gt, pos_gt[i]] #(695,2)
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_gt = total_pos_gt.astype(int)
    total_pos_train = total_pos_train.astype(int)
    total_pos_test = total_pos_test.astype(int)

    return input_normalize,gt_hsi,num_classes, total_pos_train, total_pos_test, total_pos_gt, number_train, number_test, number_gt
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(ds,height,width,band,input_normalize,patch,cnn_patch):
    # 除法，取向下接近的整数（非四舍五入）
    padding = 4 + patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    zero_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    zero_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    # 镜像似乎没有达到需求----------
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    # print("**************************************************")
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    # print("**************************************************")
    return mirror_hsi, zero_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
# 以第一个像素点为例，左上角第一个点(x0,y0)+patch//2为镜像区域，所以刚好第一个像素点位于patch中心位置，
def gain_neighborhood_pixel(mirror_image, point, i, patch=7, oa_bi=0):
    x = point[i,0]
    y = point[i,1]
    temp_image = np.mean(mirror_image[x+oa_bi:(x+patch+oa_bi), y+oa_bi:(y+patch+oa_bi), :], axis=(0, 1), keepdims=False)
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    # 将（pointNum, band, patch*patch）展平为了（pointNum, band, patch*patch）
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    # x_train_band的size （pointNum, patch*patch*band_patch, band）
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据，以patch的形式，其中训练/测试的像素点在patch中心位置
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    # 得到每一个训练/测试/完整数据集 所有像素点坐标的patch数据
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    # 得到每一个训练/测试/完整数据集 所有像素点光谱信息的patch数据
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train, x_test, x_true

def corr_train_and_test_data(mirror_image, band, corr_pos_train, test_point, true_point, oa_bi, patch=5, nnc=6, nns=3):
    n_train, n_corr, _ = corr_pos_train.shape
    x_train = np.zeros((n_train, n_corr, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], nns+1, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], nns+1, band), dtype=float)

    # 得到每一个训练点的6个相关点的patch
    for i in range(n_train):
        for j in range(n_corr):
            vc = gain_neighborhood_pixel(mirror_image, corr_pos_train[i], j, patch, oa_bi)
            x_train[i, j, :] = vc

    # C(6,3)展开，排列保持相关性从小到大的顺序
    x_train_band, nn_ex = combs_exp(x_train, nnc, nns)

    # 得到测试数据的patch数据
    for i in range(test_point.shape[0]):
        for j in range(nns+1):
            x_test[i, j, :] = gain_neighborhood_pixel(mirror_image, test_point[i], j, patch, oa_bi)
    x_test[:, :-1, :] = x_test[:, 1:, :]

    # 得到完整数据集的patch数据
    for i in range(true_point.shape[0]):
        for j in range(nns+1):
            x_true[i, j, :] = gain_neighborhood_pixel(mirror_image, true_point[i], j, patch, oa_bi)
    x_true[:, :-1, :] = x_true[:, 1:, :]

    print("corr_x_train shape = {}, type = {}".format(x_train_band.shape, x_train.dtype))
    print("corr_x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("corr_x_true  shape = {}, type = {}".format(x_true.shape, x_true.dtype))
    print("**************************************************")

    return x_train_band, x_test, x_true, nn_ex

def get_combs(a, b):
    # 生成从 0 到 a-1 的列表
    elements = list(range(a))
    # 获取所有组合
    comb_array = list(combinations(elements, b))
    # 转换为 NumPy 数组
    result = np.append(comb_array, np.full((np.shape(comb_array)[0], 1), a - 1), axis=1)
    return np.array(result), np.shape(result)[0]

def combs_exp(x, nc, ns):
    combs, n_ex = get_combs(nc, ns)

    # 获取 x 的形状
    x0, x1, x4 = x.shape

    # 重新排列 x
    X = np.empty((x0 * n_ex, ns+1, x4))

    for i in range(x0):
        X[i * n_ex:(i + 1) * n_ex] = x[i, combs]
    return X, n_ex

#-------------------------------------------------------------------------------
# 标签y_train, y_test
# 形参number_* 每个类别的训练/测试/所有样本数，是一个一维列表，num_classes类别数
#
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
    # for i in range(num_classes+1):
    for i in range(num_classes):
        for j in range(number_true[i]):
            y_true.append(i)
    corr_y_train = np.array(corr_y_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    # print("corr_y_train: shape = {} ,type = {}".format(corr_y_train.shape, corr_y_train.dtype))
    # print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    # print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    # print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    # print("**************************************************")
    return corr_y_train, y_train, y_test, y_true
#-------------------------------------------------------------------------------
# 求解预测结果的均值
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
# 计算预测精度
# 形参中的topk=(1,) 定义了一个元组（不可改变的列表），该元组名为topk，只有一个元素，值为1
# output尺寸 batchSize*catagoryNum, target尺寸 batchSize
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)
  # 此处的topk是torch的一个内置函数，用来求解tensor中某个维度（dim）的前k大或前k小的值以及对应的下标index
  # torch.topk(input,k,dim=None,largest=True,sorted=True,out=None)，返回Tensor和LongTensor
  # 常用于求一个样本被网络认为前k个最可能属于的类别，例如IndianPine为16中类别，此处获取16种类别的排序最高分对应的下标（类别）
  _, pred = output.topk(maxk, 1, True, True)
  # 矩阵转置
  pred = pred.t()
  # view函数：将target的维度改为1*N，其中N计算而得
  # expand_as仿照括号内的张量的size对调用者进行扩展，使两者size相同
  # eq方法比较调用者与形参每个元素是否相等
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  # 此处topk=1，也就是k=[0,1)
  for k in topk:
    #   获取[0,k)维度数据，按顺序展平为1维，转换为float类型，然后求和，sum（[start=0]）其中start为相加的参数
    correct_k = correct[:k].view(-1).float().sum(0)
    # 计算获取一个batch中预测正确的像素点个数比例
    res.append(correct_k.mul_(100.0/batch_size))
  # squeeze降维，torch.squeeze(input, dim=None, out=None)，去除size为1的维度，包括行和列。当维度大于等于2时，squeeze()无作用
  # res 一个batch中预测正确的比例*100，target真值，pred.squeeze()预测类别值矩阵压缩为列表
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    #batch数据下标，图像矩阵，标签真值
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        # 每轮训练，将梯度置零
        optimizer.zero_grad()
        # 使用模型预测结果
        batch_pred = model(batch_data)  # batch_pre(64,16)
        # 确保 target 是 torch.LongTensor
        batch_target0 = batch_target.long()
        # 计算损失函数值
        loss = criterion(batch_pred, batch_target0)
        # 对损失函数求导，得到损失函数的梯度
        loss.backward()
        # 更新模型的参数，从而最小化损失函数
        optimizer.step()
        # prec1 一个batch中预测正确的比例*100，t标签类别列表，p预测类别列表
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        # n为batchSize
        n = batch_data.shape[0]
        # 累计损失函数值，self.sum+=loss.data * n
        objs.update(loss.data, n)
        # 累计预测精度值,self.sum+=prec1的值*n
        top1.update(prec1[0].data, n)
        # batch真值列表由GPU迁移到CPU并合并
        tar = np.append(tar, t.data.cpu().numpy())
        # batch预测值列表由GPU迁移至CPU并合并
        pre = np.append(pre, p.data.cpu().numpy())
    #     top1.avg总体预测精度均值，objs.avg总体损失函数值均值，tar标签类别列表，pre预测类别值列表
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    # 存储最终结果的对象及数组
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # enumerate函数作用：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        # 数据加载到GPU
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        # 使用模型进行预测，一次性预测一个batch（如64）个像素点类别
        batch_pred = model(batch_data)
        batch_target0 = batch_target.long()
        # 计算损失函数值
        loss = criterion(batch_pred, batch_target0)
        # prec1 一个batch中预测正确的比例*100，t标签类别列表，p预测类别列表
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        # n为batchSize
        n = batch_data.shape[0]
        # 累计损失函数值，self.sum+=loss.data * n
        objs.update(loss.data, n)
        # 累计预测精度值,self.sum+=prec1的值*n
        top1.update(prec1[0].data, n)
        # batch真值列表由GPU迁移到CPU并合并
        tar = np.append(tar, t.data.cpu().numpy())
        # batch真值列表由GPU迁移到CPU并合并
        pre = np.append(pre, p.data.cpu().numpy())
    #     top1.avg总体预测精度均值，objs.avg总体损失函数值均值，tar标签类别列表，pre预测类别值列表
    return top1.avg, objs.avg, tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    # 存储计算结果的对象及数组
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # batch数据下标，图像数据，对应标签
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        # 数据加载到GPU
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        # 模型预测batch数据
        batch_pred = model(batch_data)
        # 此处的topk是torch的一个内置函数，用来求解tensor中某个维度（dim）的前k大或前k小的值以及对应的下标index
        # torch.topk(input,k,dim=None,largest=True,sorted=True,out=None)，返回Tensor和LongTensor
        # 常用于求一个样本被网络认为前k个最可能属于的类别，例如IndianPine为16中类别，此处获取16种类别的排序最高分对应的下标（类别）
        _, pred = batch_pred.topk(1, 1, True, True)
        # 压缩维度
        pp = pred.squeeze()
        # GPU数据迁移到CPU，追加保存到pre数组中
        pre = np.append(pre, pp.data.cpu().numpy())
    #     返回预测类别值数组
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    # 计算混淆矩阵
    matrix = confusion_matrix(tar, pre)
    # 计算性能指标
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, matrix
#-------------------------------------------------------------------------------
def cal_results(matrix):
    # 获取维度信息
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # 平均准确度（AA）—表示分类精度的平均值
    AA = np.zeros([shape[0]], dtype=np.float16)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    #     总体分类精度（OA）：指被正确分类的类别像元数与总的类别个数的比值
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    # Kappa系数（Kappa）:Kappa系数是一种比例，代表着分类与完全随机的分类产生错误减少的比例
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------

# 将标签类别图转换成RGB色彩图
def list_to_colormap(num_classes):
    y = np.zeros((19, 3))
    y[0] = np.array([1,1,1])#000黑色~111白色
    y[1] = np.array([1, 1, 0.5])#浅青色
    y[2] = np.array([0, 0, 1])#蓝色
    y[3] = np.array([1, 0, 0])#红色
    y[4] = np.array([0, 1, 0.25])#浅绿色
    y[5] = np.array([1, 0, 1])#品红色
    y[6] = np.array([0.5, 0, 1])#蓝紫色
    y[7] = np.array([0, 0.5, 1])#绿松石色
    y[8] = np.array([0, 1, 0]) #绿色[1, 0.7, 0][0, 1, 0]
    y[9] = np.array([0.5, 0.5, 0.25])#浅灰色
    y[10] = np.array([0.5, 0, 0.5])#紫罗兰
    y[11] = np.array([0, 0.5, 1])#青绿色
    y[12] = np.array([0, 0.25, 0.5])#浅蓝绿
    y[13] = np.array([0, 0.5, 0.25])#浅黄绿
    y[14] = np.array([0.5, 0.25, 0])#黄绿色
    y[15] = np.array([0, 1, 0.5])#深青色
    y[16] = np.array([1, 1, 0])#灰色
    y[17] = np.array([0.25,0.5,1])#
    y[18] = np.array([1,1,0.75])

    return y[:num_classes+1,:]

def classification_map(map, height, width, dpi, save_path = './classification_maps/test'):
    fig = plt.figure(frameon=False)

    # 创建一个 Axes 对象，设置其位置和大小以填充整个 Figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()  # 关闭坐标轴
    fig.add_axes(ax)  # 将 Axes 添加到 Figure 中
    # 根据 map 的最大值创建颜色映射
    color_matrix = list_to_colormap(np.max(map).astype(int))
    # 使用 imshow 显示图像，并应用颜色映射
    ax.imshow(map, cmap=colors.ListedColormap(color_matrix))
    # 移除坐标轴的刻度
    plt.xticks([])
    plt.yticks([])
    # 显示图像
    plt.show()

    # 保存图像，指定 DPI 和保存路径
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
    # 使用zip()将三个子列表对应位置上的元素组合在一起
    combined_lists = zip(*aa_ae)
    # 将生成器对象转换为列表
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
    # sum_matrix = np.sum(conf_matrices_ae, axis=0)
    # average_matrix = sum_matrix / len(conf_matrices_ae)
    # std_matrix = np.std(conf_matrices_ae, axis=0)
    # sentence8 = "Mean of all elements in confusion matrix: " + str(average_matrix) + '\n'
    # f.write(sentence8)
    # sentence9 = "Standard deviation of all elements in confusion matrix: " + str(std_matrix) + '\n'
    # f.write(sentence9)
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
        # 获取类别值等于i+1的下标（像素2D位置，即x,y）
        each_class = np.argwhere(train_data==(i+1))
        # 获取该类别的所有像素点个数，追加到number_train序列中
        number_train.append(each_class.shape[0])
        # pos_train[i]存储第i+1类别的所有像素坐标信息
        pos_train[i] = each_class
    # 将pos_train数组转换成一个大矩阵total_pos_train，且已经实现了坐标集按照类别由小到大顺序进行排列
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        # 合并矩阵，保持列数不变
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
    # 将pos_test数组转换成一个大矩阵total_pos_test，且已经实现了坐标集按照类别由小到大顺序进行排列
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
    # 将pos_true数组转换成一个大矩阵total_pos_true，且已经实现了坐标集按照类别由小到大顺序进行排列
    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    # 返回训练坐标集、测试坐标集和所有样本坐标集，且已经实现了坐标集按照类别由小到大顺序进行排列
    # 返回每种类别的训练/测试/所有样本个数列表
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

def corr_data(mir_gt_hsi, pos_train, patches, n):
    half_patch = 4 + patches//2
    n_samples = pos_train.shape[0]
    corr_pos_train = np.zeros((n_samples, n, 2), dtype=int)
    # 遍历每一个样本点
    for i in range(n_samples):
        x, y = pos_train[i]

        # # 获取patch的边界
        # x_min = max(x - half_patch, 0)
        # x_max = min(x + half_patch + 1, mir_gt_hsi.shape[0])
        # y_min = max(y - half_patch, 0)
        # y_max = min(y + half_patch + 1, mir_gt_hsi.shape[1])
        # 获取patch的边界
        x_min = x + 4
        x_max = x + 4 + patches
        y_min = y + 4
        y_max = y + 4 + patches

        # 提取patch
        patch = mir_gt_hsi[x_min:x_max, y_min:y_max, :]
        patch_shape = patch.shape[:2]
        # 提取中心点的光谱数据
        center_spectrum = mir_gt_hsi[x + half_patch, y + half_patch, :]
        # 计算相关性
        patch_reshaped = patch.reshape(-1, mir_gt_hsi.shape[2])
        correlations = np.array([np.corrcoef(center_spectrum, p)[0, 1] for p in patch_reshaped])
        # 找到相关性最高的6个点
        top_indices = np.argsort(np.nan_to_num(correlations, nan=0))[-n:]
        # 将线性索引转换为二维索引
        top_coords = np.array([np.unravel_index(idx, patch_shape) for idx in top_indices])

        # 转换为全局坐标
        top_global_coords = top_coords + np.array([x_min, y_min]) - half_patch

        # 保存到输出张量中
        corr_pos_train[i] = top_global_coords

    return corr_pos_train

def read_corr(pos_true, pos_train, patches, n):
    half_patch = 4 + patches//2
    n_samples = pos_train.shape[0]
    corr_pos_train = np.zeros((n_samples, n, 2), dtype=int)
    # 提取 total_corr_pos_true 的最后一列 (shape: (10249, 2))
    last_column_true = pos_true[:, -1, :]
    # 找到 total_pos_test 中每一行在 last_column_true 中的索引
    indices = np.array([np.where((last_column_true == row).all(axis=1))[0][0] for row in pos_train])
    # 根据索引提取 total_corr_pos_true 的对应行，形成 total_corr_pos_test
    corr_pos_train = pos_true[indices][:, -n:, :]
    return corr_pos_train

if __name__ == "__main__":
    import torch
    import torch.utils.data as Data

    print(torch.__version__)

    patches = 11  # 相关性窗口的边长，在这个窗口内进行相关性分析
    cnn_patches = 7  # 邻域窗口，以选中的点为中心，整个邻域输入
    n_corr = 7
    n_select = 4
    batch_size = 16
    band_patches = 1
    smallTrain = False
    mod = 'rand'  # 'repeat'/'cor'/'rand'

    data_hsi, gt_hsi, num_classes, total_pos_train, total_pos_test, total_pos_true, number_train, number_test,\
        number_true = load_dataset('IN', 1, 0.95, smallTrain)  # 载入数据，划分训练集和测试集
    height, width, band = data_hsi.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    # 加载颜色映射矩阵
    color_matrix = list_to_colormap(num_classes)

    # 边界扩充
    # 当使用patch而非单个像素作为输入时，对原图像进行边界镜像扩展，以满足在图像边界像素的patch需求
    # 此处mirror镜像上下左右，注意上下左右都是添加了patch//2层二维矩阵
    mirror_image = mirror_hsi(height, width, band, data_hsi, patch=patches)  # 1

    # 通过相关性进行数据增强  # 2
    total_corr_pos_train = corr_data(mirror_image, total_pos_train, n=n_corr, mod=mod)  # total_pos_train为训练集的坐标数组(n,2)
    print(f"total_corr_pos_train  shape: {total_corr_pos_train.shape}")  # 相关性窗口内选出的6个相关性最大的像素，从小到大组成(n,6,2)
    total_corr_pos_test = corr_data(mirror_image, total_pos_test, n=n_select+1, mod=mod)
    print(f"total_corr_pos_test  shape: {total_corr_pos_test.shape}")  # 相关性窗口内选出的3个相关性最大的像素，(n,3,2)
    total_corr_pos_true = corr_data(mirror_image, total_pos_true, n=n_select+1, mod=mod)
    print(f"total_corr_pos_true  shape: {total_corr_pos_true.shape}")

    print(f'number_test :{sum(number_test)}')
    print(f'number_train :{sum(number_train)}')
    print(f'number_true :{sum(number_true)}')

    # 根据坐标取邻域光谱，即训练集：(n,6,2)→(n,6,7,7,200); 训练集：(n,3,2)→(n,3,7,7,200)
    # 训练集排列组合展开，即：(n,6,7,7,200)→(n*20,7,7,200)，20=C(6,3)
    # 3
    corr_x_train_band, corr_x_test_band, corr_x_true, nnn_ex = corr_train_and_test_data(mirror_image, band,
                                                                                total_corr_pos_train,
                                                                                total_corr_pos_test,
                                                                                total_corr_pos_true,
                                                                                patch=cnn_patches,
                                                                                nnc=n_corr,
                                                                                nns=n_select)

    # 按照类别由小到大的顺序获取光谱立方体集对应的类别标签
    corr_y_train, y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes,
                                                                 nnn_ex)

    # -------------------------------------------------------------------------------
    # 调整数据维度顺序，用torch的类存储数据

    x_train = torch.from_numpy(corr_x_train_band.transpose(0, 4, 1, 2, 3))  # [20*n, 200, 3, n_cnn, c_cnn]
    corr_y_train = torch.from_numpy(corr_y_train)  # [20*n]
    Label_train = Data.TensorDataset(x_train, corr_y_train)
    corr_x_test = torch.from_numpy(corr_x_test_band.transpose(0, 4, 1, 2, 3))  # [9671, 200, 3, n_cnn, n_cnn]
    y_test = torch.from_numpy(y_test)  # [9671]
    Label_test = Data.TensorDataset(corr_x_test, y_test)
    x_true = torch.from_numpy(corr_x_true.transpose(0, 4, 1, 2, 3))
    y_true = torch.from_numpy(y_true)
    Label_true = Data.TensorDataset(x_true, y_true)

    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=batch_size, shuffle=False)



