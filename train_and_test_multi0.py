import Functions0
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import savemat

# from vit_pytorch import ViT
# from vit_RoPE_pytorch import ViT
from src.transformerModel0 import TransformerModel
# import matplotlib.pyplot as plt
# from matplotlib import colors
import numpy as np
import time
import os
# 定义全局可用的参数变量
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['IN', 'PU', 'PV', 'SV', 'KSC', 'BS', 'LK'], default='IN',
                    help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train', 'train and test'], default='train', help='testing mark')
parser.add_argument('--model', choices=['transFormer', 'MPFormer', 'MP+transFormer', 'noneFormer'],
                    default='MP', help='model choice')
parser.add_argument('--embedding', choices=['pos', 'no_pos'], default='no_pos', help='用不用位置编码')
parser.add_argument('--data_mod', choices=['cor', 'repeat', 'rand'], default='cor', help='用不用相关性进行数据增强')
parser.add_argument('--pre_mod', choices=['cross-fuse', 'direct-connect'], default='cross-fuse', help='是否交叉融合')
parser.add_argument('--encoder_num', choices=[1, 2, 3, 4], default=2, help='encoder的数量')
parser.add_argument('--n_pool', choices=[1, 2, 4, 5, 8], default=4, help='最后池化的步长')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=128, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=11, help='number of patches')
parser.add_argument('--cnn_patches', type=int, default=7, help='cnn滑窗边长')
parser.add_argument('--n_corr', type=int, default=6, help='取相关性前n_corr')
parser.add_argument('--n_select', type=int, default=3, help='取相关性前n_corr中的n_select个')
parser.add_argument('--ave_or_cov', choices=['ave', 'cov'], default='ave', help='平均池化或cnn处理cnn_patch')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=10, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--smallTrain', type=bool, default=True, help='specified small train set')
parser.add_argument('--shots', choices=[1, 5, 10, 15, 20], default=5, help='shots number')
parser.add_argument('--VALIDATION_SPLIT', choices=[0.99, 0.95, 0.90, 0.999, 0.995], default=0.90, help='训练样本比例')
parser.add_argument('-ITER', type=int, default=10, help='使用不同的随机种子训练-测试多少次')
args = parser.parse_args()
# 指定使用编号为args.gpu_id的CUDA设备
device = torch.device("cuda", 0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# 设置数据集、模型及训练参数
args.dataset = 'IN'
args.smallTrain = False
args.shots = 3
args.VALIDATION_SPLIT = 0.99

args.epoches = 100  # 本模块的177行有训练跳出的条件
args.ITER = 3  # 10

args.embedding = 'no_pos'  # 'no_pos'/'pos'
args.data_mod = 'rand'  # 'cor'/'repeat'/'rand'
args.pre_mod = 'cross-fuse'  # 'cross-fuse'/'direct-connect'
args.model = 'MP'  # 'transFormer'/'MP'
args.ave_or_cov = 'ave'  # 'ave'/'cov'

args.n_corr = 13  # 13
args.n_select = 4  # 4
args.encoder_num = 3  # 3
args.n_pool = 4

args.patches = 7  # 7
args.cnn_patches = 9  # 9
oabi = 4 + (args.patches - args.cnn_patches)//2  # 还要扩大，满足rand的需求
args.batch_size = 256  # 512 256 128 64 32 16
args.learning_rate = 5e-4

args.seed = 2
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# 使用CUDA加速的CuDNN库时开启确定性模式
cudnn.deterministic = True
# 内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法
cudnn.benchmark = False
# 记录每次 训练-测试 结果
multi_conf_matrix = []
multi_KAPPA = []
multi_OA = []
multi_AA = []
multi_TRAINING_TIME = []
multi_TESTING_TIME = []
multi_PARAMETER = []
# prepare data
for index_iter in range(args.ITER):  # range(10) = [0,1,2...,9]
    # -------------------------------------------------------------------------------
    # obtain train and test data
    # 获取训练集、测试集、所有标签每个类别像素的（x,y）坐标信息
    # IndianPine训练集样本个数695，测试集样本个数9671，未标注（背景）样本个数10659，总样本个数145*145
    # choose_train_and_test_point
    # 返回训练坐标集、测试坐标集和所有样本坐标集，且已经实现了坐标集按照类别由小到大顺序进行排列

    data_hsi, gt_hsi, num_classes, total_pos_train, total_pos_test, total_pos_true, number_train, number_test,\
        number_true = Functions0.load_dataset(args.dataset, args.shots, args.VALIDATION_SPLIT, args.smallTrain)
    height, width, band = data_hsi.shape
    band = 64

    # 加载颜色映射矩阵
    color_matrix = Functions0.list_to_colormap(num_classes)
    # 当使用patch而非单个像素作为输入时，对原图像进行边界镜像扩展，以满足在图像边界像素的patch需求
    # 此处mirror镜像上下左右，注意上下左右都是添加了patch//2层二维矩阵
    # 1 F149
    mirror_image, zero_image = Functions0.mirror_hsi(height, width, band, data_hsi, patch=args.patches,
                                                    cnn_patch=args.cnn_patches,ds=args.dataset)

    # 2 F667
    total_corr_pos_train = Functions0.corr_data(zero_image, total_pos_train, patches=args.patches,
                                               n=args.n_corr, mod=args.data_mod)
    print(f"total_corr_pos_train  shape: {total_corr_pos_train.shape}")  # n_cor = 6
    total_corr_pos_test = Functions0.corr_data(zero_image, total_pos_test, patches=args.patches,
                                              n=args.n_select+1, mod=args.data_mod)
    print(f"total_corr_pos_test  shape: {total_corr_pos_test.shape}")  # n_se = 3
    total_corr_pos_true = Functions0.corr_data(zero_image, total_pos_true, patches=args.patches,
                                              n=args.n_select+1, mod=args.data_mod)
    print(f"total_corr_pos_true  shape: {total_corr_pos_true.shape}")

    # corr_x_train\corr_x_test\corr_x_true
    # shape = (20620\9218\10249, 3, 7, 7, 200), type = float64
    # 3 F180 242
    corr_x_train_band, corr_x_test_band, corr_x_true, nnn_ex = Functions0.corr_train_and_test_data(mirror_image,
                                                                                                  total_corr_pos_train,
                                                                                                  total_corr_pos_test,
                                                                                                  total_corr_pos_true,
                                                                                                  oabi,
                                                                                                  patch=
                                                                                                  args.cnn_patches,
                                                                                                  nnc=args.n_corr,
                                                                                                  nns=args.n_select)

    # 按照类别由小到大的顺序获取光谱立方体集对应的类别标签
    corr_y_train, y_train, y_test, y_true = Functions0.train_and_test_label(number_train, number_test, number_true,
                                                                           num_classes, nnn_ex)

    # -------------------------------------------------------------------------------
    # load data

    x_train = torch.from_numpy(corr_x_train_band.transpose(0, 1, 4, 2, 3))  # [20*n, 200, 3, n_cnn, c_cnn]
    corr_y_train = torch.from_numpy(corr_y_train)  # [20*n]
    Label_train = Data.TensorDataset(x_train, corr_y_train)
    corr_x_test = torch.from_numpy(corr_x_test_band.transpose(0, 1, 4, 2, 3))  # [9671, 200, 3, c_cnn, n_cnn]
    y_test = torch.from_numpy(y_test)  # [9671]
    Label_test = Data.TensorDataset(corr_x_test, y_test)
    x_true = torch.from_numpy(corr_x_true.transpose(0, 1, 4, 2, 3))
    y_true = torch.from_numpy(y_true)
    Label_true = Data.TensorDataset(x_true, y_true)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=args.batch_size, shuffle=False)

    # rnn batch size=128,其它batch size=64

    # -------------------------------------------------------------------------------
    # create model
    model = TransformerModel(input_dim=band, nnss=args.n_select, num_classes=num_classes, n_pool=args.n_pool,
                             cnn_m=args.ave_or_cov, in_model=args.model, en_m=args.embedding,
                             encoder_n=args.encoder_num, cnn_pa=args.cnn_patches, pre_m=args.pre_mod)
    # num_classes要根据数据集修改
    # 找打印参数量的函数
    # 轻量化、精度更高（对比transfoemer）、小样本效果高（）、样本不均衡时的效果好（百5，百10）

    # 模型加载到GPU上
    model = model.cuda()
    # model=model.to(device)
    # criterion 定义损失函数并加载到GPU上
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer 定义训练神经网络的优化器为Adam，形参为：需要优化的参数，学习率，L2正则化项的权重
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 设置训练过程中调整学习率的方法，形参为：optimizer:神经网络所使用的优化器，step_size: 多少轮循环后更新一次学习率，gamma: 每次将 lr 更新为原来的 gamma 倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------

    # train
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        scheduler.step()
        # 优化器是存放参数和学习率的，lr_scheduler会去调整优化器里的学习率，所有lr_scheduler必须要关联一个优化器才能改动里面的学习率
        # 学习率调度器在优化器调度之后
        # train model
        device = torch.device("cuda", 0)
        model = model.to(device)
        model.train()
        train_acc, train_obj, tar_t, pre_t = Functions0.train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1, confusionMatrix = Functions0.output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, train_obj, train_acc))
        if (train_obj < 0.01) & (train_acc > 99.5):  # 0.1,0.05,0.01
            break

        # if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
        #     model.eval()
        #     # valid_acc总体预测精度均值，valid_obj总体损失函数值均值，tar标签类别列表，pre预测类别值列表
        #     valid_acc, valid_obj, tar_v, pre_v = Functions.valid_epoch(model, label_test_loader, criterion, optimizer)
        #     OA2, AA_mean2, Kappa2, AA2, confusionMatrix = Functions.output_metric(tar_v, pre_v)

    toc = time.time()
    multi_TRAINING_TIME.append(toc - tic)
    print("Running Time: {:.2f}".format(toc - tic))
    multi_PARAMETER.append(sum(p.numel() for p in model.parameters()))
    print("**************************************************")
    print("--------" + args.model + " Training Finished-----------")
    # if not os.path("./pt"):
    #     os.makedirs("./pt")
    if args.smallTrain:
        pt_path = f'pt/small_sample_{args.smallTrain}/{args.shots}shot/{args.model}_{args.dataset}' \
                  f'/_iter{index_iter}.pt'
        path = f'pt/small_sample_{args.smallTrain}/{args.shots}shot/{args.model}_{args.dataset}' \
               f'/_iter{index_iter}.pt'
        map_path = f'classification_maps/small_sample_{args.smallTrain}/{args.shots}shot/{args.model}' \
                   f'_{args.dataset}/_iter{index_iter}'
    else:
        pt_path = f'pt/small_sample_{args.smallTrain}/{args.VALIDATION_SPLIT}sample/{args.model}_{args.dataset}' \
                  f'/_iter{index_iter}.pt'
        path = f'pt/small_sample_{args.smallTrain}/{args.VALIDATION_SPLIT}sample/{args.model}_{args.dataset}' \
               f'/_iter{index_iter}.pt'
        map_path = f'classification_maps/small_sample_{args.smallTrain}/{args.VALIDATION_SPLIT}sample/{args.model}' \
                   f'_{args.dataset}/_iter{index_iter}'


    print(pt_path)
    torch.save(model.state_dict(), pt_path)
    Functions0.print_args(vars(args))

    # test
    model.load_state_dict(torch.load(path))

    # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
    model.eval()
    tic = time.time()
    # valid_acc总体预测精度均值，valid_obj总体损失函数值均值，tar标签类别列表，pre预测类别值列表
    valid_acc, valid_obj, tar_v, pre_v = Functions0.valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2, confusionMatrix2 = Functions0.output_metric(tar_v, pre_v)
    toc = time.time()
    multi_TESTING_TIME.append(toc - tic)
    # output classification maps
    pre_u = Functions0.test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

    # 存储计算结果和分类图
    savemat(map_path + '_matrix.mat', {'P': prediction_matrix, 'label': gt_hsi})
    Functions0.classification_map(prediction_matrix, height, width, 300, path+'.png')
    # classification_map(gt_hsi, height,width, 300, path+'_gt.png')
    print('------Get classification maps successful-------')
    print(f"第 {index_iter} 次测试结果:")
    print("OA: {:.4f} | AA_mean: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print("AA: ")
    print(AA2)
    print("**************************************************")
    print("Parameter:")
    Functions0.print_args(vars(args))
    multi_OA.append(OA2)
    multi_AA.append(AA2)
    multi_KAPPA.append(Kappa2)
    multi_conf_matrix.append(confusionMatrix)
    print(f'模型参数量：{multi_PARAMETER}')
    print("{:.4f} ± {:.4f}".format(np.mean(multi_OA), np.std(multi_OA)))
    print("{:.4f} ± {:.4f}".format(np.mean(multi_AA), np.std(multi_AA)))
    print("{:.4f} ± {:.4f}".format(np.mean(multi_KAPPA), np.std(multi_KAPPA)))
    print("{:.4f}/{:.4f}".format(np.mean(multi_TRAINING_TIME), np.std(multi_TESTING_TIME)))

print("---------------------------------------------")
print("---------------------------------------------")
print("---------------Process finished---------------------")

if args.smallTrain:
    record_path = f"records/small_sample_{args.smallTrain}/{args.shots}shot_patches{args.patches}_" \
                  f"encoder{args.encoder_num}_cnn_patches{args.cnn_patches}_n_cor{args.n_corr}_n_sel_{args.n_select}" \
                  f"_n_pool{args.n_pool}_{args.dataset}_batch{args.batch_size}.txt"
else:
    record_path = f"records/small_sample_{args.smallTrain}/atest{args.VALIDATION_SPLIT}test_{args.data_mod}_{args.pre_mod}_{args.model}_patches{args.patches}_" \
                  f"{args.embedding}_{args.ave_or_cov}" \
                  f"_n_pool{args.n_pool}_{args.dataset}_batch{args.batch_size}_seed2.txt"

Functions0.record_output(str(vars(args)), multi_OA, multi_AA, multi_KAPPA, multi_conf_matrix,
                        multi_TRAINING_TIME, multi_TESTING_TIME, record_path)
print(f"{np.mean(multi_OA)}, {np.mean(multi_AA)}, {np.mean(multi_KAPPA)}")
print(f'模型参数量：{multi_PARAMETER}')
print("{:.4f} ± {:.4f}".format(np.mean(multi_OA), np.std(multi_OA)))
print("{:.4f} ± {:.4f}".format(np.mean(multi_AA), np.std(multi_AA)))
print("{:.4f} ± {:.4f}".format(np.mean(multi_KAPPA), np.std(multi_KAPPA)))
print("{:.4f}/{:.4f}".format(np.mean(multi_TRAINING_TIME), np.std(multi_TESTING_TIME)))
print("-----------------All Over---------------------")

# args.embedding = 'no_pos'  # 'no_pos'/'pos'
# args.data_mod = 'cor'  # 'cor'/'repeat'/'rand'
# args.pre_mod = 'cross-fuse'  # 'cross-fuse'/'direct-connect'
# args.model = 'MP'
# args.encoder_num = 2
# args.n_pool = 4
# args.patches = 11
# args.cnn_patches = 7
# args.ave_or_cov = 'ave'  # 'ave'/'cov'
# args.n_corr = 6  # 6
# args.n_select = 3  # 3
# args.batch_size = 128  # 128 64 32 16
# args.learning_rate = 5e-4
