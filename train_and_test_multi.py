import Functions
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import savemat
from ESSPFormer import ESSPFormerModel
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--dataset', choices=['IN', 'PU', 'SV', 'HC'], default='IN')
parser.add_argument('--VALIDATION_SPLIT', choices=[0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995, 0.999])

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=5.5e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--ITER', type=int, default=10)
parser.add_argument('--epoches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--patch', type=int, default=7)
parser.add_argument('--hpf_patch', type=int, default=9)
parser.add_argument('--pixelNum', type=int, default=12)
parser.add_argument('--n_perGroup', type=int, default=4)
parser.add_argument('--encoder_num', choices=[1, 2, 3, 4], default=3)
parser.add_argument('--groupPoolScale', choices=[4, 8, 16, 24, 32], default=4)
args = parser.parse_args()
device = torch.device("cuda", 0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.dataset = 'PU'
args.VALIDATION_SPLIT = 0.999
args.ITER = 2  # 10
oabi = 4 + (args.patch - args.hpf_patch)//2

cudnn.deterministic = True
cudnn.benchmark = False

multi_conf_matrix = []
multi_KAPPA = []
multi_OA = []
multi_AA = []
multi_TRAINING_TIME = []
multi_TESTING_TIME = []
multi_PARAMETER = []
Functions.print_args(vars(args))
# prepare data
for index_iter in range(args.ITER):
    seed = args.seed + index_iter
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # -------------------------------------------------------------------------------

    data_hsi, gt_hsi, num_classes, total_pos_train, total_pos_test, total_pos_true, number_train, number_test,\
        number_true = Functions.load_dataset(args.dataset, args.VALIDATION_SPLIT,)
    height, width, band = data_hsi.shape
    color_matrix = Functions.list_to_colormap(num_classes)
    mirror_image, zero_image = Functions.mirror_hsi(args.dataset, height, width, band, data_hsi, patch=args.patch)

    # SCHPFP
    corr_x_train,corr_x_test,corr_x_true = Functions.SCHPFP(args.dataset,args.patch,args.hpf_patch,
                                                            args.n_perGroup,args.pixelNum,band,oabi,
                                                            total_pos_test,total_pos_train,
                                                            total_pos_true,zero_image,mirror_image)

    # SSGPCF
    x_train_band,x_test_band,x_true_band,nnn_ex = Functions.SSGPCF(corr_x_train,corr_x_test,corr_x_true,
                                                                   args.pixelNum,args.n_perGroup)

    corr_y_train, y_train, y_test, y_true = Functions.train_and_test_label(number_train, number_test, number_true,
                                                                           num_classes, nnn_ex)

    # -------------------------------------------------------------------------------
    # load data

    x_train = torch.from_numpy(x_train_band).float()
    corr_y_train = torch.from_numpy(corr_y_train)
    Label_train = Data.TensorDataset(x_train, corr_y_train)
    corr_x_test = torch.from_numpy(x_test_band).float()
    y_test = torch.from_numpy(y_test)
    Label_test = Data.TensorDataset(corr_x_test, y_test)
    x_true = torch.from_numpy(x_true_band).float()
    y_true = torch.from_numpy(y_true)
    Label_true = Data.TensorDataset(x_true, y_true)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------------
    # create model
    model = ESSPFormerModel(input_dim=band, num_classes=num_classes, groupPoolScale=args.groupPoolScale,
                            encoder_n=args.encoder_num)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------

    # train
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        scheduler.step()
        # train model
        device = torch.device("cuda", 0)
        model = model.to(device)
        model.train()
        train_acc, train_obj, tar_t, pre_t = Functions.train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1, confusionMatrix = Functions.output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, train_obj, train_acc))
        if (train_obj < 0.01) & (train_acc > 99.5):  # 0.1,0.05,0.01
            break

    toc = time.time()
    multi_TRAINING_TIME.append(toc - tic)
    print("Running Time: {:.2f}".format(toc - tic))
    multi_PARAMETER.append(sum(p.numel() for p in model.parameters()))
    print("**************************************************")
    print("-------- MP Training Finished-----------")
    pt_path = f'pt/{args.VALIDATION_SPLIT}sample/{args.dataset}_iter{index_iter}.pt'
    path = f'pt/{args.VALIDATION_SPLIT}sample/{args.dataset}_iter{index_iter}.pt'
    map_path = f'classification_maps/{args.VALIDATION_SPLIT}sample/{args.dataset}_iter{index_iter}'
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    torch.save(model.state_dict(), pt_path)

    # test
    model.load_state_dict(torch.load(path))
    model.eval()
    tic = time.time()

    valid_acc, valid_obj, tar_v, pre_v = Functions.valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2, confusionMatrix2 = Functions.output_metric(tar_v, pre_v)
    toc = time.time()
    multi_TESTING_TIME.append(toc - tic)

    pre_u = Functions.test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

    savemat(map_path + '_matrix.mat', {'P': prediction_matrix, 'label': gt_hsi})
    Functions.classification_map(prediction_matrix, height, width, 1600, path+'.png')
    print('------Get classification maps successful-------')
    print(f"result experiment No.{index_iter + 1} :")
    print("OA: {:.4f} | AA_mean: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print("AA: ")
    print(AA2)
    print("**************************************************")
    print("Parameter:")
    multi_OA.append(OA2)
    multi_AA.append(AA2)
    multi_KAPPA.append(Kappa2)
    multi_conf_matrix.append(confusionMatrix)

print("---------------------------------------------")
print("---------------------------------------------")
print("---------------Process finished---------------------")

record_path = f"records/a{args.VALIDATION_SPLIT}test_patch{args.patch}_" \
              f"hpfpatch{args.hpf_patch}_encoder{args.encoder_num}_pN{args.pixelNum}_nPG{args.n_perGroup}" \
              f"_gPS{args.groupPoolScale}_seed{args.seed}{args.dataset}.txt"

Functions.record_output(str(vars(args)), multi_OA, multi_AA, multi_KAPPA, multi_conf_matrix,
                        multi_TRAINING_TIME, multi_TESTING_TIME, record_path)
AC_list = np.array(multi_AA)
for ii in range(num_classes):
    print("{:.4f} ± {:.4f}".format(np.mean(AC_list[:, ii]), np.std(AC_list[:, ii])))
print()
print("{:.4f} ± {:.4f}".format(np.mean(multi_OA), np.std(multi_OA)))
print("{:.4f} ± {:.4f}".format(np.mean(multi_AA), np.std(multi_AA)))
print("{:.4f} ± {:.4f}".format(np.mean(multi_KAPPA), np.std(multi_KAPPA)))
print()
print(f"{np.mean(multi_TRAINING_TIME):.2f}/{np.mean(multi_TESTING_TIME):.2f}")
print(f"{np.mean(multi_PARAMETER):.0f}")
print("-----------------All Over---------------------")
