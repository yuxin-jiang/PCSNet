import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
from casnet import Model as casnet
from cnn.resnet import wide_resnet50_2 as wrn50_2
import datasets.mvtec_test as mvtec
from datasets.mvtec_test import MVTecDataset
from torchvision import transforms as T
from utils.adaptor import *
from utils.metric import *
from utils.visualizer import *
import torch.optim as optim
import warnings
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
import math
from datasets.mvtec_train import SelfSupMVTecDataset, OBJECTS

WIDTH_BOUNDS_PCT = {
    'with_defect': ((0.01, 0.05), (0.01, 0.05)),
    '01': ((0.01, 0.05), (0.01, 0.05)),
    '02': ((0.01, 0.05), (0.01, 0.05)),
    '03': ((0.01, 0.05), (0.01, 0.05)),
    'bracket_black': ((0.01, 0.05), (0.01, 0.05)),
    'bracket_brown': ((0.01, 0.05), (0.01, 0.05)),
    'bracket_white': ((0.03, 0.4), (0.03, 0.4)),
    'connector': ((0.03, 0.4), (0.03, 0.4)),
    'metal_plate': ((0.03, 0.4), (0.03, 0.4)),
    'tubes': ((0.03, 0.4), (0.03, 0.4)),
    'bagel': ((0.03, 0.4), (0.03, 0.4)),
    'cable_gland': ((0.03, 0.4), (0.03, 0.4)),
    'carrot': ((0.03, 0.4), (0.03, 0.4)),
    'cookie': ((0.03, 0.4), (0.03, 0.4)),
    'dowel': ((0.03, 0.4), (0.03, 0.4)),
    'foam': ((0.03, 0.4), (0.03, 0.4)),
    'peach': ((0.03, 0.4), (0.03, 0.4)),
    'potato': ((0.01, 0.05), (0.01, 0.05)),
    'rope': ((0.03, 0.4), (0.03, 0.4)),
    'tire': ((0.03, 0.4), (0.03, 0.4)),
    'bottle': ((0.03, 0.4), (0.03, 0.4)),
    'MT_Blowhole': ((0.03, 0.4), (0.03, 0.4)),
    'cable': ((0.01, 0.05), (0.01, 0.05)),
    'capsule': ((0.01, 0.05), (0.01, 0.05)),
    'hazelnut': ((0.01, 0.05), (0.01, 0.1)),
    'metal_nut': ((0.05, 0.4), (0.05, 0.4)),
    'pill': ((0.01, 0.05), (0.01, 0.1)),
    'screw': ((0.01, 0.05), (0.01, 0.05)),
    'toothbrush': ((0.01, 0.05), (0.01, 0.05)),
    'transistor': ((0.03, 0.4), (0.03, 0.4)),
    'zipper': ((0.03, 0.4), (0.03, 0.4)),
    'carpet': ((0.03, 0.4), (0.03, 0.4)),
    'grid': ((0.01, 0.05), (0.01, 0.1)),
    'leather': ((0.01, 0.05), (0.01, 0.05)),
    'tile': ((0.01, 0.05), (0.01, 0.1)),
    'wood': ((0.01, 0.05), (0.01, 0.05))
}
MIN_OVERLAP_PCT = {
    'with_defect': 0.25,
    '01': 0.25,
    '02': 0.25,
    '03': 0.25,
    'bracket_black': 0.25,
    'bracket_brown': 0.25,
    'bracket_white': 0.25,
    'connector': 0.25,
    'metal_plate': 0.25,
    'tubes': 0.25,
    'bagel': 0.25,
    'cable_gland': 0.25,
    'carrot': 0.25,
    'cookie': 0.25,
    'dowel': 0.25,
    'foam': 0.25,
    'peach': 0.25,
    'potato': 0.25,
    'rope': 0.25,
    'tire': 0.25,
    'bottle': 0.25,
    'MT_Blowhole': 0.25,
    'capsule': 0.25,
    'hazelnut': 0.25,
    'metal_nut': 0.25,
    'pill': 0.25,
    'screw': 0.25,
    'toothbrush': 0.25,
    'zipper': 0.25}

MIN_OBJECT_PCT = {
    'with_defect': 0.7,
    '01': 0.7,
    '02': 0.7,
    '03': 0.7,
    'bracket_black': 0.7,
    'bracket_brown': 0.7,
    'bracket_white': 0.7,
    'connector': 0.7,
    'metal_plate': 0.7,
    'tubes': 0.7,
    'cable_gland': 0.7,
    'carrot': 0.7,
    'cookie': 0.7,
    'dowel': 0.7,
    'foam': 0.7,
    'peach': 0.7,
    'potato': 0.7,
    'rope': 0.7,
    'tire': 0.7,
    'bottle': 0.7,
    'MT_Blowhole': 0.7,
    'capsule': 0.7,
    'hazelnut': 0.7,
    'metal_nut': 0.5,
    'pill': 0.7,
    'screw': .5,
    'toothbrush': 0.25,
    'zipper': 0.7}

NUM_PATCHES = {
    'with_defect': 1,
    '01': 3,
    '02': 3,
    '03': 1,
    'bracket_black': 1,
    'bracket_brown': 3,
    'bracket_white': 1,
    'connector': 1,
    'metal_plate': 1,
    'tubes': 1,
    'bagel': 1,
    'cable_gland': 1,
    'carrot': 1,
    'cookie': 1,
    'dowel': 1,
    'foam': 1,
    'peach': 1,
    'potato': 3,
    'rope': 1,
    'tire': 1,
    'MT_Blowhole': 1,
    'bottle': 1,
    'cable': 3,
    'capsule': 1,
    'hazelnut': 3,
    'metal_nut': 1,
    'pill': 2,
    'screw': 1,
    'toothbrush': 3,
    'transistor': 3,
    'zipper': 3,
    'carpet': 2,
    'grid': 3,
    'leather': 3,
    'tile': 1,
    'wood': 3}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {
    'with_defect': (1 / 12, 24),
    '01': (1 / 12, 24),
    '02': (1 / 12, 24),
    '03': (1 / 12, 24),
    'bracket_black': (1 / 12, 24),
    'bracket_brown': (1 / 12, 24),
    'bracket_white': (1 / 12, 24),
    'connector': (1 / 12, 24),
    'metal_plate': (1 / 12, 24),
    'tubes': (1 / 12, 24),
    'bagel': (1 / 12, 24),
    'cable_gland': (1 / 12, 24),
    'carrot': (1 / 12, 24),
    'cookie': (1 / 12, 24),
    'dowel': (1 / 12, 24),
    'foam': (1 / 12, 24),
    'peach': (1 / 12, 24),
    'potato': (1 / 12, 24),
    'rope': (1 / 12, 24),
    'tire': (1 / 12, 24),
    'MT_Blowhole': (1 / 12, 24),
    'bottle': (1 / 12, 24),
    'cable': (1 / 12, 24),
    'capsule': (1 / 2, 4),'hazelnut': (1 / 12, 24), 'metal_nut': (1 / 3, 7),
    'pill': (1 / 3, 7), 'screw': (1, 3), 'toothbrush': (1 / 6, 15),
    'transistor': (1 / 6, 15), 'zipper': (1 / 6, 15),
    'carpet': (1 / 3, 7), 'grid': (1 / 3, 7), 'leather': (1 / 3, 7), 'tile': (1 / 3, 7),
    'wood': (1 / 6, 15)}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']


# brightness, threshold pairs
BACKGROUND = {
    'bottle': (200, 60), 'screw': (200, 60),
    'capsule': (200, 60), 'zipper': (200, 60),
    'hazelnut': (20, 20), 'pill': (20, 20), 'toothbrush': (20, 20), 'metal_nut': (20, 20)}

#####################################
def parse_args():
    parser = argparse.ArgumentParser('CFA configuration')
    parser.add_argument('--data_path', type=str, default='./datasets/mvtec')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument("-s", "--setting", type=str, default="Shift-Intensity-923874273")
    parser.add_argument('--Rd', type=bool, default=False)
    parser.add_argument('--size', type=int, choices=[224, 256], default=224)
    parser.add_argument('--gamma_c', type=int, default=1)
    parser.add_argument('--class_name', type=str, default='toothbrush')

    return parser.parse_args()

def weight_init(m):
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

def run():
    seed = 512
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    args = parse_args()
    class_names = mvtec.CLASS_NAMES if args.class_name == 'all' else [args.class_name]

    total_roc_auc = []
    total_pixel_roc_auc = []
    total_pixel_pro_auc = []

    indexs = 1
    for index_class, class_name in enumerate(class_names):
        auroc_px_list = []
        auroc_sp_list = []
        aupro_px_list = []

        for Index in range(indexs):
            best_img_roc = -1
            best_pxl_roc = -1
            best_pxl_pro = -1
            print(' ')
            print('%s | newly initialized...' % class_name)
            BACKGROUND = {'bracket_white': (130, 60), 'bottle': (200, 60), 'screw': (200, 60), 'capsule': (200, 60),
                          'zipper': (200, 60),
                          'hazelnut': (20, 20), 'pill': (20, 20), 'toothbrush': (20, 20), 'metal_nut': (20, 20)}

            # load data
            if class_name in UNALIGNED_OBJECTS:
                train_transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(256),
                ])
            elif class_name in OBJECTS:
                train_transform = T.Compose([T.Resize(256),
                                             T.CenterCrop(256)
                                             ])
            else:  # texture
                train_transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(256),
                ])

            train_dat = SelfSupMVTecDataset(root_path=args.data_path, class_name=class_name, is_train=True,
                                           transform=train_transform)

            train_dat.configure_self_sup(self_sup_args={'gamma_params': (2, 0.05, 0.03), 'resize': False,
                                                        'shift': True, 'same': True, 'mode': 'swap',
                                                        'label_mode': 'binary'})
            train_dat.configure_self_sup(self_sup_args={'skip_background': BACKGROUND.get(class_name)})
            train_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
                                                                 'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(
                                                                     class_name),
                                                                 'num_patches': NUM_PATCHES.get(class_name),
                                                                 'min_object_pct': MIN_OBJECT_PCT.get(class_name),
                                                                 'min_overlap_pct': MIN_OVERLAP_PCT.get(class_name)})

            test_dataset = MVTecDataset(dataset_path=args.data_path,
                                        class_name=class_name,
                                        resize=256,
                                        cropsize=args.size,
                                        is_train=False,
                                        wild_ver=args.Rd)

            train_loader = DataLoader(dataset=train_dat,
                                      batch_size=2,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True, )

            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     pin_memory=True,
                                     drop_last=True)

            model = wrn50_2(pretrained=True, progress=False)

            model = model.to(device)
            A = adaptor(model, train_loader, args.gamma_c, device, class_name).to(device)
            A.apply(weight_init)
            CAS = casnet().to(device)
            CAS.apply(weight_init)
            epochs = 50
            optimizer = optim.Adam([
                {'params': A.parameters(), 'lr': 0.001},
                {'params': CAS.parameters(), 'lr': 0.0001}],
                weight_decay=1e-5,
                amsgrad=True)

            for epoch in tqdm(range(epochs), '%s -->' % (class_name)):
                r'TEST PHASE'
                A.train()
                CAS.train()
                model.eval()
                MSE_loss = torch.nn.MSELoss().to(device)
                tr_entropy_loss_func = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
                for (x, aug_img, _, _, mask0, mask1) in train_loader:
                    if x.shape[0] % 2 == 1:
                        a = x.shape[0] / 2 + 1
                    else:
                        a = x.shape[0] / 2
                    res = random.sample(range(0, x.shape[0]), int(a))
                    mix_img_list = x.clone()
                    mix_img_list[res] = aug_img[res]
                    mix_mask0_list = torch.zeros_like(mask0)
                    mix_mask0_list[res] = mask0[res]
                    target = []
                    target.extend([0] * x.shape[0])
                    target = torch.tensor(target)
                    target[res] = int(1)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        normal_ori_feature = model(x.to(device))  ## normal
                        aug_ori_feature = model(aug_img.to(device))  ## abnormal
                        mix_ori_feature = model(mix_img_list.to(device))  ### Some parts are normal, some parts are abnormal.
                    L_NFC, _, normal_score, normal_feature = A(normal_ori_feature, 0, mask1.to(device))
                    L_AFS, _, aug_score, aug_feature = A(aug_ori_feature, 1, mask1.to(device))
                    _, score_1, mix_score, mix_feature = A(mix_ori_feature, 2, mask1.to(device))
                    out = CAS(mix_feature, mix_score)
                    L_SEG = MSE_loss(out.to(device), mix_mask0_list.to(device))
                    L_PDC = tr_entropy_loss_func(score_1.squeeze(), target.float().to(device))
                    loss = L_NFC + L_AFS + L_PDC * 50 + L_SEG * 40
                    loss.backward()
                    optimizer.step()
                print('loss: {:.4f}'.format(loss.item()))

                if epoch % 10 == 0:
                    test_imgs = list()
                    gt_mask_list = list()
                    gt_list = list()
                    heatmaps = None
                    A.eval()
                    CAS.eval()
                    model.eval()
                    for x, y, mask in tqdm(test_loader):

                        test_imgs.extend(x.cpu().detach().numpy())
                        gt_list.extend(y.cpu().detach().numpy())
                        mask = torch.where(mask < 0.5, 0, 1)
                        gt_mask_list.extend(mask.cpu().detach().numpy())
                        with torch.no_grad():
                            ori_feature = model(x.to(device))
                            score, feature = A(ori_feature, 2, None)
                            score = CAS(feature, score)
                        heatmap = score.cpu().detach()
                        heatmap = torch.mean(heatmap, dim=1)
                        heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
                    heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
                    heatmaps = gaussian_smooth(heatmaps, sigma=4)

                    gt_mask = np.asarray(gt_mask_list)
                    scores = rescale(heatmaps)
                    scores = scores
                    threshold = get_threshold(gt_mask, scores)

                    r'Image-level AUROC'
                    fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
                    best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc

                    r'Pixel-level AUROC'
                    fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask, scores)
                    best_pxl_roc = per_pixel_rocauc if per_pixel_rocauc > best_pxl_roc else best_pxl_roc

                    r'Pixel-level AUPRO'
                    per_pixel_proauc = cal_pxl_pro(gt_mask, scores)
                    best_pxl_pro = per_pixel_proauc if per_pixel_proauc > best_pxl_pro else best_pxl_pro

                    print('[%d / %d]image ROCAUC: %.3f | best: %.3f' % (epoch, epochs, img_roc_auc, best_img_roc))
                    print('[%d / %d]pixel ROCAUC: %.3f | best: %.3f' % (epoch, epochs, per_pixel_rocauc, best_pxl_roc))
                    print('[%d / %d]pixel PROAUC: %.3f | best: %.3f' % (epoch, epochs, per_pixel_proauc, best_pxl_pro))

            print('image ROCAUC: %.3f' % (best_img_roc))
            print('pixel ROCAUC: %.3f' % (best_pxl_roc))
            print('pixel ROCAUC: %.3f' % (best_pxl_pro))

            auroc_px_list.append(best_pxl_roc)

            auroc_sp_list.append(best_img_roc)

            aupro_px_list.append(best_pxl_pro)

            total_roc_auc.append(best_img_roc)
            total_pixel_roc_auc.append(best_pxl_roc)
            total_pixel_pro_auc.append(best_pxl_pro)

            # save_dir = ''
            # os.makedirs(save_dir, exist_ok=True)
            # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    print('Average pixel PROUAC: %.3f' % np.mean(total_pixel_pro_auc))

if __name__ == '__main__':
    run()
