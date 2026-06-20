import argparse
import sys
import os
import time
import numpy as np
import torch.nn.functional as F
import torch
import torchsummary
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import RawNet1, RawNet2
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from torchsummary import summary
from torchstat import stat
from thop import profile
from torch.optim.lr_scheduler import StepLR

CUDA_LAUNCH_BLOCKING = 1  #


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda
def evaluate_accuracy(dev_loader, model, criterion_cls, device, args):
    num_correct = 0.0
    running_loss = 0  #
    num_total = 0.0
    ii = 0  #
    model.eval()
    # set objective (Loss) functions
    with torch.no_grad():  #
        for batch_x, batch_y in dev_loader:
            # 0，1
            batch_size = batch_x.size(0)  # [32,64600],32
            num_total += batch_size
            ii += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)  # 0,1
            _, batch_out = model(batch_x, batch_y)
            batch_loss = criterion_cls(batch_out, batch_y)
            _, batch_pred = batch_out.max(dim=1)  #
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            running_loss += (batch_loss.item() * batch_size)  #
            if ii % 10 == 0:  # 10
                sys.stdout.write('\r \t {:.2f}'.format(
                    (num_correct / num_total) * 100))
        running_loss /= num_total  #
        dev_accuracy = (num_correct / num_total) * 100  #

        # return 100 * (num_correct / num_total)
    return dev_accuracy, running_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()

    for batch_x, utt_id, batch_y in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)  # 0,1
        _, batch_out = model(batch_x, batch_y)
        batch_score = (batch_out[:, 1]
        ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, module_list, learning_rate, device, epoch, wd, criterion_list, trainable_list, args):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    index = 0
    # ---
    for module in module_list:
        module.train()

    # set teacher model as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]  # 0

    model_s = module_list[0]
    model_t = module_list[-1]

    # ---------------------------adam---------------------------------
    adam = torch.optim.Adam(trainable_list.parameters(), lr=learning_rate, weight_decay=wd)  #
    # ----------------------------------------------------------------

    for batch_x, batch_y in train_loader:  #
        # progress = float(epoch * len(train_loader) + index) / (200 * len(train_loader))
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        # print(torch.cuda.is_available())
        # exit(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # batch_out = model(batch_x, batch_y, progress)

        # ==================================================================
        feat_s, logit_s = model_s(batch_x, batch_y)
        with torch.no_grad():  #
            feat_t, logit_t = model_t(batch_x, batch_y)
        # cls + kl div
        loss_cls = criterion_cls(logit_s, batch_y)
        loss_div = criterion_div(logit_s, logit_t)
        # -----------hint--------------------
        f_s = module_list[1](feat_s[args.hint_layer])
        f_t = feat_t[args.hint_layer]
        loss_kd = criterion_kd(f_s, f_t)
        # ----------------------------------
        loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd
        # ==================================================================
        # batch_out = model(batch_x, batch_y)

        # batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = logit_s.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        adam.zero_grad()
        loss.backward()
        adam.step()

    print("\n%d learning_rate：%f" % (epoch, adam.param_groups[0]['lr']))
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def load_teacher(model_path):
    print('==> loading teacher model')
    model = RawNet1(parser1['model'], device).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print('==> done')
    return model


# ---------------KD---------------
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T  # 4

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()  #

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""  # teacher model and student model
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H = s_shape  # 3D
        t_N, t_C, t_H = t_shape  #
        if s_H == 2 * t_H:
            self.conv = nn.Conv1d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose1d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv1d(s_C, t_C, kernel_size=1+s_H-t_H)
        elif s_H < t_H:  #
            self.conv = nn.Sequential(
                nn.Upsample(size=t_H, mode='nearest'),
                nn.Conv1d(s_C, t_C, kernel_size=3, padding=1)
            )
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm1d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='D:/Task1/ASVspoof2019/PA/',
                        help='Change this to user\'s full directory address of PA database '
                             '(ASVspoof2019- for training & validation, ASVspoof2021 for evaluation scores). '
                             'We assume that all three ASVspoof 2019 PA train, PA dev and ASVspoof2021 PA eval data '
                             'folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- PA
    %      |- ASVspoof2021_PA_eval/flac
    %      |- ASVspoof2019_PA_train/flac
    %      |- ASVspoof2019_PA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='D:/Task1/ASVspoof2019/PA/',
                        help='Change with path to user\'s PA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_PA_cm_protocols
    %      |- ASVspoof2021.PA.cm.eval.trl.txt
    %      |- ASVspoof2019.PA.cm.dev.trl.txt
    %      |- ASVspoof2019.PA.cm.train.trn.txt
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default='', help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='PA', choices=['LA', 'PA', 'DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')
    # teacher model
    parser.add_argument('--path_t', type=str, default='D:/Task1/rawnet/git/PA/Baseline-RawNet2'
                                                      '/ResNeXt50_mel_junfen_log/model_PA_CCE_200_32_0.0002/epoch_151.pth',
                        help='teacher model snapshot')
    # model_s:D:/Task1/end_to_end/end_to_end_new/PA/Baseline-RawNet2/all_models/ResNeXt-50_knowledge_6resblocks_hint/model_PA_CCE_200_32_0.0002/epoch_145.pth
    # student model
    # parser.add_argument('--model_s', type=str, default='resnet34')

    # distillation
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for other losses')
    # Exact Loss
    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4, 5])

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')  # softmax

    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    if not os.path.exists('best_model_Random_Seeds_42'):
        os.mkdir('best_model_Random_Seeds_42')
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    # Database
    prefix2019_ = 'ASVspoof2019_{}'.format(track)  # PA
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    # prefix_2021 = 'ASVspoof2021.{}'.format(track)

    # define model saving path # print log
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('best_model_Random_Seeds_42', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model_t = load_teacher(args.path_t)
    model_s = RawNet2(parser1['model'], device).cuda()  # load student model
    # ------------------------------------------------------------------
    data = torch.randn(32, 8192).to(device)
    data_label = torch.randint(0, 2, (32,)).to(device)
    model_t.eval()
    model_s.eval()
    # ------------------------------
    feat_t, _ = model_t(data, data_label)
    feat_s, _ = model_s(data, data_label)
    # -----------------------------------dataloader-------------------------------
    # define train dataloader
    # ------------------------------------eval-------------------------------------
    if args.model_path:
        model_s.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print('Model loaded : {}'.format(args.model_path))
    # -----------------------ASVspoof2019PA---------------------
    if args.eval:
        d_label_eva, file_eval = genSpoof_list(dir_meta=os.path.join(
            args.protocols_path + '{}_asv_protocols/{}.asv.eval.gi.trl.txt'.format(prefix2019_, prefix_2019)),
                                               is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, labels=d_label_eva, base_dir=os.path.join(
            args.database_path + 'ASVspoof2019_{}_eval/'.format(args.track)))  # 153522个
        produce_evaluation_file(eval_set, model_s, device, args.eval_output)
        sys.exit(0)
    # ------------------train----------------------
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(
        args.protocols_path + '{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix2019_, prefix_2019)), is_train=True,
        is_eval=False)
    print('no. of training trials', len(file_train))  # 54000
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(
        args.database_path + 'ASVspoof2019_{}_train/'.format(args.track)))
    # value、label
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    del train_set, d_label_trn

    # define validation dataloader
    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(
        args.protocols_path + '{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix2019_, prefix_2019)), is_train=False,
        is_eval=False)
    print('no. of validation trials', len(file_dev))
    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev,
                                         labels=d_label_dev,
                                         base_dir=os.path.join(
                                             args.database_path + 'ASVspoof2019_{}_dev/'.format(args.track)))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set, d_label_dev

    # ------------------------------------------------------------------------------------------------------------------
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion_cls = nn.CrossEntropyLoss(weight=weight)
    # KD
    criterion_div = DistillKL(args.kd_T)

    # hints
    criterion_kd = HintLoss()  #
    regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape).to(device)
    module_list.append(regress_s)  # model list(teacher model + student model)
    trainable_list.append(regress_s)  # Adam

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    module_list.append(model_t)
    # validate teacher acc
    valid_accuracy, _ = evaluate_accuracy(dev_loader, model_t, criterion_cls, device, args)
    print('\nteacher acc:', valid_accuracy)
    # ------------------------------------------------------------------------------------------------------------------

    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs_best_model_Random_Seeds_42/{}'.format(model_tag))
    best_acc = 0
    num = []
    loss_all = []
    epoch_all = []
    dev_loss_all = []
    learning_rate = args.lr  # 0.0002
    wd = args.weight_decay  # 0.0001
    print('init learning_rate:', learning_rate)
    for epoch in range(num_epochs):
        time1 = time.time()
        if (epoch > 0) and (epoch % 5 == 0):
            learning_rate = learning_rate * 0.9  # 0.9 decay
        running_loss, train_accuracy = train_epoch(train_loader, module_list, learning_rate, device, epoch, wd,
                                                   criterion_list, trainable_list,  args)
        valid_accuracy, dev_loss = evaluate_accuracy(dev_loader, model_s, criterion_cls, device, args)
        time2 = time.time()
        print('\nepoch {}, total time {:.2f}second'.format(epoch, time2 - time1))
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('dev_loss', dev_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f} - {}'.format(epoch,
                                                        running_loss, train_accuracy, valid_accuracy,
                                                        dev_loss))
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
            num.append(epoch)
        best_acc = max(valid_accuracy, best_acc)
        torch.save(model_s.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        print("it hased:", epoch)
        loss_all.append(running_loss)
        epoch_all.append(epoch)
        dev_loss_all.append(dev_loss)
    print("Good model:")
    for i in num:
        print(i)
    # print('paras_num:', nb_params)
    plt.plot(epoch_all, loss_all, color="green", label="train_loss")
    plt.plot(epoch_all, dev_loss_all, color='blue', label='dev_loss')
    plt.xlabel("epoch")
    plt.ylabel("loss_or_errors")
    plt.title("train_loss_or_errors on epoch")
    plt.legend()
    plt.xticks(epoch_all[::10])
    plt.show()
    for epoch in range(num_epochs):
        if (epoch > 0) and (epoch % 5 == 0):
            learning_rate = learning_rate * 0.9