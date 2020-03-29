import argparse
import sys

def parseTrainingArgs():

    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    # Arguments for training with focal stacks
    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('data', metavar='DIR', help='path to dataset')
    focalstack_args.add_argument('name', metavar='NAME', help='experiment name')
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    focalstack_args.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/", help='where to save outputs')
    focalstack_args.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    focalstack_args.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : [euler, quat]')
    focalstack_args.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
    focalstack_args.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    focalstack_args.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    focalstack_args.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    focalstack_args.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
    focalstack_args.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    focalstack_args.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    focalstack_args.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
    focalstack_args.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
    focalstack_args.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
    focalstack_args.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
    focalstack_args.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    focalstack_args.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
    focalstack_args.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
    focalstack_args.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
    focalstack_args.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
    focalstack_args.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
    focalstack_args.add_argument('-g', '--gt-pose-loss-weight', type=float, help='weight for ground truth pose supervision loss', metavar='W', default=0)
    focalstack_args.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
    focalstack_args.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output', metavar='N', default=0)
    focalstack_args.add_argument('--gray', action='store_true', help="images are grayscale")

    # Arguments for training with colour channel stacks
    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('data', metavar='DIR', help='path to dataset')
    stack_args.add_argument('name', metavar='NAME', help='experiment name')
    stack_args.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/", help='where to save outputs')
    stack_args.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    stack_args.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : [euler, quat]')
    stack_args.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
    stack_args.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    stack_args.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    stack_args.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    stack_args.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
    stack_args.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    stack_args.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    stack_args.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
    stack_args.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
    stack_args.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
    stack_args.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
    stack_args.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    stack_args.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
    stack_args.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
    stack_args.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
    stack_args.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
    stack_args.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
    stack_args.add_argument('-g', '--gt-pose-loss-weight', type=float, help='weight for ground truth pose supervision loss', metavar='W', default=0)
    stack_args.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
    stack_args.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output', metavar='N', default=0)
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    stack_args.add_argument('--gray', action='store_true', help="images are grayscale")

    args = parser.parse_args()
    return args


def parseMultiwarpTrainingArgs():

    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    # Arguments for training with focal stacks
    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('data', metavar='DIR', help='path to dataset')
    focalstack_args.add_argument('name', metavar='NAME', help='experiment name')
    focalstack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    focalstack_args.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/", help='where to save outputs')
    focalstack_args.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    focalstack_args.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : [euler, quat]')
    focalstack_args.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
    focalstack_args.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    focalstack_args.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    focalstack_args.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    focalstack_args.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
    focalstack_args.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    focalstack_args.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    focalstack_args.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
    focalstack_args.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
    focalstack_args.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
    focalstack_args.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
    focalstack_args.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    focalstack_args.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
    focalstack_args.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
    focalstack_args.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
    focalstack_args.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
    focalstack_args.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
    focalstack_args.add_argument('-g', '--gt-pose-loss-weight', type=float, help='weight for ground truth pose supervision loss', metavar='W', default=0)
    focalstack_args.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
    focalstack_args.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output', metavar='N', default=0)
    focalstack_args.add_argument('--gray', action='store_true', help="images are grayscale")

    # Arguments for training with colour channel stacks
    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('data', metavar='DIR', help='path to dataset')
    stack_args.add_argument('name', metavar='NAME', help='experiment name')
    stack_args.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/", help='where to save outputs')
    stack_args.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    stack_args.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : [euler, quat]')
    stack_args.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
    stack_args.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    stack_args.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    stack_args.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    stack_args.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
    stack_args.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    stack_args.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    stack_args.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
    stack_args.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
    stack_args.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
    stack_args.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
    stack_args.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    stack_args.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
    stack_args.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
    stack_args.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
    stack_args.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
    stack_args.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
    stack_args.add_argument('-g', '--gt-pose-loss-weight', type=float, help='weight for ground truth pose supervision loss', metavar='W', default=0)
    stack_args.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
    stack_args.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output', metavar='N', default=0)
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    stack_args.add_argument('--gray', action='store_true', help="images are grayscale")

    epi_args = subparsers.add_parser('epi', help="Train using epipolar plane images")
    epi_args.add_argument('data', metavar='DIR', help='path to dataset')
    epi_args.add_argument('name', metavar='NAME', help='experiment name')
    epi_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    epi_args.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/",
                                 help='where to save outputs')
    epi_args.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training',
                                 default=3)
    epi_args.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                                 help='rotation mode for PoseExpnet : [euler, quat]')
    epi_args.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                                 help='padding mode for image warping')
    epi_args.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers')
    epi_args.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    epi_args.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    epi_args.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR',
                                 help='initial learning rate')
    epi_args.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                 help='momentum for sgd, alpha parameter for adam')
    epi_args.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    epi_args.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
    epi_args.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
    epi_args.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                                 help='path to pre-trained dispnet model')
    epi_args.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                                 help='path to pre-trained Exp Pose net model')
    epi_args.add_argument('--seed', default=0, type=int,
                                 help='seed for random functions, and network initialization')
    epi_args.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                                 help='csv where to save per-epoch train and valid stats')
    epi_args.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                                 help='csv where to save per-gradient descent train stats')
    epi_args.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss',
                                 metavar='W', default=1)
    epi_args.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss',
                                 metavar='W', default=0)
    epi_args.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss',
                                 metavar='W', default=0.1)
    epi_args.add_argument('-g', '--gt-pose-loss-weight', type=float,
                                 help='weight for ground truth pose supervision loss', metavar='W', default=0)
    epi_args.add_argument('--log-output', action='store_true',
                                 help='will log dispnet outputs and warped imgs at validation step')
    epi_args.add_argument('-f', '--training-output-freq', type=int,
                                 help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                                 metavar='N', default=0)
    epi_args.add_argument('--gray', action='store_true', help="images are grayscale")

    args = parser.parse_args()
    return args
    