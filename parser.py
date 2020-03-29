import argparse

def parseTrainingArgs():
    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    # Focal stack training arguments
    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    addCommonArguments(focalstack_args)

    # Image-volume training arguments
    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    addCommonArguments(stack_args)

    # Parse the args yo
    args = parser.parse_args()
    return args


def parseMultiwarpTrainingArgs():
    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    addCommonArguments(focalstack_args)

    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    addCommonArguments(stack_args)

    epi_args = subparsers.add_parser('epi', help="Train using epipolar plane images")
    epi_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    addCommonArguments(epi_args)

    args = parser.parse_args()
    return args


def addCommonArguments(subparser):
    # Metadata
    subparser.add_argument('data', metavar='DIR', help='path to dataset')
    subparser.add_argument('name', metavar='NAME', help='experiment name')
    subparser.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/",
                           help='where to save outputs')

    # Training parameters
    subparser.add_argument('--gray', action='store_true',
                           help="images are grayscale")
    subparser.add_argument('--sequence-length', type=int, metavar='N',
                           help='sequence length for training', default=3)
    subparser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                           help='rotation mode for PoseExpnet : [euler, quat]')
    subparser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                           help='padding mode for image warping')
    subparser.add_argument('--epochs', default=200, type=int, metavar='N',
                           help='number of total epochs to run')
    subparser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                           help='mini-batch size')
    subparser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR',
                           help='initial learning rate')
    subparser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                           help='momentum for sgd, alpha parameter for adam')
    subparser.add_argument('--beta', default=0.999, type=float, metavar='M',
                           help='beta parameters for adam')
    subparser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                           help='weight decay')
    subparser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                           help='path to pre-trained dispnet model')
    subparser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                           help='path to pre-trained Exp Pose net model')
    subparser.add_argument('--seed', default=0, type=int,
                           help='seed for random functions, and network initialization')
    subparser.add_argument('-p', '--photo-loss-weight', type=float,
                           help='weight for photometric loss', metavar='W', default=1)
    subparser.add_argument('-m', '--mask-loss-weight', type=float,
                           help='weight for explainabilty mask loss', metavar='W', default=0)
    subparser.add_argument('-s', '--smooth-loss-weight', type=float,
                           help='weight for disparity smoothness loss', metavar='W', default=0.1)
    subparser.add_argument('-g', '--gt-pose-loss-weight', type=float,
                           help='weight for ground truth pose supervision loss', metavar='W', default=0)

    # Other configurations
    subparser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                           help='number of data loading workers')
    subparser.add_argument('--print-freq', default=10, type=int, metavar='N',
                           help='print frequency')
    subparser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                           help='csv where to save per-epoch train and valid stats')
    subparser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                           help='csv where to save per-gradient descent train stats')
    subparser.add_argument('--log-output', action='store_true',
                           help='will log dispnet outputs and warped imgs at validation step')
    subparser.add_argument('-f', '--training-output-freq', type=int,
                           help='frequency for outputting dispnet outputs and warped imgs at training for all scales if\
                           0 will not output', metavar='N', default=0)
