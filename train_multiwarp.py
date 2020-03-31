from multiwarp_dataloader import getFocalstackLoaders, getStackedLFLoaders, getEpiLoaders
from parser import parseMultiwarpTrainingArgs
import time
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import lfmodels as models

from utils import tensor2array, save_checkpoint, make_save_path, dump_config
from loss_functions import multiwarp_photometric_loss, explainability_loss, smooth_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parseMultiwarpTrainingArgs()
    args.training_output_freq = 100
    save_path = make_save_path(args)
    args.save_path = save_path
    dump_config(save_path, args)
    print('\n\n=> Saving checkpoints to {}'.format(save_path))
    torch.manual_seed(args.seed)
    tb_writer = SummaryWriter(save_path)

    # Data pre-processing
    train_transform = valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create data loader
    print("=> Fetching scenes in '{}'".format(args.data))
    train_set, val_set = None, None
    if args.lfformat == 'focalstack':
        train_set, val_set = getFocalstackLoaders(args, train_transform, valid_transform)
    elif args.lfformat == 'stack':
        train_set, val_set = getStackedLFLoaders(args, train_transform, valid_transform)
    elif args.lfformat == 'epi':
        train_set, val_set = getEpiLoaders(args, train_transform, valid_transform)

    print('=> {} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('=> {} samples found in {} validation scenes'.format(len(val_set), len(val_set.scenes)))

    print('=> Multi-warp training, warping {} sub-apertures'.format(len(args.cameras)))

    # Create batch loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # Pull first example from dataset to check number of channels
    input_channels = train_set[0]['tgt_lf_formatted'].shape[0]
    output_channels = len(args.cameras)
    args.epoch_size = len(train_loader)
    print("=> [DispNet] Using {} input channels, {} output channels".format(input_channels, output_channels))
    print("=> [PoseNet] Using {} input channels".format(input_channels))

    # create model
    print("=> Creating models")

    if args.lfformat == "epi":
        print("=> Using epipolar plane image dispnet")
        disp_net = models.EPIDispNet(input_channels).to(device)
    else:
        disp_net = models.LFDispNet(in_channels=input_channels, out_channels=output_channels).to(device)

    pose_exp_net = models.LFPoseNet(in_channels=input_channels, nb_ref_imgs=args.sequence_length - 1,
                                    output_exp=args.mask_loss_weight > 0).to(device)

    if args.pretrained_exp_pose:
        print("=> [PoseNet] Using pre-trained weights for pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        print("=> [PoseNet] training from scratch")
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> [DispNet] Using pre-trained weights for DispNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        print("=> [DispNet] training from scratch")
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> Setting adam solver')

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    with open(save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(save_path / args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, logger, tb_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, tb_writer)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(save_path, {
            'epoch': epoch + 1,
            'state_dict': disp_net.module.state_dict()
        }, {
                            'epoch': epoch + 1,
                            'state_dict': pose_exp_net.module.state_dict()
                        }, is_best)

        with open(save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, logger, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3, w4 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.gt_pose_loss_weight

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, trainingdata in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_lf = trainingdata['tgt_lf'].to(device)
        ref_lfs = [img.to(device) for img in trainingdata['ref_lfs']]
        tgt_lf_formatted = trainingdata['tgt_lf_formatted'].to(device)
        ref_lfs_formatted = [lf.to(device) for lf in trainingdata['ref_lfs_formatted']]
        intrinsics = trainingdata['intrinsics'].to(device)
        pose_gt = trainingdata['pose_gt'].to(device)
        metadata = trainingdata['metadata']

        # compute output
        disparities = disp_net(tgt_lf_formatted)
        depth = [1 / disp for disp in disparities]

        explainability_mask, pose = pose_exp_net(tgt_lf_formatted, ref_lfs_formatted)

        loss_1, warped, diff = multiwarp_photometric_loss(
            tgt_lf, ref_lfs, intrinsics, depth, pose, metadata, args.rotation_mode, args.padding_mode
        )

        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth)

        pred_pose_magnitude = pose[:, :, :3].norm(dim=2)
        pose_gt_magnitude = pose_gt[:, :, :3].norm(dim=2)
        pose_loss = (pred_pose_magnitude - pose_gt_magnitude).abs().mean()

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * pose_loss

        if log_losses:
            tb_writer.add_scalar('train/photometric_error', loss_1.item(), n_iter)
            tb_writer.add_scalar('train/smoothness_loss', loss_3.item(), n_iter)
            tb_writer.add_scalar('train/total_loss', loss.item(), n_iter)
            tb_writer.add_scalar('train/pose_loss', pose_loss.item(), n_iter)
        if log_output:
            b, n, h, w = tgt_lf_formatted.shape
            vis_img = tgt_lf_formatted[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5
            vis_depth = tensor2array(depth[0][0, 0, :, :], colormap='magma')
            vis_disp = tensor2array(disparities[0][0, 0, :, :], colormap='magma')
            tb_writer.add_image('train/input', vis_img, n_iter)
            tb_writer.add_image('train/depth', vis_depth, n_iter)
            tb_writer.add_image('train/disp', vis_disp, n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path / args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, tb_writer, sample_nb_to_log=2):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = sample_nb_to_log > 0
    w1, w2, w3, w4 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.gt_pose_loss_weight
    # poses = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1),6))
    # disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, validdata in enumerate(val_loader):
        tgt_lf = validdata['tgt_lf'].to(device)
        ref_lfs = [ref.to(device) for ref in validdata['ref_lfs']]
        tgt_lf_formatted = validdata['tgt_lf_formatted'].to(device)
        ref_lfs_formatted = [lf.to(device) for lf in validdata['ref_lfs_formatted']]
        intrinsics = validdata['intrinsics'].to(device)
        pose_gt = validdata['pose_gt'].to(device)
        metadata = validdata['metadata']

        # compute output
        disp = disp_net(tgt_lf_formatted)
        depth = 1 / disp
        explainability_mask, pose = pose_exp_net(tgt_lf_formatted, ref_lfs_formatted)

        loss_1, warped, diff = multiwarp_photometric_loss(
            tgt_lf, ref_lfs, intrinsics, depth, pose, metadata, args.rotation_mode, args.padding_mode
        )

        loss_1 = loss_1.item()
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).item()
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth).item()

        pred_pose_magnitude = pose[:, :, :3].norm(dim=2)
        pose_gt_magnitude = pose_gt[:, :, :3].norm(dim=2)
        pose_loss = (pred_pose_magnitude - pose_gt_magnitude).abs().mean()

        if log_outputs and i < sample_nb_to_log - 1:  # log first output of first batches
            b, n, h, w = tgt_lf_formatted.shape
            vis_img = tgt_lf_formatted[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5
            vis_depth = tensor2array(depth[0, 0, :, :], colormap='magma')
            vis_disp = tensor2array(disp[0, 0, :, :], colormap='magma')

            tb_writer.add_image('val/target_image', vis_img, n_iter)
            tb_writer.add_image('val/disp', vis_disp, n_iter)
            tb_writer.add_image('val/depth', vis_depth, n_iter)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * pose_loss
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['val/total_loss', 'val/photometric_error', 'val/explainability_loss']


if __name__ == '__main__':
    main()
