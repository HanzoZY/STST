import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer
# import torchvision
import numpy as np
import time
import pickle
import cv2


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def clip_grad_norm_(parameters, max_grad):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))
    max_grad = float(max_grad)

    for name, p in parameters:
        grad = p.grad.data.abs()
        if grad.isnan().any():
            ind = grad.isnan()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.isinf().any():
            ind = grad.isinf()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.max() > max_grad:
            ind = grad>max_grad
            p.grad.data[ind] = p.grad.data[ind]/grad[ind]*max_grad  # sign x val


def train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer):
    process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    # process = tqdm(data_loader, desc='Train: ')
    for index, (inputs, labels) in enumerate(process):

        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.mix_up_num > 0:
            # self.print_log('using mixup data: ', self.arg.mix_up_num)
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
            inputs, targets = mixup(inputs, targets, np.random.beta(args.mix_up_num, args.mix_up_num))
        elif args.label_smoothing_num != 0 or args.loss == 'cross_entropy_naive':
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        # inputs, labels = Variable(inputs.cuda(non_blocking=True)), Variable(labels.cuda(non_blocking=True))

        # inputs, targets, labels = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        # net = torch.nn.DataParallel(model, device_ids=args.device_id)
        outputs, pretext_loss, PC_loss, PT_loss, PS_loss, RT_loss, CL_loss = model(inputs)
        loss = loss_function(outputs, targets)
        loss = loss + pretext_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.named_parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        if len(outputs.data.shape) == 3:  # T N cls
            _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        else:
            _, predict_label = torch.max(outputs.data, 1)
        acc_loss = loss_function(outputs, targets)
        ls = acc_loss.data.item()
        pretext_ls = pretext_loss.mean().item()
        PC_loss = PC_loss.mean().item()
        PT_loss = PT_loss.mean().item()
        PS_loss = PS_loss.mean().item()
        RT_loss = RT_loss.mean().item()
        CL_loss = CL_loss.mean().item()
        acc = torch.mean((predict_label == labels.data).float()).item()
        # ls = loss.data[0]
        # acc = torch.mean((predict_label == labels.data).float())
        lr = optimizer.param_groups[0]['lr']
        process.set_description(
            'Train: acc: {:4f}, loss: {:4f}, pretext: {:4f}, PC: {:4f}, PT: {:4f}, PS: {:4f}, RT: {:4f}, CL: {:4f}, batch time: {:4f}, lr: {:4f}'.format(
                acc, ls, pretext_ls, PC_loss, PT_loss, PS_loss, RT_loss, CL_loss,
                process.iterable.last_duration,
                lr))

        # 每个batch记录一次
        if args.mode == 'train_val':
            writer.add_scalar('acc', acc, global_step)
            writer.add_scalar('loss', ls, global_step)
            writer.add_scalar('pretext', pretext_ls, global_step)
            writer.add_scalar('PC', PC_loss, global_step)
            writer.add_scalar('PT', PT_loss, global_step)
            writer.add_scalar('PS', PS_loss, global_step)
            writer.add_scalar('RT', RT_loss, global_step)
            writer.add_scalar('CL', CL_loss, global_step)
            writer.add_scalar('batch_time', process.iterable.last_duration, global_step)
            # if len(inputs.shape) == 5:
            #     if index % 500 == 0:
            #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
            #         # NCLHW->LNCHW
            #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
            #         writer.add_image('img', img, global_step=global_step)
            # elif len(inputs.shape) == 4:
            #     if index % 500 == 0:
            #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
            #                          global_step=global_step)

    process.close()
    return global_step


def val_classifier(data_loader, model, loss_function, global_step, args, writer):
    right_num_total = 0
    total_num = 0
    loss_total = 0

    pretext_loss_total = 0
    mask_loss_total = 0
    jigsaw_T_loss_total = 0
    joint_loss_total = 0
    reverse_loss_total = 0
    simloss_total = 0


    step = 0
    # process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    process = tqdm(data_loader, desc='Val: ')
    # s = time.time()
    # t=0
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    for index, (inputs, labels, path) in enumerate(process):
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.loss == 'cross_entropy_naive':
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        with torch.no_grad():
            if type(args.device_id) is list and len(args.device_id) > 0:
                inputs, targets, labels = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            else:
                pass
            outputs, pretext_loss, mask_loss, jigsaw_T_loss, joint_loss, reverse_loss, simloss = model(inputs)
            if len(outputs.data.shape) == 3:  # T N cls
                _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                score_frag.append(outputs.data.cpu().numpy().transpose(1,0,2))
            else:
                _, predict_label = torch.max(outputs.data, 1)
                score_frag.append(outputs.data.cpu().numpy())
            loss = loss_function(outputs, targets)

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + ',' + str(true[i]) + '\n')
            if x != true[i]:
                wrong_path_pre_ture.append(str(path[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()
        pretext_loss = pretext_loss.mean().item()
        mask_loss = mask_loss.mean().item()
        jigsaw_T_loss = jigsaw_T_loss.mean().item()
        joint_loss = joint_loss.mean().item()
        reverse_loss = reverse_loss.mean().item()
        simloss = simloss.mean().item()

        # ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        pretext_loss_total += pretext_loss
        mask_loss_total += mask_loss
        jigsaw_T_loss_total += jigsaw_T_loss
        joint_loss_total += joint_loss
        reverse_loss_total += reverse_loss
        simloss_total += simloss
        step += 1

        process.set_description(
            'Val-batch: acc: {:4f}, loss: {:4f}, pret: {:4f}, mask: {:4f}, PT: {:4f}, PS: {:4f}, RE: {:4f}, sim: {:4f}, time: {:4f}'.format(acc, ls, pretext_loss, mask_loss, jigsaw_T_loss, joint_loss, reverse_loss, simloss, process.iterable.last_duration))
        # process.set_description_str(
        #     'Val: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(t, t, t), refresh=False)
        # if len(inputs.shape) == 5:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         # NCLHW->LNCHW
        #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
        #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
        #         writer.add_image('img', img, global_step=global_step)
        # elif len(inputs.shape) == 4:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
        #                          global_step=global_step)
    # t = time.time()-s
    # print('time: ', t)
    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    process.close()
    loss = loss_total / step
    pretext_loss_val = pretext_loss_total / step
    mask_loss_val = mask_loss_total / step
    jigsaw_T_loss_val = jigsaw_T_loss_total / step
    joint_loss_val = joint_loss_total / step
    reverse_loss_val = reverse_loss_total / step
    simloss_val = simloss_total /step
    accuracy = right_num_total / total_num
    print('test result: acc: {:4f}, loss: {:4f}, pret: {:4f}, mask: {:4f}, PT: {:4f}, PS: {:4f}, RE: {:4f}, sim: {:4f}'.format(accuracy, loss, pretext_loss_val, mask_loss_val, jigsaw_T_loss_val, joint_loss_val, reverse_loss_val, simloss_val))
    if args.mode == 'train_val' and writer is not None:
        writer.add_scalar('loss', loss, global_step)
        writer.add_scalar('pretext', pretext_loss_val, global_step)
        writer.add_scalar('mask_ls', mask_loss_val, global_step)
        writer.add_scalar('jigT_ls', jigsaw_T_loss_val, global_step)
        writer.add_scalar('Joint_ls', joint_loss_val, global_step)
        writer.add_scalar('Reverse_ls', reverse_loss_val, global_step)
        writer.add_scalar('sim_ls', simloss_val, global_step)
        writer.add_scalar('acc', accuracy, global_step)
        writer.add_scalar('batch time', process.iterable.last_duration, global_step)

    return loss, accuracy, score_dict, all_pre_true, wrong_path_pre_ture

