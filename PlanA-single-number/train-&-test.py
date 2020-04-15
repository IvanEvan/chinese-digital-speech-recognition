#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Evan
import torch
import os
import numpy as np

from utils.dataProcess import getFolder, loadDataset, Featurize, dynamic_vad, featurize_from_nparray
from utils.VGG import vgg13_bn

import shutil
from tensorboardX import SummaryWriter

writer = SummaryWriter('log')


def save_checkpointer(checkpointer, is_best, file_name='log/checkpoint.pth'):
    torch.save(checkpointer, file_name)
    if is_best:
        shutil.copyfile(file_name, 'model_best.pth')


def train(mod, epochs, batch_size, lr=4e-5, weight_decay=0.000001):
    mod.cuda()
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # switch to train mode
    mod.train()
    print("训练开始")
    folder = getFolder()

    best_acc = 0

    for epoch in range(epochs):

        train_data_loader, valid_data_loader = loadDataset(folder=folder, partition=epoch % len(folder),
                                                           batch_size=batch_size)
        losses = []
        top1 = []

        for i, data_load in enumerate(train_data_loader):
            dat, labl = data_load

            data = torch.autograd.Variable(dat.cuda())

            label = torch.autograd.Variable(labl.cuda())

            predict_y = mod(data)
            loss = criterion(predict_y, label)

            prec1, _ = accuracy(predict_y.data, label, topk=(1, 5))

            losses.append(loss.item())
            top1.append(prec1.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # batch_time.update(time.time() - end)
            niter = epoch * len(train_data_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), niter)
            writer.add_scalar('Train/Prec@1', prec1.item(), niter)

            print('epoch-%d step-%d : loss-%.4f prec@1-%.4f' % (epoch, i, loss.item(), prec1.item()))

        print('epoch-%d average : loss-%.4f prec@1-%.4f' % (epoch, sum(losses) / len(losses), sum(top1) / len(top1)))
        acc = validate(epoch, valid_data_loader, mod, criterion)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        checkpointer = {
            'epoch': epoch + 1,
            'acc': acc,
            'state_dict': mod.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpointer(checkpointer, is_best)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).cuda())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def validate(epoch, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        losses = []
        top1 = []
        for i, (input, target) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, _ = accuracy(output.data, target, topk=(1, 5))

            losses.append(loss.item())
            top1.append(prec1.item())

            niter = epoch * len(val_loader) + i
            writer.add_scalar('Test/Loss', loss.item(), niter)
            writer.add_scalar('Test/Prec@1', prec1.item(), niter)

    print('validation average : loss-%.4f prec@1-%.4f' % (sum(losses) / len(losses), sum(top1) / len(top1)))

    return sum(top1) / len(top1)


def infer(model, data_path):
    feature = Featurize(data_path)
    feature = feature[None, :, :, :]
    model.eval()
    with torch.no_grad():
        feature_var = torch.autograd.Variable(torch.tensor(feature))
    output = model(feature_var)
    argmax = torch.argmax(output, dim=1).item()

    return argmax


def predict(root_dir, model, model_file_path):
    folder = getFolder(root_dir=root_dir, fold_n=1)[0]

    state_dict = torch.load(model_file_path, map_location=lambda storage, loc: storage)['state_dict']

    model.load_state_dict(state_dict)

    # model.load('model.path', map_location='cpu')
    # switch to evaluate mode

    correct = 0
    for i in folder:
        wav_path, true_label = i
        wav_name = os.path.basename(wav_path)
        pre_label = infer(model, wav_path)

        if str(pre_label) == true_label:
            correct += 1

        print('file: %s | true lable: %s | pre label" %s' % (wav_name, true_label, str(pre_label)))

    print('Summary | tolal file: %d | correct file %d | acc %.4f' % (len(folder), correct, correct / len(folder)))


def continuous_sequence_prediction(target_file, model):
    lable_num = str(os.path.basename(target_file).split('.wav')[0]).split('_')[-1]
    wavs, seg_index = dynamic_vad(target_file)

    pre_lab = ''
    for nm, sliced in enumerate(seg_index):
        wav_pice = np.array(wavs[sliced[0]:sliced[1]])

        feature = featurize_from_nparray(wav_pice)
        feature = feature[None, :, :, :]

        with torch.no_grad():
            feature_var = torch.autograd.Variable(torch.tensor(feature))
        output = model(feature_var)
        argmax = torch.argmax(output, dim=1).item()

        pre_lab += str(argmax)

    return '%s--->%s' % (lable_num, pre_lab)


def run_eval(wav_file_path, model_file_path):
    model = vgg13_bn()
    # model.train()
    # train(model, epochs=50, batch_size=64)
    # predict('D:\\数字\\test_data', model)
    # predict('E:\\code\\chinese-digital-speech-recognition\\process_lab\\test', model)

    state_dict = torch.load(model_file_path, map_location=lambda storage, loc: storage)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    result = continuous_sequence_prediction(wav_file_path, model)

    return result


if __name__ == '__main__':
    # train model
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = vgg13_bn()
    model.train()
    train(model, epochs=50, batch_size=64)

    # predict single number from wave folder
    predict(r'D:\your_wave_folder', model, r'pretrained-model\model.path')

    # predict continuous number from wave
    print(run_eval(r'E:\yourfolder\40958312.wav', r'pretrained-model\model.path'))

