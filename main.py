import os
import math
import logging
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import imagenet_class_list
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', choices=('DC, DSA'))
    parser.add_argument('--imagenet_group', type=str, choices=('automobile', 'terrier', 'fish', 'truck', 'lizard', 'insect'))
    parser.add_argument('--contrast', action='store_true')
    parser.add_argument('--outer_warmup', type=int, default=250)
    parser.add_argument('--inner_warmup', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--outer_loop', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.inner_loop, args.T = get_loops(args.ipc)
    assert args.inner_warmup <= args.inner_loop
    assert args.outer_warmup <= args.outer_loop
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    eval_it_pool = np.arange(0, args.outer_loop+1, args.eval_freq).tolist() if args.eval_mode == 'S' else [args.outer_loop] # The list of iterations when we evaluate models and record results.
    icl = None if args.imagenet_group == None else imagenet_class_list.icl(args.imagenet_group)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, icl)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    if args.eval_mode == 'S':
        accs_logs = dict() # record performances of all experiments
        for eval_it in eval_it_pool:
            accs_logs[eval_it] = []

    data_save = []

    for exp in range(args.num_exp):
        logging.info('\n================== Exp %d ==================\n '%exp)
        logging.info('Hyper-parameters: \n%s'%args)
        logging.info('Evaluation model pool: %s'%model_eval_pool)

        ''' organize the real dataset '''
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            logging.info('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            logging.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        '''image_syn: |---class 0, phase 0---|---class 1, phase 0---|---...---|---class 8, phase n---|---class 9, phase n---|'''

        logging.info('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        criterion_ = nn.CrossEntropyLoss(reduction='none').to(args.device)
        logging.info('%s training begins'%get_time())

        for ol in range(args.outer_loop+1):

            ''' Evaluate synthetic data '''
            if ol in eval_it_pool:
                for model_eval in model_eval_pool:
                    logging.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, outer_loop = %d'%(args.model, model_eval, ol))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        logging.info('DSA augmentation strategy: \n%s'%args.dsa_strategy)
                        logging.info('DSA augmentation parameters: \n%s'%args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        logging.info('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    logging.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if args.eval_mode == 'S':
                        accs_logs[ol] += accs

                    if ol == args.outer_loop: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, ol))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
                if ol == args.outer_loop: # record the final results
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))
                    break

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            for il in range(args.inner_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                gw_real = [0. for i in range(len(net_parameters))]
                gw_real_accu = [0. for i in range(len(net_parameters))]
                gw_syn_accu = [0. for i in range(len(net_parameters))]
                dsa = []
                for c in range(num_classes):
                    if args.contrast and (args.outer_warmup <= ol or args.inner_warmup <= il):
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        if args.dsa:
                            img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            dsa.append(img_syn)
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real_i = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((gw_real_i[i].detach().clone() / float(num_classes) + _ for i, _ in enumerate(gw_real)))
                    else:
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real_accu = list((gw_real[i].detach().clone() / float(num_classes) + _ for i, _ in enumerate(gw_real_accu)))
                        gw_real = list((_.detach().clone() for _ in gw_real))
                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                        gw_syn_accu = list((gw_syn[i].detach().clone() / float(num_classes) + _ for i, _ in enumerate(gw_syn_accu)))
                        loss += match_loss(gw_syn, gw_real, args)

                if args.contrast and (args.outer_warmup <= ol or args.inner_warmup <= il):
                    if args.dsa:
                        img_syn = torch.cat(dsa, dim=0)
                    else:
                        img_syn = image_syn
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, label_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss += match_loss(gw_syn, gw_real, args)
                    loss_print = loss.item()
                else:
                    loss_print = match_loss(gw_syn_accu, gw_real_accu, args)
                    loss_print = loss_print.item()
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss_print

                if il == args.inner_loop - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_net_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_net_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.T):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

            loss_avg /= args.inner_loop

            if ol%10 == 0:
                logging.info('%s iter = %04d, loss = %.4f' % (get_time(), ol, loss_avg))


    logging.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logging.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    if args.eval_mode == 'S':
        for eval_it in eval_it_pool:
            accs = accs_logs[eval_it]
            logging.info('Iter:%d, mean  = %.2f%%  std = %.2f%%'%(eval_it, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


