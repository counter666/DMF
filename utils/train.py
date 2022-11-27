import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from data2 import train_dataset
import torch.optim as optim
from utils.loss_function import DiceLoss
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from DMF import DMFF
from utils import ramps, losses, metrics


parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str,
                    default='LA_CA/DMF_weight')
parser.add_argument('--max_iterations', type=int,
                    default=6500)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--labelnum', type=int, default=10)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--deterministic', type=int, default=1)

args = parser.parse_args() 

all_train_iter_loss = [] 

a = 0
outputs_tanh = []
outputs = []

snapshot_path = "D:/Study/" + args.exp + \
                "_{}labels_beta_{}/".format(args.labelnum, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr


if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')  

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        net = DMFF(in_channels=3, classes=2)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_() 
        return model


    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(train_dataset, batch_size=8,
                              pin_memory=True, worker_init_fn=worker_init_fn)



    model.train() 

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001) 

    ce_loss = DiceLoss()

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1  
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70) 
    for epoch_num in iterator:
        time1 = time.time() 
        for i_batch, (img1, img2, label_batch) in enumerate(trainloader):
            time2 = time.time()
            img1, label_batch = img1.cuda(), label_batch.cuda()
            img2 = img2.cuda()
            outputs = model(img1, img2)
            outputs_soft = torch.sigmoid(outputs)  
            loss = ce_loss(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            mask_np = label_batch.cpu().detach().numpy().copy()
            mask_np = np.argmin(mask_np, axis=1)
            outputs_soft_np = outputs_soft.cpu().detach().numpy().copy()
            outputs_soft_np = np.argmin(outputs_soft_np, axis=1)

            dc = metrics.dc(outputs_soft_np, mask_np) 
            jc = metrics.jc(outputs_soft_np, mask_np)

            print('dc:', dc)
            print('jc:', jc)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num) 

            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration, %d, loss, %f, dc, %f, jc, %f' % (iter_num, loss.item(), dc, jc))

            if iter_num % 1 == 0:  
                lr_ = base_lr * ((1 - (iter_num / max_iterations)) ** 0.9)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                    print('lr', param_group['lr'])
            if iter_num % 500 == 0: 
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
