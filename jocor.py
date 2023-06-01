
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset,DataLoader  #Gives easier dataset managment and creates mini batches
from torchvision.transforms.functional import InterpolationMode
from skimage import exposure, img_as_ubyte
from numpy.testing import assert_array_almost_equal

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse, sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.init as init 
import wandb
wandb.init(project="Jocor_bach")


# writer = SummaryWriter()
#https://pytorch.org/docs/stable/tensorboard.html



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--w_decay', type=float, default=0.001) # weight decay changed from 0.03 to 0.001
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type=int, default=2,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='bach')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=50000)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--co_lambda', type=float, default=0.45)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn,gnet]', default='gnet')
parser.add_argument('--save_model', type=str, help='save model?', default="False")   # change from True to False
parser.add_argument('--save_result', type=str, help='save result?', default="True")


torch.cuda.empty_cache()
args = parser.parse_args(args=[])

#number of classes
img_res = 4096
n_outputs=4
batch_size = 64
model_path = "/workspace/subrat/"
#B_size = batch_size


# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)
print(f"the device is :{device}")   

# Hyper Parameters
#batch_size = batch_size
learning_rate = args.lr

def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
       root (str): Path to directory whose folders need to be listed
       prefix (bool, optional): If true, prepends the path to each result, otherwise
          only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
         filter(
              lambda p: os.path.isdir(os.path.join(root, p)),
              os.listdir(root)
         )
    )

    # if prefix is True:
    #      directories = [os.path.join(root, d) for d in directories]

    return directories



#def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root"""


def build_for_bach(size, noise):
        """ The noise matrix flips to the "next" class with probability 'noise'.
        """

        assert(noise >= 0.) and (noise <= 1.)

        P = (1. - noise) * np.eye(size)
        for i in np.arange(size - 1):
            P[i, i+1] = noise

         # adjust last row
        P[size-1, 0] = noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P




# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
    """
   # print(np.max(y), P.shape[0])

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

     # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        #draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0] # P[i,:][0], Subrat Modified here
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=4):
        """mistakes:
          flip in the pair
        """
        P = np.eye(nb_classes)
        n = noise

        if n > 0.0:
            # 0 -> 1
            P[0, 0], P[0, 1] = 1. - n, n
            # print(f" p_oo, p_o1:{P[0, 0]}, {P[0, 1] }")
            for i in range(1, nb_classes-1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

            y_train_noisy = multiclass_noisify(y_train, P=P,
                                            random_state=random_state)
            # actual_noise = (y_train_noisy != y_train) #gora_debug
            # print(actual_noise)  #gora_debug
            #exit()

            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
        # print(P) 
        return y_train, actual_noise




def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=4):
    """mistakes:flip in the symmetric way"""
    # print(f"train labels inside noisify_multiclass_symmetric: {y_train}")
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        
        P[nb_classes-1, nb_classes-1] = 1. - n
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)   #alternative: read noisy_label_csv in this line
        # print(f" the y_train is: {y_train.hist()} ")
        #print(f" the y_train_noisy is false: {y_train_noisy==y_train} ")  # SUBRAT commented this line on 4 feb
        # exit()
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
       # print('Actual noise %.2f' % actual_noise) # ////////////////// 4 feb
        y_train = y_train_noisy
    
    # print(P)

    return y_train, actual_noise



def noisify(dataset='bach', nb_classes=4, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    # print(f"trin labels inside noisify: {train_labels}")

    train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
 
   #  if noise_type == 'pairflip':
   #      train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
   #  if noise_type == 'symmetric':
   #      train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
   #  if noise_type == 'asymmetric':
   #      if dataset == 'mnist':
   #          train_noisy_labels, actual_noise_rate = noisify_mnist_asymmetric(train_labels, noise_rate, random_state=random_state)
   #      elif dataset == 'cifar10':
   #          train_noisy_labels, actual_noise_rate = noisify_cifar10_asymmetric(train_labels, noise_rate, random_state=random_state)
   #      elif dataset == 'cifar100':
   #          train_noisy_labels, actual_noise_rate = noisify_cifar100_asymmetric(train_labels, noise_rate, random_state=random_state)
   #      elif dataset == 'bach':
   #          train_noisy_labels, actual_noise_rate = noisify_cifar100_asymmetric(train_labels, noise_rate, random_state=random_state)
    return train_noisy_labels, actual_noise_rate




def call_bn(bn, x):
    return bn(x)

# Model
gnet = torchvision.models.googlenet(pretrained=True)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(img_resolution, 256)
        self.fc2 = nn.Linear(256, n_outputs)

    def forward(self, x):
        x = torch.mean(x,1)   ####### added for combining all the three channel to one by taking mean
        x = x.view(-1, img_resolution)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def plot_result(accuracy_list,pure_ratio_list,name="test.png"):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_list, label='test_accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(pure_ratio_list, label='test_pure_ratio')
    plt.savefig(name)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
   #batch_size = B_size

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]



def kl_loss_compute(pred, soft_targets, reduction=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')

    if reduction:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)



def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.45, is_train = None):

    loss_pick_1 = F.cross_entropy(y_1, t, reduction='none') * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduction='none') * (1-co_lambda)
   # print(f"the value loss_pick_1 and loss_pick_2 are : {loss_pick_1}, {loss_pick_2}") # //////////////// 4 feb
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduction='none') + co_lambda * kl_loss_compute(y_2, y_1,reduction='none')).cpu()
    
   # print(f" forget rate is  {forget_rate}")  #//////////////// 4 feb
   # print(f" loss pick is {loss_pick}") # ////////////////// 4 feb
   # print(f" ind is {ind}") # //////////////// 4 feb    
    
    ind_sorted = np.argsort(loss_pick.data)  #sorts the array and then returns the indices in terms of the input array
    loss_sorted = loss_pick[ind_sorted] #

    remember_rate = 1 - forget_rate
   # print(f" loss sorted is: {loss_sorted}") # ///////////// 4 feb
     
    #print(len(loss_sorted))
    num_remember = int(remember_rate * len(loss_sorted))  # chnge float to int by SUBRAT
    #print(float(num_remember))
   # print(remember_rate)#//////////////////////////// 4 feb
   # print(len(loss_sorted)) # ////////////////////// 4 feb
    
    #if is_train:

   # print(num_remember) # //////////////////////////////// 4feb
    #num_remember=float(num_remember)
   # print(num_remember)# //////////////////////////////// 4 feb
    
    pure_ratio = float(np.sum(noise_or_not[ind[ind_sorted[:num_remember]]]))/float(num_remember)  # change int to float by SUBRAT 
    #print(pure_ratio)

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])
    lcon_loss = (kl_loss_compute(y_1, y_2,reduction='none') + kl_loss_compute(y_2, y_1,reduction='none')).cpu()
    
    wandb.log({"Loss": loss})
    return loss, lcon_loss , pure_ratio, pure_ratio  #why twice the variable 'loss' is getting passed?? because they are same for the both model getting calculated from both of the model


class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = batch_size
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.8
        mom2 = 0.2
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
#         print(np.linspace(0, forget_rate ** args.exponent, args.num_gradual).shape())
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        
        elif args.model_type == 'gnet':
            self.model1 = gnet
            self.model2 = gnet
         
        self.model1 =  self.model1.to(device)
        print(f'The parameters of model 1 is :{self.model1.parameters}')
        
        self.model2 = self.model2.to(device)
        print(f'Device: {device}')
        print(f"model2 is {self.model2}")
        #self.optimizer = torch.optim.AdamW(params, lr = args.lr, weight_decay = args.w_decay)
        #self.optimizer = torch.optim.AdamW(params, lr = args.lr, weight_decay = args.w_decay)
        self.optimizer = torch.optim.AdamW(list(self.model1.parameters()) + list(self.model2.parameters()), lr=learning_rate)

        self.loss_fn = loss_jocor
        self.adjust_lr = args.adjust_lr


    def evaluate_1(self, val_loader, epoch):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        #if self.adjust_lr == 1:
        #    self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        loss_1 = []
        loss_2 = []
        for i, (images, labels, indexes) in enumerate(val_loader):
            #print("hey jude")
            ind = indexes.cpu().numpy().transpose()
            #print("nooooooooooooooooooo")
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)
        
            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1_, loss_2_, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], ind, self.noise_or_not, self.co_lambda, is_train = False)
            loss_1.append(loss_1_)
            loss_2.append(loss_2_)

     
            #self.optimizer.zero_grad()
            #loss_1_.backward()
            #self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

           # if (i + 1) % self.print_freq == 0:
           #     print(
           #         'Epoch [%d/%d], Iter [%d/%d] Evaluation Accuracy1: %.4F, Evaluation Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
           #         % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
           #            loss_1_.data.item(), loss_2_.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        avg_test_acc1 = float(train_correct) / float(train_total)
        avg_test_acc2 = float(train_correct2) / float(train_total2)
        avg_loss_1 = sum(loss_1) / len(loss_1) 
        avg_loss_2 = sum(loss_2) / len(loss_2)

       # print('Epoch [%d/%d], Iter [%d/%d] Evaluation Accuracy1: %.4F, Evaluation Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
       #       % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,loss_1_.data.item(), loss_2_.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))


        return avg_test_acc1, avg_test_acc2, pure_ratio_1_list, pure_ratio_2_list, avg_loss_1, avg_loss_2



    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'rain' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        loss_1 = []
        loss_2 = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1_, loss_2_, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], ind, self.noise_or_not, self.co_lambda, is_train = True)
            loss_1.append(loss_1_)
            loss_2.append(loss_2_)
     
            self.optimizer.zero_grad()
            loss_1_.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            #if (i + 1) % self.print_freq == 0:
            #    print(
            #        'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
            #        % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
            #           loss_1_.data.item(), loss_2_.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        avg_loss_1 = sum(loss_1) / len(loss_1) 
        avg_loss_2 = sum(loss_2) / len(loss_2)


        #print('Epoch [%d/%d]: Train Acc1: %.4F, Train Acc2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
        #            % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
        #               loss_1_.data.item(), loss_2_.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))


        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, avg_loss_1, avg_loss_2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1



root_dir = '/workspace/Data/train_image/'
csv_file = '/workspace/Data/Clean_train_data_encd.csv'

class BACH(Dataset):
    def __init__(self, csv_file=None, transform=None,noise_type=None,noise_rate=0.2,random_state = 0 ):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = 'bach'
        self.train_labels = self.csv_data["label"]        
        self.noise_type=noise_type
        self.nb_classes = 4 
        ## gora####
        self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
        
        # for i in self.train_noisy_labels:
        #     print(f"  {i}")

        #self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
        
        # print(f"length of y_train and y_train_noisy is {_train_labels.shape(), self.train_noisy_labels.shape()}, ")
        # _train_labels=[i[0] for i in self.train_labels]
        self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(self.train_labels)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        label=self.csv_data.loc[index, 'label']
        img_path = self.csv_data.loc[index, 'Name']

        img_path = os.path.join(root_dir, img_path)
        #print(img_path)
        image = io.imread(img_path)
#         print(image)
      

        image = img_as_ubyte(exposure.rescale_intensity(image))
#         print(image)
#         exit()
        
        #plt.imshow(image, cmap=None)
#         transform = transforms.Resize((300,350))
#         resized_img = transform(image)
#         print(resized_img.shape)
#         y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform: 
       
            image = self.transform(image)    #how do you make this inside the gpu
        #print(image.size())
       
        return (image, label, index)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



# load dataset
# if args.dataset == 'HAM10000':
#     input_channel = 3
#     num_classes = 7
#     init_epoch = 10
#     filter_outlier = True
#     args.epoch_decay_start = 40
#     args.model_type = "mlp"
#     args.n_epoch = 200
#     dataset = HAM10000(root='./home/subrat/JoCoR-env/archive/HAM10000_images_part_1/',
#                           train=True,
#                           transform=transforms.ToTensor(),
#                           noise_type=args.noise_type,
#                           noise_rate=args.noise_rate
#                           )
#     print(dataset)
#     test_dataset = HAM10000(root='/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/',
#                          train=False,


if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

    
# input_channel = 3
# num_classes = 7
# init_epoch = 10
# filter_outlier = True
# args.epoch_decay_start = 40
# args.model_type = "mlp"
# args.n_epoch = 200
# dataset = HAM10000(root='./home/subrat/JoCoR-env/archive/HAM10000_images_part_1/',
#                       train=True,
#                       transform=transforms.ToTensor(),
#                       noise_type=args.noise_type,
#                       noise_rate=args.noise_rate
#                       )
# Load Data
# transform for rectangular resize

transform=transforms.Compose([ 
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        #[transforms.RandomRotation(degrees=d)(Image) for d in range(50,151,50)],
        #[transforms.CenterCrop(size=size)(images) for size in (128,64,32)],
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 0.2)),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        #transforms.RandomResizedCrop(size=(75, 150)),
        transforms.RandomRotation(degrees=45, interpolation = InterpolationMode.BILINEAR),
        transforms.Resize(350, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None),  #changed from 150 to 400 by SUBRAT
        #transforms.CenterCrop(size=128)
        ]) 

# img = Image.fromarray(np.astype(np.uint8))
# load dataset

if args.dataset == 'bach':
    input_channel = 3
    num_classes = 4
    init_epoch = 10      #Confusion about init_epoch
    filter_outlier = True
    args.epoch_decay_start = 150
    args.model_type = "gnet"
    args.n_epoch = 200


train_dataset = BACH(csv_file="/workspace/Data/train_data_clean.csv",transform=transform, noise_type=args.noise_type) 
val_dataset = BACH(csv_file="/workspace/Data/val_data_clean.csv",transform=transform, noise_type=args.noise_type)
# test_dataset = BACH(csv_file="/home/subrat/JoCoR_bach/Data/test_data_clean.csv",transform=transform,noise_type=args.noise_type)


def main():
    # Data Loader (Input Pipeline)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2)
    print('dataset is loaded')

    # Define models
    print('building model...')

    # _noise_or_not_ = 
    model = JoCoR(args, train_dataset, device, input_channel, num_classes)
    print(model)

    train_acc1 = 0
    train_acc2 = 0
    epoch = 0
    # evaluate models with random weights
    val_acc1, val _acc2, _, _, _, _ = model.evaluate_1(val_loader, epoch)
    print(
        'Epoch [%d/%d] Validation_Accuracy on the %s val_images: Model1 %.4f %% Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(val_dataset), val_acc1, val_acc2))


    acc_list1 = []
    acc_list2 = [] 
    # training
    for epoch in range(1, args.n_epoch):
        print(f"Epoch: {epoch} in progress...")
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, train_loss_1, train_loss_2 = model.train(train_loader, epoch)

        # evaluate models
        val_acc1, val_acc2,_,_, loss_1, loss_2 = model.evaluate_1(val_loader, epoch)

        print(  'Train Loss: Model1 %.4f %% Model2 %.4f' % (
                train_loss_1, train_loss_2))
        
        print(  'Train  Acc: Model1 %.4f %% Model2 %.4f' % (
                train_acc1, train_acc2))
        print(  'Val   Loss: Model1 %.5f %% Model2 %.5f' % ( loss_1, loss_2))

        print(  'Val Accuracy on the %s val images: Model1 %.4f %% Model2 %.4f' % (
                 len(val_dataset), val_acc1, val_acc2))
        # save results
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Train Loss: Model1 %.4f %% Model2 %.4f' % (
                len(train_dataset), train_loss_1, train_loss_2))

            print(
                'Train Accuracy: Model1 %.4f %% Model2 %.4f' % (
                len(train_dataset), train_acc1, train_acc2))

            print(
                'Val Loss: Model1 %.4f %% Model2 %.4f' % ( loss_1, loss_2))

            print(
                'Val Accuracy on the %s val images: Model1 %.4f %% Model2 %.4f' % (
                 len(val_dataset), val_acc1, val_acc2))


            # writer.add_scalar("train_acc1_model1", train_acc1, epoch)
            # writer.add_scalar("train_acc2_model2", train_acc2, epoch)
            # writer.add_scalar("train_loss1_model1", train_loss_1, epoch)
            # writer.add_scalar("train_loss2_model2", train_loss_2, epoch)

        
            # writer.add_scalar("eval_acc1_model1", val_acc1, epoch)
            # writer.add_scalar("eval_acc_model2", val_acc2, epoch)
            # writer.add_scalar("eval_loss1_model1", loss_1, epoch)
            # writer.add_scalar("eval_loss2_model2", loss_2, epoch)

        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)



          #  print(
          #      'Epoch [%d/%d] Validation_Accuracy on the %s val images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (
          #          epoch + 1, args.n_epoch, len(val_dataset), val_acc1, val_acc2, mean_pure_ratio1,
          #          mean_pure_ratio2))


        if epoch >= 190:
            acc_list1.extend([val_acc1])
            acc_list2.extend([val_acc2])

    avg_acc1 = sum(acc_list1)/len(acc_list1)
    avg_acc2 = sum(acc_list2)/len(acc_list2)
    print(len(acc_list1))
    print(len(acc_list2))
    print("the average acc1 in last 10 epochs: {}".format(str(avg_acc1)))
    print("the average acc2 in last 10 epochs: {}".format(str(avg_acc2)))
    writer.close()
    torch.save(model, model_path)


if __name__ == '__main__':
    main()
