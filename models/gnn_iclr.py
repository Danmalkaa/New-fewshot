#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor



# class PDE_GCN(nn.Module): # Our GConv
#     def __init__(self, nf_input, nf_output, J, bn_bool=True):
#         super(PDE_GCN, self).__init__()
#         stdv = 1e-2
#         self.K1Nopen = nn.Parameter(torch.randn(nf_input, nNin) * stdv)
#         self.K2Nopen = nn.Parameter(torch.randn(nf_input, nf_input) * stdv)
#         self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  # num_output on left size
#
#         self.KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)
#         rrnd = torch.rand(nlayer, Nfeatures, nhid) * (1e-3)
#         self.KN1 = nn.Parameter(identityInit(self.KN1) + rrnd)
#
#         self.alpha = nn.Parameter(-0 * torch.ones(1, 1))
#
#         self.KN2 = nn.Parameter(torch.rand(nlayer, nhid, 1 * nhid) * stdvp)
#         self.KN2 = nn.Parameter(identityInit(self.KN2))
#
#         self.J = J #TODO:FIX
#         self.num_inputs = J*nf_input #TODO:FIX
#         self.num_outputs = nf_output #TODO:FIX
#         self.fc = nn.Linear(self.num_inputs, self.num_outputs) #TODO:FIX
#
#         self.bn_bool = bn_bool #TODO:FIX
#         if self.bn_bool: #TODO:FIX
#             self.bn = nn.BatchNorm1d(self.num_outputs) #TODO:FIX
#
#     def forward(self, input):
#         gradX = self.grad(input) #TODO: FIX
#         gradX = self.drop(gradX)
#         dxn = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i])
#         dxn = self.edgeDiv(dxn)
#
#         tmp_xn = xn.clone()
#         beta = F.sigmoid(self.alpha)
#         alpha = 1 - beta
#         alpha = alpha / self.h
#         beta = beta / (self.h ** 2)
#
#         xn = (2 * beta * xn - beta * xn_old + alpha * xn - dxn) / (beta + alpha)
#         xn_old = tmp_xn
#
#         W = input[0]  #TODO: FIX
#         x = gmul(input) # out has size (bs, N, num_inputs) #TODO: FIX
#         #if self.J == 1:
#         #    x = torch.abs(x)
#         x_size = x.size() #TODO: FIX
#         x = x.contiguous() #TODO: FIX
#         x = x.view(-1, self.num_inputs) #TODO: FIX
#         x = self.fc(x) # has size (bs*N, num_outputs) #TODO: FIX
#
#         if self.bn_bool: #TODO: FIX
#             x = self.bn(x)
#
#         x = x.view(*x_size[:-1], self.num_outputs) #TODO: FIX
#
#
#         return W, x
class Wcompute_pde(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute_pde, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.activation = activation

    def forward(self, features, W_id, first_run=True):
        features = features.squeeze()
        features = torch.transpose(features, 1,2)
        D = torch.relu(torch.sum(features ** 2, dim=2, keepdim=True) + torch.transpose(torch.sum(features ** 2, dim=2,
                       keepdim = True), 1, 2) - 2 * features @ torch.transpose(features, 1, 2))
        D_std = torch.permute(torch.std(D, (1, 2)).repeat(6, 6, 1), (2, 0, 1)) #std for every 5x5 matrix, repeat and permute to have the same dimentions as D [50,5,5]
        D = D / D_std
        D = torch.exp(-2 * D)
        W_new = D.unsqueeze(3)
        W_new = W_new.contiguous()

        if self.activation == 'softmax':
         #   print("W new size in softmax:", W_new.size())
         #   print("W id size in softmax:", W_id.size())
            W_new = W_new - W_id.expand_as(W_new) * 1e1 # TODO: change back to 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            # W_new = F.softmax(W_new) # TODO: Uncomment
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new


def identityInit(tensor):
    I = torch.eye(tensor.shape[1], tensor.shape[2]).unsqueeze(0)
    II = torch.repeat_interleave(I, repeats=tensor.shape[0], dim=0)
    return II

class PDE_GCN(nn.Module): #

    def conv1(X, Kernel):
        return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


    def nodeGrad(self, x, W=[]): # insert our Weight Matrix to W
        if len(W) == 0:
            W = self.W

        if x.dim() == 3 :
        # has Batch
            g = W[:,:,:,1].unsqueeze(1) * (x.unsqueeze(2) - x.unsqueeze(3))

        if x.dim() == 2 :
        # no Batch dim
            g = W[:,:,:,1].unsqueeze(1)* (x.unsqueeze(1) - x.unsqueeze(2))
        # if W.shape[0] == x.shape[2]:
        #     # if its degree matrix
        #     g = W[self.iInd] * (x[:, :, self.iInd] - x[:, :, self.jInd])
        # else:
        #     g = W * (x[:, :, self.iInd] - x[:, :, self.jInd])


        # W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        # W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        # W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N
        return g

    def edgeConv(self, xe, K, groups=1):
        # if xe.dim() == 4:
        #     if K.dim() == 2:
        #         xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
        #     else:
        #         xe = conv2(xe, K, groups=groups)
        if xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            # else:
                # xe = conv1(xe, K, groups=groups)
        return xe

    def singleLayer(self, x, K, groups=1):
        # if K.shape[0] != K.shape[1]:
        x = self.edgeConv(x, K, groups=groups)
        x = F.relu(x)
        return x

    def edgeDiv(self, g, W=[]): #TODO - Second LOOK
        if len(W) == 0: # TODO: Decide if W or Ones_MAT
            W = self.W
        # x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device) # [5,64,5]
        x = torch.sum((W[:, :, :, 1].unsqueeze(1) * g), dim=2)    # TODO: Double-Check
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]
        # if W.shape[0] != g.shape[2]:
        #     x.index_add_(2, self.iInd, W[self.iInd] * g)
        #     x.index_add_(2, self.iInd, W[self.jInd] * g)
        # else:
        #     x.index_add_(2, self.iInd, W * g)
        #     x.index_add_(2, self.jInd, -W * g)

        return x

    def finalDoubleLayer(self, x, K1, K2): # taken from PDE-GCN
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def __init__(self, args, input_features, nf, J):
        super(PDE_GCN, self).__init__()
        self.args = args
        self.nnodes = args.train_N_way * args.train_N_shots + 1 # 1 for the unknown sample
        self.input_features = input_features
        self.nf = nf
        self.J = J
        self.num_layers = 4  # TODO: change to 2 - here we change the number of layers

        self.dropout = 0.01 # TODO: Change
        self.h = nn.Parameter(torch.Tensor([0.1])) # Our Change

        stdv = 1e-1 # TODO: Change to  1e-2
        stdvp = 1e-1 # TODO: Change to  1e-2

        self.K1Nopen = nn.Parameter(torch.randn(input_features, input_features) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(input_features, input_features) * stdv)
        self.KNclose = nn.Parameter(torch.randn(args.train_N_way * args.train_N_shots + 1 , input_features) * stdv)  # num_output on left size

        self.KN1 = nn.Parameter(torch.rand(self.num_layers, input_features, input_features) * stdvp)
        rrnd = torch.rand(self.num_layers, input_features, input_features) * (1e-2) # TODO: Change to random 1e-3
        self.KN1 = nn.Parameter(identityInit(self.KN1) + rrnd)

        self.alpha = nn.Parameter(-0 * torch.ones(1, 1))

        self.KN2 = nn.Parameter(torch.rand(self.num_layers, input_features, input_features) * stdvp)
        self.KN2 = nn.Parameter(identityInit(self.KN2))

        flag1,flag2 = 1, 0
        for i in range(self.num_layers):
            # if i >= 1:  #changed
            #     flag1 = 0
            #     flag2 = 1
            # else:
            #     flag1 = 1
            #     flag2 = 0
            module_w = Wcompute_pde(flag1 * self.input_features + flag2 * int(nf / 2),
                                flag1 * self.input_features + flag2 * int(nf / 2), operator='J2', activation='softmax',
                                ratio=[2, 1.5, 1, 1], drop=False)
            # module_l = Gconv(flag1 * self.input_features + int(nf / 2) * flag2, int(nf / 2), 2)  #changed
            self.add_module('layer_w{}'.format(i), module_w)
            # self.add_module('layer_l{}'.format(i), module_l)

        #self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2)*self.num_layers,
        #                           self.input_features + int(self.nf / 2) * (self.num_layers - 1), #changed
        #                          operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        #self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=True) #changed
        self.w_comp_last = Wcompute_pde(int(self.nf / 2)+1,  #this is the last layer without concatenating
                                    int(self.nf / 2)+1,  #changed
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        # self.layer_last = Gconv(int(self.nf / 2), args.train_N_way, 2, bn_bool=True)  #changed



    def forward(self, x):
        # x = [B,N,C]
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        if self.args.cuda:
            x = x.cuda()

        x = torch.transpose(x, 1,2)         # x = [B,C,N]

        xn = F.dropout(x, p=self.dropout)
        xn = self.singleLayer(xn, self.K1Nopen)  # First Layer
        x0 = xn.clone()
        xn_old = x0
        first_flag = True
        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](xn, W_init,first_flag)  #changed
            first_flag = False
            #print("Wi.size: ", Wi.size())
            #print("Wi: ", Wi[0,:,:,1])

            gradX = self.nodeGrad(xn, W=Wi)  #TODO: FIX # insert our Weight Matrix to W
            gradX = F.dropout(gradX, p=self.dropout)
            dxn = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i])
            dxn = self.edgeDiv(dxn,Wi)

            tmp_xn = xn.clone()
            beta = F.sigmoid(self.alpha)
            alpha = 1 - beta
            alpha = alpha / self.h
            beta = beta / (self.h ** 2)

            xn = (2 * beta * xn - beta * xn_old + alpha * xn - dxn) / (beta + alpha)
            xn_old = tmp_xn

        out = F.dropout(xn, p=self.dropout, training=self.training)
        out = F.conv1d(out, self.KNclose.unsqueeze(-1))

        out = torch.transpose(out.squeeze(),1,2)

        Wl = self.w_comp_last(out, W_init)  #changed
        #print("Wl.size", Wl.size())
        #print("Wl: ", Wl[0,:,:,1])
        # out = self.layer_last([Wl, xn])[1]  #changed # TODO: FIX
        #print("out.size", out.size())
        #print("this is the x before sending to models")
        #print(x_next)

        # F.log_softmax(xn, dim=1) # PDE-GCN PAPER RETURN

        # return out[:, 0, :], xn, Wl

        return out[:, 0, :], xn, Wl




def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id, first_run=True):
        if first_run:
            x = torch.transpose(x, 1,2)
            W1 = x.unsqueeze(2)

        else:
            x = torch.transpose(x, 1,2)
            W1 = x.unsqueeze(1)

        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new


class GNN_nl_omniglot(nn.Module): # Todo: Change name to Naive and uncomment Regular GNN
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl_omniglot, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 6  # TODO: change to 2 - here we change the number of layers
        for i in range(self.num_layers):
            if i >= 1:  #changed
                flag1 = 0
                flag2 = 1
            else:
                flag1 = 1
                flag2 = 0
            module_w = Wcompute(flag1 * self.input_features + flag2 * int(nf / 2),
                                flag1 * self.input_features + flag2 * int(nf / 2), operator='J2', activation='softmax',
                                ratio=[2, 1.5, 1, 1], drop=False)
            module_l = Gconv(flag1 * self.input_features + int(nf / 2) * flag2, int(nf / 2), 2)  #changed
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        #self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2)*self.num_layers,
        #                           self.input_features + int(self.nf / 2) * (self.num_layers - 1), #changed
        #                          operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        #self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=True) #changed
        self.w_comp_last = Wcompute(int(self.nf / 2),  #this is the last layer without concatenating
                                    int(self.nf / 2),  #changed
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        self.layer_last = Gconv(int(self.nf / 2), args.train_N_way, 2, bn_bool=True)  #changed

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        x_next = x  #changed
        if self.args.cuda:
            x_next = x_next.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x_next, W_init)  #changed
            #print("Wi.size: ", Wi.size())
            #print("Wi: ", Wi[0,:,:,1])

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x_next])[1])  #changed
            x_next = x_new  #changed  #print("x next.size", x_next.size())  #x = torch.cat([x, x_new], 2)

        # if self.args.cuda:
        #    x_next = x_next.cuda()

        Wl = self.w_comp_last(x_next, W_init, False)  #changed
        #print("Wl.size", Wl.size())
        #print("Wl: ", Wl[0,:,:,1])
        out = self.layer_last([Wl, x_next])[1]  #changed
        #print("out.size", out.size())
        #print("this is the x before sending to models")
        #print(x_next)

        return out[:, 0, :], x_next, Wl

class GNN_PDE_GCN_omniglot(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_PDE_GCN_omniglot, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 6  # TODO: change to 2 - here we change the number of layers
        for i in range(self.num_layers):
            if i >= 1:  #changed
                flag1 = 0
                flag2 = 1
            else:
                flag1 = 1
                flag2 = 0
            module_w = Wcompute(flag1 * self.input_features + flag2 * int(nf / 2),
                                flag1 * self.input_features + flag2 * int(nf / 2), operator='J2', activation='softmax',
                                ratio=[2, 1.5, 1, 1], drop=False)
            module_l = Gconv(flag1 * self.input_features + int(nf / 2) * flag2, int(nf / 2), 2)  #changed
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        #self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2)*self.num_layers,
        #                           self.input_features + int(self.nf / 2) * (self.num_layers - 1), #changed
        #                          operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        #self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=True) #changed
        self.w_comp_last = Wcompute(int(self.nf / 2),  #this is the last layer without concatenating
                                    int(self.nf / 2),  #changed
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        self.layer_last = Gconv(int(self.nf / 2), args.train_N_way, 2, bn_bool=True)  #changed

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        x_next = x  #changed
        if self.args.cuda:
            x_next = x_next.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x_next, W_init)  #changed
            #print("Wi.size: ", Wi.size())
            #print("Wi: ", Wi[0,:,:,1])

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x_next])[1])  #changed
            x_next = x_new  #changed  #print("x next.size", x_next.size())  #x = torch.cat([x, x_new], 2)

        # if self.args.cuda:
        #    x_next = x_next.cuda()

        Wl = self.w_comp_last(x_next, W_init)  #changed
        #print("Wl.size", Wl.size())
        #print("Wl: ", Wl[0,:,:,1])
        out = self.layer_last([Wl, x_next])[1]  #changed
        #print("out.size", out.size())
        #print("this is the x before sending to models")
        #print(x_next)

        return out[:, 0, :], x_next, Wl



# class GNN_nl_omniglot(nn.Module):
#     def __init__(self, args, input_features, nf, J):
#         super(GNN_nl_omniglot, self).__init__()
#         self.args = args
#         self.input_features = input_features
#         self.nf = nf
#         self.J = J
#
#         self.num_layers = 2
#         for i in range(self.num_layers):
#             module_w = Wcompute(self.input_features + int(nf / 2) * i,
#                                 self.input_features + int(nf / 2) * i,
#                                 operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=False)
#             module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
#             self.add_module('layer_w{}'.format(i), module_w)
#             self.add_module('layer_l{}'.format(i), module_l)
#
#         self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers,
#                                     self.input_features + int(self.nf / 2) * (self.num_layers - 1),
#                                     operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
#         self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=True)
#
#     def forward(self, x):
#         W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
#         if self.args.cuda:
#             W_init = W_init.cuda()
#
#         for i in range(self.num_layers):
#             Wi = self._modules['layer_w{}'.format(i)](x, W_init)
#
#             x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
#             x = torch.cat([x, x_new], 2)
#
#         Wl = self.w_comp_last(x, W_init)
#         out = self.layer_last([Wl, x])[1]
#
#         return out[:, 0, :]


class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        if args.dataset == 'mini_imagenet':
            self.num_layers = 2
        else:
            self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=False)

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl=self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]

class GNN_active(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_active, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers // 2):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)

            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.conv_active = nn.Conv1d(self.input_features + int(nf / 2) * 1, 1, 1, bias=False)
        nn.init.uniform_(self.conv_active.weight.data)

        for i in range(int(self.num_layers/2), self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=False)

    def active(self, x, oracles_yi, hidden_labels):
        '''
        :param x: torch.Size([40, 26, 181])
        :param oracles_yi: torch.Size([40, 26, 5])
        :param hidden_labels: torch.Size([40, 26])
        :return:
        '''


        x_active = torch.transpose(x, 1, 2)
        x_to_classify = x_active[:, :, 0:1]

        x_active = - ((x_active - x_to_classify) ** 2).detach()
        x_active = self.conv_active(x_active)
        x_active = torch.transpose(x_active, 1, 2)
        x_active = x_active.squeeze(-1)  # torch.Size([40, 26])

        if self.args.active_random == 1:
            x_active.data.fill_(1. / x_active.size(1))

        # assigning lower prob to uncover the labels we already know
        x_active = x_active - (1 - hidden_labels) * 1e8

        if self.args.active_random == 1:
            mapping = F.gumbel_softmax(x_active, hard=True).unsqueeze(-1)
            mapping = mapping.detach()
        else:
            if self.training:
                mapping = F.gumbel_softmax(x_active, hard=True).unsqueeze(-1)
            else:
                temperature = 1e5  # larger temperature at test to pick the most likely
                mapping = F.gumbel_softmax(x_active * temperature, hard=True).unsqueeze(-1)


        label2add = oracles_yi * mapping

        # add ppadding
        padd = torch.zeros(x.size(0), x.size(1), x.size(2) - label2add.size(2))
        padd = Variable(padd).detach()
        if self.args.cuda:
            padd = padd.cuda()
        label2add = torch.cat([label2add, padd], 2)
        x = x + label2add
        return x

    def forward(self, x, oracles_yi, hidden_labels):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers // 2):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        x = self.active(x, oracles_yi, hidden_labels)

        for i in range(int(self.num_layers/2), self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)


        Wl=self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]

if __name__ == '__main__':
    # test modules
    num_features = 64
    bs =  50
    nf = 10
    num_layers = 5
    N = 5
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
    import argparse

    parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
    parser.add_argument('--exp_name', type=str, default='debug_vx', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=50, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--batch_size_test', type=int, default=50, metavar='batch_size', help='Size of batch)')
    # parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
    #                     help='Size of batch)')
    # parser.add_argument('--batch_size_test', type=int, default=10, metavar='batch_size',
    #                     help='Size of batch)')
    parser.add_argument('--iterations', type=int, default=2500, metavar='N', help='number of epochs to train ')
    # parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
    #                     help='Learning rate decay interval')
    parser.add_argument('--decay_interval', type=int, default=10000, metavar='N', help='Learning rate decay interval')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')  # LR for Omniglot
    # parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
    #                     help='learning rate (default: 0.01)') # LR for MiniImagenet
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=300000, metavar='N',
                        help='how many batches between each model saving')
    parser.add_argument('--test_interval', type=int, default=250, metavar='N',
                        help='how many batches between each test')
    parser.add_argument('--test_N_way', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--train_N_way', type=int, default=5, metavar='N',
                        help='Number of classes for doing each training comparison')
    parser.add_argument('--test_N_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=1, metavar='N', help='Number of shots when training')
    parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N', help='Number of shots when training')
    parser.add_argument('--metric_network', type=str, default='gnn_iclr_nl', metavar='N',
                        help='gnn_iclr_nl' + 'gnn_iclr_active')
    parser.add_argument('--active_random', type=int, default=0, metavar='N', help='random active ? ')
    parser.add_argument('--dataset_root', type=str, default='datasets', metavar='N', help='Root dataset')
    parser.add_argument('--test_samples', type=int, default=30000, metavar='N', help='Number of shots')
    # parser.add_argument('--dataset', type=str, default='mini_imagenet', metavar='N',
    #                     help='omniglot')
    parser.add_argument('--dataset', type=str, default='omniglot', metavar='N', help='omniglot')
    # parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
    #                     help='Decreasing the learning rate every x iterations')
    # parser.add_argument('--dec_lr', type=int, default=1000, metavar='N',
    #                     help='Decreasing the learning rate every x iterations')
    parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                        help='Decreasing the learning rate every x iterations')
    args = parser.parse_args(args=[])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    ######################## test PDE-GCN ##############################
    bs = 10
    N = 5
    num_features = 64

    x = torch.ones((bs, N, num_features))

    input = Variable(x)
    gnn = PDE_GCN(args, num_features, 10, J)
    out = gnn(input)
    print(out[0].size())


# D/torch.permute(torch.std(D, (1,2)).repeat(5,5,1),(2,0,1)) # TODO: For Renana
