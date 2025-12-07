import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_weight(conf):
    if conf['weight_type'] == 'linear1':
        weight = Linear(conf['c'])
    if conf['weight_type'] == 'linear2':
        weight = Linear2(conf['c'],conf['prior'])
        #weight = Linear2(conf['c'])
    if conf['weight_type'] == 'linear3':
        weight = Linear3(conf['c'],conf['prior'])
    if conf['weight_type'] == 'linear4':
        weight = Linear4(conf['c'],conf['prior'])
    if conf['weight_type'] == 'linear5':
        weight = Linear5(conf['c'],conf['prior'])
    if conf['weight_type'] == 'linear6':
        weight = Linear6(conf['c'],conf['prior'])
    if conf['weight_type'] == 'separate':
        weight = Separate(conf['c'],conf['prior'])
    if conf['weight_type'] == 'separate2':
        weight = Separate2(conf['c'],conf['prior'])
    if conf['weight_type'] == 'separate3':
        weight = Separate3(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'separate4':
        weight = Separate4(conf['c'],conf['prior'])
    if conf['weight_type'] == 'same':
        weight = Same(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same2':
        weight = Same2(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same3':
        weight = Same3(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same5':
        weight = Same5(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same6':
        weight = Same6(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same7':
        weight = Same7(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same8':
        weight = Same8(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same9':
        weight = Same9(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same10':
        weight = Same10(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same11':
        weight = Same11(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same12':
        weight = Same12(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same13':
        weight = Same13(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same14':
        weight = Same14(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'same15':
        weight = Same15(conf['c'],conf['alpha'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'both':
        weight = Both_nonoverlap(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'both2':
        weight = Both_nonoverlap2(conf['c'],conf['prior'],conf['val'])
    if conf['weight_type'] == 'matrix':
        weight = Matrix(conf['pos_type'])
    if conf['weight_type'] == 'triangular':
        weight = Triangular(conf['pos_type'])
    return weight


class Linear(nn.Module):

    def __init__(self,c):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(1600, c)
        self.fc2 = nn.Linear(c, 1600, bias=False)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        pho_matrix = pho_matrix.reshape(-1,1600)
        non_zero_counts = torch.sum(pho_matrix != 0, dim=-1)
        weight = self.fc2(self.tanh(self.fc1(pho_matrix)))
        weighted_pho_matrix = torch.sum(weight*pho_matrix,dim=-1)/(non_zero_counts+1e-6)

        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis
class Linear2(nn.Module):

    def __init__(self,c,prior=False):
        super(Linear2, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc0 = nn.Linear(2, 1)
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0])))
        self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(1600).unsqueeze(0).unsqueeze(-1)

        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        non_zero_counts = torch.sum(pho_matrix.reshape(K*K,1600)!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.reshape(K*K,1600).unsqueeze(-1)
        weight = pho_matrix
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.repeat(K*K,1,1)),dim=-1)
            weight = self.relu(self.fc0(weight))
        weight = self.fc2(self.tanh(self.fc1(weight)))
        weighted_pho_matrix = torch.sum((weight.reshape(K*K,1600))*(pho_matrix.reshape(K*K,1600)),dim=-1)/non_zero_counts
        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis  

class Linear3(nn.Module):

    def __init__(self,c,prior=False):
        super(Linear3, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0])))
        self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(1600).unsqueeze(0).unsqueeze(-1)

        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        non_zero_counts = torch.sum(pho_matrix.reshape(K*K,1600)!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.reshape(K*K,1600).unsqueeze(-1)
        weight = pho_matrix
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.repeat(K*K,1,1)),dim=-1)
        weight = self.fc2(self.tanh(self.fc1(weight)))
        weighted_pho_matrix = torch.sum((weight.reshape(K*K,1600))*(pho_matrix.reshape(K*K,1600)),dim=-1)/non_zero_counts
        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis  
class Both_nonoverlap2(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Both_nonoverlap2, self).__init__()
        self.prior = prior

        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if val=='training':
            mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
            self.weight=self.weight.reshape(40).unsqueeze(0)
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        else:
            mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.mask = torch.ones(40, 40, requires_grad=False).cuda()-torch.eye(40, requires_grad=False).cuda()
        self.pho_weight = nn.Parameter(torch.FloatTensor(40))
        self.pos = Norm()
    def forward(self, enroll, test, infer=False):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pos(self.pho_weight).unsqueeze(0).unsqueeze(0).repeat(K,K,1)
        print(self.pho_weight)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho*positive_weight!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.relu(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.relu(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts
        
        cross_pho = (self.mask.unsqueeze(0).unsqueeze(0).repeat(K,K,1,1))*pho_matrix
        non_zero_counts = torch.sum(cross_pho.reshape(K*K,1600)!= 0, dim=-1)+1e-6
        cross_weight = self.fc4(self.tanh(self.fc3(cross_pho.reshape(K*K,1600).unsqueeze(-1))))
        cross_weight_pho = self.tanh((cross_weight.reshape(K*K,1600))*(cross_pho.reshape(K*K,1600)))
        cross_score = torch.sum(cross_weight_pho,dim=-1)/non_zero_counts
        combine = torch.cat((same_score.unsqueeze(-1),cross_score.reshape(K,K).unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)


        if infer:
            return combine, [same_weight.squeeze().detach().cpu().numpy(), cross_weight.squeeze().detach().cpu().numpy()], [same_pho.squeeze().detach().cpu().numpy(),pho_matrix.squeeze().detach().cpu().numpy()]
        return combine
class Both_nonoverlap(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Both_nonoverlap, self).__init__()
        self.prior = prior

        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if val=='training':
            mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
            self.weight=self.weight.reshape(40).unsqueeze(0)
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        else:
            mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.mask = torch.ones(40, 40, requires_grad=False).cuda()-torch.eye(40, requires_grad=False).cuda()
        
    def forward(self, enroll, test, infer=False):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.relu(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.relu(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        
        cross_pho = (self.mask.unsqueeze(0).unsqueeze(0).repeat(K,K,1,1))*pho_matrix
        non_zero_counts = torch.sum(cross_pho.reshape(K*K,1600)!= 0, dim=-1)+1e-6
        cross_weight = self.fc4(self.tanh(self.fc3(cross_pho.reshape(K*K,1600).unsqueeze(-1))))
        cross_weight_pho = self.tanh((cross_weight.reshape(K*K,1600))*(cross_pho.reshape(K*K,1600)))
        cross_score = torch.sum(cross_weight_pho,dim=-1)/non_zero_counts
        combine = torch.cat((same_score.unsqueeze(-1),cross_score.reshape(K,K).unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)
        if infer:
            return combine, [same_weight.squeeze().detach().cpu().numpy(), cross_weight.squeeze().detach().cpu().numpy()], [same_pho.squeeze().detach().cpu().numpy(),pho_matrix.squeeze().detach().cpu().numpy()]
        return combine

class Linear4(nn.Module):

    def __init__(self,c,prior=False):
        super(Linear4, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0])))
        self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(1600).unsqueeze(0).unsqueeze(-1)

        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        non_zero_counts = torch.sum(pho_matrix.reshape(K*K,1600)!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.reshape(K*K,1600).unsqueeze(-1)
        weight = pho_matrix
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.repeat(K*K,1,1)),dim=-1)

        weight = self.fc2(self.tanh(self.fc1(weight)))
        weighted_pho_matrix = self.tanh((weight.reshape(K*K,1600))*(pho_matrix.reshape(K*K,1600))) 
        weighted_pho_matrix = torch.sum(weighted_pho_matrix,dim=-1)/non_zero_counts
        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis  

class Linear5(nn.Module):

    def __init__(self,c,prior=False):
        super(Linear5, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc0 = nn.Linear(2, 1)
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0]))).cuda()
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(40).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        #weight = pho_matrix
        if self.prior:
            #print(pho_matrix.shape)
            #print(self.weight.shape)
            #quit()
            weight = torch.cat((pho_matrix,self.weight.unsqueeze(-1).repeat(K,K,1,40,1)),dim=-1)
            weight = self.relu(self.fc0(weight))
            weight = self.fc2(self.tanh(self.fc1(weight)))
        else:
            weight = self.fc2(self.tanh(self.fc1(pho_matrix)))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.repeat(K,K,1,1)),dim=-1)
            weight = self.relu(self.fc0(weight))
            weight = self.fc2(self.tanh(self.fc1(weight)))
        else:
            weight = self.fc2(self.tanh(self.fc1(pho_matrix)))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40)
        weighted_pho_dis = torch.sum(pho_matrix,dim=-1)/non_zero_counts
        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        #weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis  

class Linear6(nn.Module):

    def __init__(self,c,prior=False):
        super(Linear6, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc0 = nn.Linear(2, 1)
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0]))).cuda()
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(40).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        #self.weight = nn.Parameter(torch.FloatTensor(40, 1))
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        #weight = pho_matrix
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.unsqueeze(-1).repeat(K,K,1,40,1)),dim=-1)
            weight = self.relu(self.fc0(weight))
            weight = self.fc2(self.tanh(self.fc1(weight)))
        else:
            weight = self.fc2(self.tanh(self.fc1(pho_matrix)))
        pho_matrix = self.tanh(weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        if self.prior:
            weight = torch.cat((pho_matrix,self.weight.repeat(K,K,1,1)),dim=-1)
            weight = self.relu(self.fc0(weight))
            weight = self.fc2(self.tanh(self.fc1(weight)))
        else:
            weight = self.fc2(self.tanh(self.fc1(pho_matrix)))
        pho_matrix = self.tanh(weight*pho_matrix).reshape(K,K,40)
        weighted_pho_dis = torch.sum(pho_matrix,dim=-1)/non_zero_counts
        #weighted_pho_dis = torch.bmm(pho_matrix.reshape(-1,40,40),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,1]
            
        #weighted_pho_dis = torch.bmm(weighted_pho_dis.transpose(-1,-2),self.weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,1,1]
        #weighted_pho_dis = weighted_pho_matrix.reshape(K,K)
        
        return weighted_pho_dis  

class Separate(nn.Module):

    def __init__(self,c,prior=False):
        super(Separate, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0]))).cuda()
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(40).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        #same with linear5
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        weight = self.fc4(self.tanh(self.fc3(pho_matrix)))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)

        weight = self.fc4(self.tanh(self.fc3(pho_matrix)))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40)
        diff_score = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        combine = torch.cat((same_score.unsqueeze(-1),diff_score.unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)

        return combine

class Same(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test, infer=False,dele=None):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same2(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same2, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(40, 40, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test, infer=False):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(self.fc3(same_weight_pho),dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same3(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same3, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(40))
        self.pos = Norm()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pos(self.pho_weight).unsqueeze(0).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        non_zero_counts = torch.sum(condition*positive_weight, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same5(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same5, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False,dele=None):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        if dele is not None:
            same_pho[:,:,dele]=0
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same7(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same7, self).__init__()
        self.prior=prior
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6

        same_score = torch.sum(same_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,self.pho_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same6(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same6, self).__init__()
        self.prior = prior


        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        #self.pos = Norm2()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pho_weight.unsqueeze(0).repeat(K,K,1)

        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        #non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        non_zero_counts = torch.sum(condition, dim=-1)+1e-6

        #same_weight_pho = self.tanh(same_pho).reshape(K,K,40)
        same_score = torch.sum(same_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_pho.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same8(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same8, self).__init__()
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.relu(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same9(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Same9, self).__init__()
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm3()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Same10(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same10, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        #nn.init.xavier_uniform_(self.weight)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.pho_weight.unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]

        condition = same_pho!=0
    
        
        positive_weight = self.alpha*self.pos(positive_weight*condition)*condition
        non_zero_counts = torch.sum(positive_weight, dim=-1)+1e-6
        #print('non_zero_counts',torch.any(torch.isnan(non_zero_counts)))
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        #print('same_weight_pho',torch.any(torch.isnan(same_weight_pho)))
        same_score = torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts

        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same11(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same11, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False,dele=None):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        if dele is not None:
            same_pho[:,:,dele]=0
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = (same_weight*same_pho).reshape(K,K,40)
        same_score = self.tanh(torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts)
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
    
class Same14(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same14, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm3()
    def forward(self, enroll, test, infer=False,dele=None):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        if dele is not None:
            same_pho[:,:,dele]=0
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = (same_weight*same_pho).reshape(K,K,40)
        same_score = self.tanh(torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts)
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same15(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same15, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)

    def forward(self, enroll, test, infer=False,dele=None):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.relu(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        if dele is not None:
            same_pho[:,:,dele]=0
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = (same_weight*same_pho).reshape(K,K,40)
        same_score = self.tanh(torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts)
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same13(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same13, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if self.prior:
            if val=='training':
                mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
                self.weight=self.weight.reshape(40).unsqueeze(0)
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            else:
                mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
                self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
                max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
                min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
                self.weight = (self.weight-min)/(max-min)
            self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        
    def forward(self, enroll, test, infer=False,dele=None):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.sig(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        if dele is not None:
            same_pho[:,:,dele]=0
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = (same_weight*same_pho).reshape(K,K,40)
        same_score = self.tanh(torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts)
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score
class Same12(nn.Module):

    def __init__(self,c,alpha,prior=False,val='training'):
        super(Same12, self).__init__()
        self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.alpha=alpha
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softsign = nn.Softsign()
        self.tanh = nn.Tanh()
        self.pho_weight = nn.Parameter(torch.FloatTensor(1,40))
        nn.init.normal_(self.pho_weight, mean=0, std=1)
        self.pos = Norm2()
    def forward(self, enroll, test, infer=False):
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        positive_weight = self.alpha*self.pos(self.pho_weight).unsqueeze(0).repeat(K,K,1)
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        condition = same_pho!=0
        non_zero_counts = torch.sum(positive_weight*condition, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = (same_weight*same_pho).reshape(K,K,40)
        same_score = self.softsign(torch.sum(same_weight_pho*positive_weight,dim=-1)/non_zero_counts)
        if infer:
            return same_score,same_weight.squeeze().detach().cpu().numpy(), same_pho.squeeze().detach().cpu().numpy()
        return same_score

class Separate2(nn.Module):

    def __init__(self,c,prior=False):
        super(Separate2, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0]))).cuda()
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(40).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.tanh(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.tanh(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        
        indices = torch.arange(40)
        #pho_matrix[:, :, indices, indices] = 0
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        weight = self.fc4(self.tanh(self.fc3(pho_matrix)))
        pho_matrix = self.tanh(weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)

        weight = self.fc4(self.tanh(self.fc3(pho_matrix)))
        pho_matrix = self.tanh(weight*pho_matrix).reshape(K,K,40)
        diff_score = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        combine = torch.cat((same_score.unsqueeze(-1),diff_score.unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)
        return combine

class Separate3(nn.Module):

    def __init__(self,c,prior=False,val='training'):
        super(Separate3, self).__init__()
        self.prior = prior

        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if val=='training':
            mean = torch.mean(torch.from_numpy(np.load('exp/train_ratio.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/train_ratio.npy')).float(),mean)).cuda()
            self.weight=self.weight.reshape(40).unsqueeze(0)
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        else:
            mean = torch.mean(torch.from_numpy(np.load('exp/val.npy')).float()).unsqueeze(0)
            self.weight = torch.cat((torch.from_numpy(np.load('exp/val.npy')).float(),mean)).cuda()
            max = torch.amax(self.weight,dim=-1).unsqueeze(-1)
            min = torch.amin(self.weight,dim=-1).unsqueeze(-1)
            self.weight = (self.weight-min)/(max-min)
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test, infer=False):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.fc2(self.relu(self.fc1(same_weight)))
        else:
            same_weight = self.fc2(self.relu(self.fc1(same_pho)))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        weight = self.fc4(self.relu(self.fc3(pho_matrix)))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)

        weight = self.fc4(self.relu(self.fc3(pho_matrix)))
        pho_matrix_ = (weight*pho_matrix).reshape(K,K,40)
        diff_score = torch.sum(pho_matrix_,dim=-1)/non_zero_counts

        combine = torch.cat((same_score.unsqueeze(-1),diff_score.unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)
        if infer:
            return combine, [same_weight.squeeze().detach().cpu().numpy(), weight.squeeze().detach().cpu().numpy()], [same_pho.squeeze().detach().cpu().numpy(),pho_matrix.squeeze().detach().cpu().numpy()]
        return combine

class Separate4(nn.Module):

    def __init__(self,c,prior=False):
        super(Separate4, self).__init__()
        self.prior = prior
        if self.prior:
            self.fc1 = nn.Linear(2, c)
        else:
            self.fc1 = nn.Linear(1, c)
        self.fc2 = nn.Linear(c, 1, bias=False)
        self.fc3 = nn.Linear(1, c)
        self.fc4 = nn.Linear(c, 1, bias=False)
        self.fc5 = nn.Linear(2, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight = torch.cat((torch.from_numpy(np.load('exp/dscrmn.npy')).float(),torch.tensor([0]))).cuda()
        #self.weight = torch.matmul(self.weight.unsqueeze(1), self.weight.unsqueeze(0)).cuda()
        self.weight = self.weight.reshape(40).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    def forward(self, enroll, test):
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K = pho_matrix.shape[0]
        same_pho = pho_matrix.diagonal(dim1=-2, dim2=-1)#[B,B,40]
        non_zero_counts = torch.sum(same_pho!= 0, dim=-1)+1e-6
        same_pho = same_pho.unsqueeze(-1)
        if self.prior:
            same_weight = torch.cat((same_pho,self.weight.repeat(K,K,1,1)),dim=-1)
            same_weight = self.relu(self.fc2(self.relu(self.fc1(same_weight))))
        else:
            same_weight = self.relu(self.fc2(self.relu(self.fc1(same_pho))))

        same_weight_pho = self.tanh(same_weight*same_pho).reshape(K,K,40)
        same_score = torch.sum(same_weight_pho,dim=-1)/non_zero_counts
        
        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)
        weight = self.relu(self.fc4(self.relu(self.fc3(pho_matrix))))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40,40) 
        pho_matrix = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        non_zero_counts = torch.sum(pho_matrix!= 0, dim=-1)+1e-6
        pho_matrix = pho_matrix.unsqueeze(-1)

        weight = self.relu(self.fc4(self.relu(self.fc3(pho_matrix))))
        pho_matrix = (weight*pho_matrix).reshape(K,K,40)
        diff_score = torch.sum(pho_matrix,dim=-1)/non_zero_counts

        combine = torch.cat((same_score.unsqueeze(-1),diff_score.unsqueeze(-1)),dim=-1)
        combine = self.tanh(self.fc5(combine)).reshape(K,K)
        return combine

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self,weight):

        weight = (weight-torch.amin(weight))/(torch.amax(weight)-torch.amin(weight))
        return weight

class Norm2(nn.Module):
    def __init__(self):
        super(Norm2, self).__init__()
    def forward(self,weight):
        
        weight = (weight-torch.amin(weight,dim=-1).unsqueeze(-1))/(1e-6+torch.amax(weight,dim=-1).unsqueeze(-1)-torch.amin(weight,dim=-1).unsqueeze(-1))
        return weight

class Norm3(nn.Module):
    def __init__(self):
        super(Norm3, self).__init__()
    def forward(self,weight):
        weight = weight-torch.amin(weight,dim=-1).unsqueeze(-1)
        return weight
class Tri_norm(nn.Module):
    def __init__(self):
        super(Tri_norm, self).__init__()
    def forward(self,weight):
        non_zero_elements = weight[weight != 0]
        max_val = non_zero_elements.max()
        min_val = non_zero_elements.min()
        weight = (weight-min_val)/(max_val-min_val)
        return weight.triu()

class Matrix(nn.Module):

    def __init__(self,pos_type):
        super(Matrix, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(40,40))
        nn.init.uniform_(self.weight)
        if pos_type=='softplus':
            self.pos = nn.Softplus()
        elif pos_type=='sigmoid':
            self.pos = nn.Sigmoid()
        elif pos_type=='relu':
            self.pos = nn.ReLU()
        elif pos_type=='norm':
            self.pos = Norm()
        else:
            self.pos = nn.Identity()


    def forward(self, enroll, test):
        positive_weight = self.pos(self.weight)
        #positive_weight = self.weight
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)

        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K=pho_matrix.shape[0]
        weighted_pho_dis = (pho_matrix.reshape(-1,40,40))*(positive_weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,40]
            
        weighted_pho_dis = torch.sum(weighted_pho_dis,(-1,-2)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_dis.reshape(pho_matrix.shape[0],pho_matrix.shape[0])
        
        return weighted_pho_dis

class Triangular(nn.Module):

    def __init__(self,pos_type):
        super(Triangular, self).__init__()
        self.weight = nn.Parameter(torch.triu(torch.rand(40,40)))
        #nn.init.uniform_(self.weight)
        if pos_type=='norm':
            self.pos = Tri_norm()
        else:
            self.pos = nn.Identity()
    def forward(self, enroll, test):
        positive_weight = self.pos(self.weight.triu())
        positive_weight = positive_weight+positive_weight.T-torch.diag(positive_weight.diag())
        
        enroll = torch.nn.functional.normalize(enroll,p=2,dim=1)
        test = torch.nn.functional.normalize(test,p=2,dim=1)
        pho_matrix = torch.matmul(enroll.unsqueeze(1).transpose(-2,-1), test.unsqueeze(0))#[B,B,40,40]
        K=pho_matrix.shape[0]
        weighted_pho_dis = (pho_matrix.reshape(-1,40,40))*(positive_weight.unsqueeze(0).repeat(K**2,1,1)) #[B**2,40,40]
        weighted_pho_dis = torch.sum(weighted_pho_dis,(-1,-2)) #[B**2,1,1]
        weighted_pho_dis = weighted_pho_dis.reshape(pho_matrix.shape[0],pho_matrix.shape[0])        
        return weighted_pho_dis
