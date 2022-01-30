import torch.nn as nn
import torch
import torch.nn.functional as F

def save_grad(name):
    def hook(grad):
        tag = name+' grad is'
        print(tag, grad)
    return hook


class JigsawPrediction_T(nn.Module):
    """
    Designed for the PT task
    """

    def __init__(self, hid_dim, num_perm):
        """
        :param hidden: dimesion of hiden layer
        :param num_perm: Types of permutation
        """
        super(JigsawPrediction_T, self).__init__()
        self.num_perm = num_perm
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_perm))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, labels_T):
        """
        Args:
            x: batchsize, channel

        Returns:
            loss of each sample
        """
        N, C  = x.size()
        N_L= labels_T.size(0)
        assert N == N_L
        x = self.MLP(x)
        loss_Jigsaw_T = self.loss_func(input=x, target=labels_T)
        return loss_Jigsaw_T


class Joint_Prediction(nn.Module):
    """
    Desinged for the PS task
    """

    def __init__(self, hid_dim, num_joints):
        """
        :param hidden: dimesion of hiden layer
        """
        super(Joint_Prediction, self).__init__()
        self.num_joints = num_joints
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_joints))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        Joints_label = torch.tensor(list(range(self.num_joints)),dtype=torch.long)
        Joints_label.requires_grad = False
        self.register_buffer('Joints_label', Joints_label)

    def forward(self, x):
        """
        return the mean loss of each joint
        """
        N, C, V = x.size()
        assert V == self.num_joints
        x = x.permute(0, 2, 1).contiguous().view(N*V,C)
        x = self.MLP(x)
        label = self.Joints_label.unsqueeze(0).expand(N,-1).reshape(N*V)
        Joint_loss = self.loss_func(input=x, target=label).view(N,V)
        Joint_loss = Joint_loss.mean(1)
        return Joint_loss





class MaskedPrediction(nn.Module):
    """
    predicting origin coordinates from masked input sequence
    """

    def __init__(self, hidden, num_person, reconstruct=3):
        """
        :param hidden: dimension of hiden layer
        :reconstruct: the original channel of coordinates
        """
        super(MaskedPrediction, self).__init__()
        self.num_person = num_person
        self.reconstruct = reconstruct
        self.hid_dim = hidden
        self.MLP = nn.Sequential(nn.Conv2d(self.hid_dim, self.hid_dim, 1, 1, padding=0, bias=True),
                                 nn.BatchNorm2d(num_features=self.hid_dim),
                                 nn.LeakyReLU(), nn.Conv2d(self.hid_dim, reconstruct, 1, 1, padding=0, bias=True))
        self.loss_func = torch.nn.L1Loss(reduction='none')


    def forward(self, x_origin, x_masked):

        """
        Args:
            x_origin: the original coordinates of joints
            x_masked: the embedding of all joints including normal, masked, and noised

        Returns:
            the loss of maskprediction of each sample N \times 1
        """
        N_0,C,T,V = x_origin.size()
        assert C==self.reconstruct
        x_origin = x_origin.detach()
        x_masked = self.MLP(x_masked)
        loss_all = self.loss_func(input=x_masked, target=x_origin)
        loss_all = loss_all.view(-1, self.num_person*C*T*V)
        loss_all = loss_all.mean(-1)
        return loss_all




class ReversePrediction(nn.Module):
    """
    Predicting the direction of sample on temporal dimension
    2-class classification problem
    """
    def __init__(self, hidden):
        """
        :param hidden: dimesions of hiden layers
        """
        super(ReversePrediction, self).__init__()
        self.num_class = 2
        reverse_label = torch.tensor(list(range(self.num_class)), dtype=torch.long)
        self.register_buffer('reverse_label', reverse_label)

        self.hid_dim = hidden
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_class))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, x_reverse):
        """
        Args:
            x_origin: the original coordinates of joints
            x_masked: the embedding of all joints including normal, masked, and noised

        Returns:
            the loss of maskprediction of each sample N \times 1
        """
        N_0, C_0 = x.size()
        N_1, C_1 = x_reverse.size()
        assert N_0==N_1 and C_0==C_1
        x_all = torch.cat([x, x_reverse], dim=0)
        label = self.reverse_label.unsqueeze(-1).expand(-1,N_0).reshape(self.num_class*N_0)
        x_all = x_all.view(self.num_class*N_0, C_0)
        x_all = self.MLP(x_all)
        loss_reverse = self.loss_func(input=x_all, target=label)
        loss_reverse = loss_reverse.view(self.num_class, N_0)
        loss_reverse = loss_reverse.mean(0)

        return loss_reverse



class ContrastiveLearning_SimSiam(nn.Module):
    """
    SimSiam algorithm for more than two stream
    """

    def __init__(self, hid_dim):
        """
        :param hid_dim: output size of BERT model
        """
        super(ContrastiveLearning_SimSiam, self).__init__()
        self.hid_dim = hid_dim
        self.projection = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim),
                                        nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.hid_dim), nn.LeakyReLU(),
                                        nn.BatchNorm1d(num_features=self.hid_dim))
        self.prediction = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.hid_dim))

    def SimSiamLoss(self,p,z):
        K_p, N_p, C_p = p.size()
        K_z, N_z, C_z = z.size()
        z = z.detach()
        p = F.normalize(p, dim=-1)  # l2-normalize
        z = F.normalize(z, dim=-1)  # l2-normalize
        similarity_map = torch.einsum('pnc,znc->pznc', [p, z])
        simloss = -similarity_map.sum(-1).mean(0).mean(0)
        return simloss

    def forward(self, x):
        """
        return the mean loss of each generated sample
        """
        K, N, C = x.size()
        x = x.view(K*N,C)
        z = self.projection(x)
        p = self.prediction(z)
        z = z.view(K, N, C)
        p = p.view(K, N, C)
        simloss = self.SimSiamLoss(z=z, p=p)
        return simloss