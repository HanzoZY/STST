import torch
import torch.nn as nn
import math
import numpy as np
from model.pretext_task import MaskedPrediction, ContrastiveLearning_SimSiam, Joint_Prediction, ReversePrediction, JigsawPrediction_T
import itertools
EPS = 1e-8


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


def get_mask_array(size_in, divide_rate= [0.8,0.1,0.1]):
    '''
    generate mask for each subset masks of batch data, masks[0] represents the index of unmasked samples,

    Input:
         size_in: the length of mask array
         divide_rate: the first: the rate of remain; the second: the rate of set empty(0); the third: the rate of noise
    Returns:
         N masks, each mask has the size :

    '''
    chosen_list = []
    for i in divide_rate:
        chosen_list.append(int(size_in*i))
    new_array = np.zeros(size_in)
    flag = 0
    for idx, num in enumerate(chosen_list):
        new_array[flag:flag+num] = idx
        flag = flag+num
    np.random.shuffle(new_array)
    map_clip = []
    for idx in range(len(divide_rate)):
         map_clip.append((new_array==idx).astype(int))

    return np.stack(map_clip)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEmbedding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain
        if domain == "temporal":
            self.PE = nn.Parameter(torch.zeros(channel, self.time_len))
        elif domain == "spatial":
            self.PE = nn.Parameter(torch.zeros(channel, self.joint_num))
        nn.init.uniform_(self.PE)

    def forward(self, x):  # nctv
        if self.domain == "spatial":
            pe = self.PE.unsqueeze(1).unsqueeze(0)
        else:
            pe = self.PE.unsqueeze(-1).unsqueeze(0)
        x = x + pe
        return x


class STST_Block(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, attention_head_S=3, attention_head_T=2, num_node=25,
                 num_frame=32, parallel=False, S_atten='free', T_atten='context',
                 kernel_size=1, stride=1, glo_reg_s=True, directed=True, TCN=True,
                 attentiondrop=0):
        super(STST_Block, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.attention_head_S = attention_head_S
        self.attention_head_T = attention_head_T
        self.glo_reg_s = glo_reg_s
        self.directed = directed
        self.TCN = TCN
        self.parallel = parallel
        self.S_atten = S_atten
        self.T_atten = T_atten

        backward_mask = torch.triu(torch.ones(num_frame, num_frame))
        self.register_buffer('backward_mask', backward_mask)

        pad = int((kernel_size - 1) / 2)

        #  S-Block
        atts = torch.zeros((1, self.attention_head_S, num_node, num_node))
        self.alphas = nn.Parameter(torch.ones(1, self.attention_head_S, 1, 1), requires_grad=True)
        if glo_reg_s:
            self.attention0s = nn.Parameter(torch.ones(1, self.attention_head_S, num_node, num_node) / num_node,
                                            requires_grad=True)
        self.register_buffer('atts', atts)
        self.in_nets = nn.Conv2d(in_channels, 2 * self.attention_head_S * inter_channels, 1, bias=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * self.attention_head_S, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        #  T-Block
        if self.directed == True:
            self.alphat_b = nn.Parameter(torch.ones(1, self.attention_head_T, 1, 1), requires_grad=True)
            self.alphat_f = nn.Parameter(torch.ones(1, self.attention_head_T, 1, 1), requires_grad=True)
        else:
            self.alphat = nn.Parameter(torch.ones(1, 2 * self.attention_head_T, 1, 1), requires_grad=True)

        if self.parallel == True:
            self.in_nett = nn.Conv2d(in_channels, 4 * self.attention_head_T * inter_channels, 1, bias=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(in_channels * self.attention_head_T * 2, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.in_channels_t = in_channels
        else:
            self.in_nett = nn.Conv2d(out_channels, 4 * self.attention_head_T * inter_channels, 1, bias=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * self.attention_head_T * 2, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.in_channels_t = out_channels

        self.ff_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
            nn.BatchNorm2d(out_channels),
        )


        if self.TCN==True:
            self.out_nett_extend = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels), )
        if in_channels != out_channels or stride != 1:
            self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            self.downt1 = nn.Sequential(
                nn.Conv2d(self.in_channels_t, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(self.in_channels_t, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if self.TCN==True:
                self.downtTCN = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt1 = lambda x: x
            self.downt2 = lambda x: x
            if self.TCN==True:
                self.downtTCN = lambda x: x

        # self.ff_net_all = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
        #     nn.BatchNorm2d(out_channels),
        # )

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()

        attention = self.atts
        if self.glo_reg_s:
            attention = attention + self.attention0s

        if self.S_atten == 'free':
            attention = attention.unsqueeze(2)
            alphas = self.alphas.unsqueeze(2)
        else:
            alphas = self.alphas


        y = x
        if self.S_atten == 'context_new':
            q, k = torch.chunk(self.in_nets(torch.mean(input=y,dim=-2,keepdim=True)).view(N, 2 * self.attention_head_S, self.inter_channels, V), 2,
                               dim=1)  # nctv -> n num_subset c'tv
        else:
            q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.attention_head_S, self.inter_channels, T, V), 2,
                           dim=1)  # nctv -> n num_subset c'tv
        if self.S_atten == 'context' or self.S_atten == 'context_new':
            if self.S_atten == 'context':
                q = q.mean(-2)
                k = k.mean(-2)
            attention = attention + self.tan(
                torch.einsum('nscu,nscv->nsuv', [q, k]) / (self.inter_channels)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.attention_head_S * self.in_channels,
                                                                                   T, V)
        elif self.S_atten == 'avg':
            attention = attention + self.tan(
                torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.attention_head_S * self.in_channels,
                                                                                   T, V)
        else:
            assert self.S_atten == 'free'
            attention = attention + self.tan(
                torch.einsum('nsctu,nsctv->nstuv', [q, k]) / (self.inter_channels)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nstuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.attention_head_S * self.in_channels,
                                                                                   T, V)

        y = self.out_nets(y)  # nctv
        y = self.relu(self.downs1(x) + y)
        y = self.ff_nets(y)
        s_out = self.relu(self.downs2(x) + y)


        # set_trace()
        # y_1 = self.out_nett_extend(y)
        # y_1 = self.relu(self.downt3(y) + y_1)
        # y = y_1
        if self.parallel:
            t_in = x
        else:
            t_in = s_out
        z = t_in
        if self.directed == True:
            forward_mask = self.backward_mask.transpose(-1, -2)
            backward_mask = self.backward_mask
            if self.T_atten == 'context_new':
                q_k_in = self.in_nett(torch.mean(input=z,dim=-1,keepdim=True)).view(N, 4 * self.attention_head_T, self.inter_channels, T)
            else:
                q_k_in = self.in_nett(z).view(N, 4 * self.attention_head_T, self.inter_channels, T, V)
            if self.T_atten == 'context':
                q_k_in = q_k_in.mean(-1)
            q_f, q_b, k_f, k_b = torch.chunk(q_k_in, 4, dim=1)

            if self.T_atten == 'context' or self.T_atten == 'context_new':
                attention_b = self.tan(torch.einsum('nsct,nscq->nstq', [q_b, k_b]) / (self.inter_channels)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsct,nscq->nstq', [q_f, k_f]) / (self.inter_channels)) * self.alphat_f
                attention_b = torch.einsum('nstq,tq->nstq', [attention_b, backward_mask])
                attention_f = torch.einsum('nstq,tq->nstq', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstq->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstq->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
            elif self.T_atten == 'avg':
                attention_b = self.tan(torch.einsum('nsctv,nscqv->nstq', [q_b, k_b]) / (self.inter_channels * V)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsctv,nscqv->nstq', [q_f, k_f]) / (self.inter_channels * V)) * self.alphat_f
                attention_b = torch.einsum('nstq,tq->nstq', [attention_b, backward_mask])
                attention_f = torch.einsum('nstq,tq->nstq', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstq->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstq->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
            else:
                assert self.T_atten == 'free'
                attention_b = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q_b, k_b]) / (self.inter_channels)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q_f, k_f]) / (self.inter_channels)) * self.alphat_f
                attention_b = torch.einsum('nstqv,tq->nstqv', [attention_b, backward_mask])
                attention_f = torch.einsum('nstqv,tq->nstqv', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstqv->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstqv->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.attention_head_T * self.in_channels_t, T, V)
            z = torch.cat([z_f, z_b], dim=-3)
        else:
            num_head_temp = 2 * self.attention_head_T
            if self.T_atten == 'context_new':
                q_k_in = self.in_nett(torch.mean(input=z,dim=-1,keepdim=True)).view(N, 2 * num_head_temp, self.inter_channels, T)
            else:
                q_k_in = self.in_nett(z).view(N, 2 * num_head_temp, self.inter_channels, T, V)
            if self.T_atten == 'context':
                q_k_in = q_k_in.mean(-1)
            q, k = torch.chunk(q_k_in, 2, dim=1)
            if self.T_atten == 'context' or self.T_atten == 'context_new':
                attention_t = self.tan(torch.einsum('nsct,nscq->nstq', [q, k]) / (self.inter_channels)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstq->nscqv', [y, attention_t]).contiguous().view(N, num_head_temp * self.in_channels_t, T, V)
            elif self.T_atten == 'avg':
                attention_t = self.tan(torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstq->nscqv', [y, attention_t]).contiguous().view(N, num_head_temp * self.in_channels_t, T, V)
            else:
                assert self.T_atten == 'free'
                attention_t = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q, k]) / (self.inter_channels)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstqv->nscqv', [y, attention_t]).contiguous().view(N, num_head_temp * self.in_channels_t, T, V)



        z = self.out_nett(z)  # nctv
        z = self.relu(self.downt1(t_in) + z)
        z = self.ff_nett(z)
        t_out = self.relu(self.downt2(t_in) + z)

        if self.parallel:
            z = s_out + t_out
        else:
            z = t_out
        # set_trace()
        if self.TCN==True:
            z_1 = self.out_nett_extend(z)
            z_1 = self.relu(self.downtTCN(z) + z_1)
            z = z_1
        return z


class STST_option(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, attention_head_S=3, attention_head_T=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, mask_divide=[0.8, 0.1, 0.1],
                 var=0.15, use_SSL=False, num_seg=3, directed=False, TCN=True,
                 attentiondrop=0, dropout2d=0, parallel=False, S_atten='free', T_atten='context', use_pet=True, use_pes=True,
                 SSL_weight={'PC': 0.1, 'PS': 0.1, 'PT': 0.1, 'RT': 0.1, 'CL': 0.1},
                 SSL_option={'PC': True, 'PS': True, 'PT': False, 'RT': False, 'CL': True}):
        super(STST_option, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.var = var
        self.use_SSL = use_SSL
        self.SSL_weight = SSL_weight
        self.SSL_option = SSL_option
        self.mask_divide = mask_divide
        self.num_seg = num_seg
        self.in_channels = in_channels
        self.num_person = num_person
        self.num_point = num_point
        self.num_frame = num_frame
        if use_pet:
            self.pet = PositionalEncoding(in_channels, num_point, num_frame, 'temporal')
        else:
            self.pet = lambda x: x
        if use_pes:
            self.pes = PositionalEmbedding(in_channels, num_point, num_frame, 'spatial')
        else:
            self.pes = lambda x: x
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'attention_head_S': attention_head_S,
            'attention_head_T': attention_head_T,
            'glo_reg_s': glo_reg_s,
            'attentiondrop': attentiondrop,
            'directed': directed,
            'TCN': TCN,
            'parallel': parallel,
            'S_atten': S_atten,
            'T_atten': T_atten
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STST_Block(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Linear(self.out_channels, num_class)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        self.init_SSL()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

        self.SSL_mask = MaskedPrediction(hidden=self.out_channels, num_person=self.num_person, reconstruct=num_channel)
        self.SSL_JigsawT = JigsawPrediction_T(hid_dim=self.out_channels, num_perm=len(self.permutations_T))
        self.SSL_JointP = Joint_Prediction(hid_dim=self.out_channels, num_joints=self.num_point)
        self.SSL_ReverseT = ReversePrediction(hidden = self.out_channels)
        self.SSL_Contra = ContrastiveLearning_SimSiam(hid_dim=self.out_channels)


    def init_SSL(self):
        # self.use_SSL = self.use_SSL and (np.array(list(self.SSL_option.values())) == True).any()
        self.use_SSL = self.use_SSL
        self.init_jigsaw()
        self.init_loss()

    def init_jigsaw(self):
        # initialize permutation for temporal dimension
        temp_list_T = list(range(self.num_seg))
        self.permutations_T = list(itertools.permutations(temp_list_T))

    def init_loss(self):
        pretext_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('pretext_loss_init', pretext_loss_init)
        mask_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('mask_loss_init', mask_loss_init)
        joint_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('joint_loss_init', joint_loss_init)
        simloss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('simloss_init', simloss_init)
        jigsaw_T_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('jigsaw_T_loss_init', jigsaw_T_loss_init)
        reverse_T_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('reverse_T_loss_init', reverse_T_loss_init)


    def Jigsaw_T_generate_labeled(self,x):
        N, M, C, T, V = x.shape
        idx = list(range(T))
        cut_num = int(T/self.num_seg)
        cut_idx = np.array([idx[i*cut_num:(i+1)*cut_num] if i < self.num_seg-1 else idx[i*cut_num:] for i in range(self.num_seg)])
        x_list = []
        num_perm = len(self.permutations_T)
        labels = np.random.randint(low=0, high=num_perm, size=N, dtype='l')
        labels_T = torch.from_numpy(labels)
        labels_T.requires_grad = False
        # labels_T = labels_T.to(x.get_device())
        for i, label in enumerate(labels):
            idx_i_permute = cut_idx[list(self.permutations_T[label])].tolist()
            idx_i = [j for k in idx_i_permute for j in k]
            sample_temp = x[i]
            x_list.append(sample_temp[:,:,idx_i,:])
        x_T = torch.stack(x_list)
        assert N == x_T.size(0)
        x_T = x_T.view(N, M, C, T, V)

        return x_T, labels_T


    def reverse_T_generate(self, x):
        N, C, T, V = x.shape
        assert T == self.num_frame
        x_reverse = torch.flip(x, dims=[-2])
        assert x[0,0,0,0] == x_reverse[0,0,-1,0]
        return x_reverse



    def random_mask_all(self,x):
        '''
        all samples share the same mask and noise
        Args:
            x: all input data

        Returns:

        '''
        N, C, T, V = x.shape
        noise = torch.FloatTensor(*x.size()[2:]).uniform_(-self.var, self.var).unsqueeze(0)
        masks = torch.tensor(data=get_mask_array(size_in=V*T, divide_rate=self.mask_divide),dtype=torch.float)

        # noise = noise.to(x.get_device())
        # masks = masks.to(x.get_device())

        x_masked = x
        num_mask = masks.size(0)
        assert num_mask==len(self.mask_divide)
        for i, mask in enumerate(masks.chunk(num_mask, dim=0)):
            if i==0:
                continue
            if i==1:
                mask = 1-mask.view(T,V).unsqueeze(0).unsqueeze(0)
                x_masked = x_masked * mask
            if i==2:
                mask_noise = mask.view(T,V).unsqueeze(0).unsqueeze(0) * noise
                x_masked = x_masked + mask_noise

        return x_masked

    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape
        pretext_loss = self.pretext_loss_init.expand(N)
        mask_loss = self.mask_loss_init.expand(N)
        jigsaw_T_loss = self.jigsaw_T_loss_init.expand(N)
        joint_loss = self.joint_loss_init.expand(N)
        reverse_loss = self.reverse_T_loss_init.expand(N)
        simloss = self.simloss_init.expand(N)




        # set_trace()
        if self.use_SSL == True and self.training == True:
            x_origin = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
            if self.SSL_option['PC'] == True:
                x_masked = self.random_mask_all(x=x_origin).detach()
                x_masked = x_masked.view(N * M, C, T, V)

            x_temp = self.input_map(x_origin)
            x = self.pet(self.pes(x_temp))
            # for the PC task, mask and predict
            if self.SSL_option['PC'] == True:
                x_masked = self.input_map(x_masked)
                x_masked = self.pet(self.pes(x_masked))

            # for the PT task, random permutate and predict
            if self.SSL_option['PT'] == True:
                x_jigsaw_T, labels_T = self.Jigsaw_T_generate_labeled(x=x_temp.view(N, M, self.in_channels, T, V))
                x_jigsaw_T = x_jigsaw_T.view(N * M, self.in_channels, T, V)
                x_jigsaw_T = self.pet(self.pes(x_jigsaw_T))

            # for the PS task,
            if self.SSL_option['PS'] == True:
                x_predict_S = self.pet(x_temp)
            if self.SSL_option['RT'] == True:
                x_reverse = self.reverse_T_generate(x_temp.view(N*M, self.in_channels, T, V))
                x_reverse = x_reverse.view(N * M, self.in_channels, T, V)
                x_reverse = self.pet(self.pes(x_reverse))

            for i, m in enumerate(self.graph_layers):
                x = m(x)
                if self.SSL_option['PC'] == True:
                    x_masked = m(x_masked)
                if self.SSL_option['PT'] == True:
                    x_jigsaw_T = m(x_jigsaw_T)
                if self.SSL_option['PS'] == True:
                    x_predict_S = m(x_predict_S)
                if self.SSL_option['RT'] == True:
                    x_reverse = m(x_reverse)


            x_all = []
            # the PC task, predict the masked coordinates
            if self.SSL_option['PC'] == True:
                mask_loss = self.SSL_mask(x_origin=x_origin, x_masked=x_masked)
                x_masked = x_masked.view(N, M, self.out_channels, T, V)
                x_masked = x_masked.mean(-1).mean(-1).mean(1).view(N,self.out_channels).unsqueeze(0)
                x_all.append(x_masked)

            # the PT task, predict the Jigsaw T labeled'''
            if self.SSL_option['PT'] == True:
                x_jigsaw_T = x_jigsaw_T.view(N, M, self.out_channels, T*V)
                x_jigsaw_T = x_jigsaw_T.mean(-1).mean(1).view(N, self.out_channels)
                jigsaw_T_loss = self.SSL_JigsawT(x=x_jigsaw_T, labels_T=labels_T)
                x_jigsaw_T = x_jigsaw_T.unsqueeze(0)
                x_all.append(x_jigsaw_T)

            # the PS task, predict the type of joint
            if self.SSL_option['PS'] == True:
                x_predict_S = x_predict_S.view(N, M, self.out_channels, T, V)
                x_predict_S = x_predict_S.mean(-2).mean(1).view(N, self.out_channels, V)
                joint_loss = self.SSL_JointP(x=x_predict_S)
                x_predict_S = x_predict_S.mean(-1).view(N, self.out_channels).unsqueeze(0)
                x_all.append(x_predict_S)

            # Distinguish the reversed stream
            if self.SSL_option['RT'] == True:
                x_reverse = x_reverse.view(N, M, self.out_channels, T*V)
                x_reverse = x_reverse.mean(-1).mean(1).view(N, self.out_channels)
                reverse_loss = self.SSL_ReverseT(x=x.view(N, M, self.out_channels, T*V).mean(-1).mean(1).view(N, self.out_channels), x_reverse=x_reverse)
                x_all.append(x_reverse.unsqueeze(0))

            #  Contrastive Learning
            if self.SSL_option['CL'] == True:
                x4SSL = x.view(N, M, self.out_channels, T*V).mean(-1).mean(1).view(N, self.out_channels).unsqueeze(0)
                x_all.append(x4SSL)
                x_all = torch.cat(x_all,dim=0)
                simloss = self.SSL_Contra(x_all)



            # downstream task, the action reconition task
            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel

            pretext_loss = pretext_loss + self.SSL_weight['PC'] * mask_loss + self.SSL_weight['PS'] * joint_loss + \
                           self.SSL_weight['PT'] * jigsaw_T_loss + self.SSL_weight['RT'] * reverse_loss + self.SSL_weight['CL'] * simloss
        else:
            # for the inference precess
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x = self.input_map(x)
            x = self.pet(self.pes(x))
            # x = self.pet(x)

            for i, m in enumerate(self.graph_layers):
                x = m(x)

            # NM, C, T, V

            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel



        return self.fc(x), pretext_loss, mask_loss, jigsaw_T_loss, joint_loss, reverse_loss, simloss


if __name__ == '__main__':
    get_mask_array(20)

