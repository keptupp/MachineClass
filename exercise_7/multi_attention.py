import torch
import copy
import math
import numpy as np
from torch.autograd import Variable

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self,size,padding_idx,smoothing=0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)#KL散度，用于衡量两个分布之间的距离
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self,x,target):
        print("===========================")
        print(x.size())
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size -2 ))#b.fill_(0),b中的元素全部填充为0
        #scatter_()函数会修改原来的Tensor；作用是可以用来对标签进行one-hot编码；
        # 沿着列的方向（1）在索引给定的位置填充self.confidence
        true_dist.scatter_(1,target.data.unsequeeze(1),self.confidence)
        true_dist[:,self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)#输出的每一行为非零元素的索引
        if mask.dim() > 0:
            true_dist.index_fill(0,mask.squeeze(),0,0)
        self.true_dist = true_dist
        return self.criterion(x,Variable(true_dist,requires_grad=False))

#多头注意力机制中的attention
def attention(query,key,value,mask=None,dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    #首先取query的最后一维的大小，对应词嵌入维度
    d_k = query.size(-1)
    #当输入都是二维时，就是普通的矩阵乘法，当输入有多维时，把多出的一维作为batch提出来，其它
    #部分做矩阵乘法
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)#论文中的公式1
    #接着判断是否使用掩码张量
    if mask is not None:
        #mask中取值为True位置对应于scores的相应位置用"-1e-9"填充
        scores = scores.masked_fill(mask == 0,-1e9)
    p_attn = torch.nn.functional.sofmax(scores,dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn

#多头注意力机制
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        #在类的初始化时，会传入三个参数，h代表头数,d_model代表词嵌入的维度，dropout代表进行dropout操作时候置
        #0的概率，默认是0.1
        super(MultiHeadedAttention,self).__init__()
        #在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除
        #这是因为我们之后要给每个头分配等量的词特性，也就是embeeding_dim/head个
        assert d_model % h ==0
        #得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h
        #传入头数h
        self.h = h
        # 创建linear层，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用，为什么是四个呢，
        # 这是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = torch.nn.ModuleList([copy.deepcopy(torch.nn.Linear(d_model,d_model)) for _ in range(4)])
        #self.attn为None,它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        """
        前向逻辑函数,它输入参数有四个，前三个就是注意力机制需要的Q,K,V,最后一个是注意力机制中可能
        需要的Mask掩码张量，默认是None
        """
        if mask is not None:
            #Same mask applied to all h heads.
            #使用unsqueeze扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        #接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表多少条样本
        nbatchs = query.size(0)

        #1) Do all the linear projection in batch from d_model=> hxd_k
        #首先利用zip将输入Q，K，V与三个线性层组到一起，然后利用for循环，将输入Q,K,V分别传到线性层中，做完线性
        #变换后，开始为每个头分割输入，这里使用view方法对线性变换的结构进行维度重塑，多加了一个维度h代表头，这
        #样就意味着每个头可以获得一部分词特性组成的句子，其中的-1代表自适应维度，计算机会根据这种变换自动计算这
        #里的值，然后对第二维和第三维进行转置操作，为了让代表句子长度和词向量维度能够相邻，这样注意力机制才能找到
        #词义与句子位置的关系，从attention函数中可以看到，利用的是原始输入的倒数第一和第二维，这样我们就可以得到
        #每个头的输入。
        query,key,value = [l(x).view(nbatchs,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        #2) Apply attention on all the projected vectors in batch
        # 得到每个头经过全连接层的输出后，接下来将他们传入到attention中，这里我们调用前面所实现的attention函数，
        #同时也将mask和dropout传入其中
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

        #3) "Concat" using a view and apply a final linear.
        #通过多头注意力计算后，我们就得到了每个头计算结果组成的4维向量，我们需要将其转换为输入的形状以方便后续
        #的计算，因此这里开始进行第一步处理环节的逆操作，先对第二维和第三维进行转置，然后使用contiguous方法。
        #这个方法的作用就是能够让转置的张量应用view方法，否则将无法直接使用，所以，下一步就是使用view重塑形状，
        #变成和输入形状相同。
        x = x.transpose(1,2).contiguous().view(nbatchs,-1,self.h * self.d_k)
        #最后使用线性层列表中的最后一个线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)

#----------------------feed forward层----------------------#
# encoder和decoder中的每两层之间都会有个feed forward 层，其实就是简单的由两个前向全连接
#层组成，核心在于，Attention模块每个时间步的输出都整合了所有时间步的信息，而Feedward Layer每个
#时间步只是对自己的特征的一个进一步整合，与其他时间无关。

class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation"
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = torch.nn.Linear(d_model,d_ff)
        self.w_2 = torch.nn.Linear(d_ff,d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self,x):
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

#------------------位置编码器Positional Encodding----------------------#
#Positional Encodding 位置编码的作用是为模型提供当前时间步的前后出现顺序的信息。因为Transformer不像
#RNN那样的循环结构有前后不同时间输入的先后顺序，所有的时间步是同时输入，并行推理的，因此在时间步的特征中
#融合进位置编码的信息是合理的。具体的使用不同频率的sin和cos函数来进行位置编码
class PositionalEncoding(torch.nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        """位置编码类的初始化函数
        d_model:词嵌入维度
        dropout: dropout触发比率
        max_len: 每个句子的最大长度
        """
        super(PositionalEncoding,self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        #Compue the positional encoding
        #注意下面代码的计算方式与公式中给出的是不同的，但是是等价的
        #这样计算是为了避免中间的数值计算结果超过float的范围
        pe = torch.zeros(max_len,d_model)#max_len=5000表示每一个句子的最大长度,d_model=512
        position = torch.arange(0,max_len).unsqueeze(1)#torch.Size([5000, 1])
        #torch.arange(0,d_model,2),取[0,model]之间步长为2的数，可以任务取偶数
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000.0) / d_model))
        #取偶数的位置编码
        pe[:,0::2] = torch.sin(position * div_term)
        #取奇数的位置位置编码
        pe[:,1::2] = torch.cos(position * div_term)
        #print(pe.size()),torch.Size([5000, 512])
        pe = pe.unsqueeze(0)
        #print(pe.size()),torch.Size([1, 5000, 512])
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x+ Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

#-----------------编码器---------------------#
def clones(module,N):
    "produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# We employ a residual conenction around each of the two sub-layers,followed by layer normalization
class LayerNorm(torch.nn.Module):
    "Construct a layernorm module(see citation for details)."
    def __init__(self,feature_size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(feature_size))
        self.b_2 = torch.nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(torch.nn.Module):
    """
    Encoder
    The encoder is composed of a stack of N=6 identical layers.
    """
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)


#full Model
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    """
    构建模型
    params:
        src_vocab:
        tgt_vocab:
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size,默认512
        d_ff: FeedForward layer层中embedding的size,默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy#深拷贝，将某一个变量的值赋值给另一个变量（两个变量地址不一样），因此相互不干扰
    attn=MultiHeadedAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    #编码器

    #解码器

def train():
    device = "cpu"
    nrof_epochs = 40
    batch_size = 32
    V = 11 #词典的数量
    sequence_len = 15 #生成的序列数据的长度
    nrof_batch_train_epoch = 30 #训练时每个epoch多少个batch
    nrof_batch_valid_epoch = 10 #验证时每个epoch多少个batch
    criterion = LabelSmoothing(size=V,padding_idx=0,smoothing=0.0)
    model = make_model(V,V,N=2)


def main():
    use_gpu = torch.cuda.is_available()
    if use_gpu == True:
        print('\n============>GPU可用:')
    else:
        print('\n============>GPU不可用')
        train()


if __name__ == '__main__':
    main()

