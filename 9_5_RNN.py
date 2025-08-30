import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from d2l import torch as d2l
config={
    'batch_size': 32,
    'num_steps': 35,
    'root':'./time_machine',
    'num_epochs':15,
    'lr':1,
    'nums_hidden':32
}

data = d2l.TimeMachine(batch_size=config['batch_size'], num_steps=config['num_steps'],root=config['root'])
train_dataset = TensorDataset(data.X[:data.num_train], data.Y[:data.num_train])
val_dataset = TensorDataset(data.X[data.num_train:data.num_train+data.num_val], 
                               data.Y[data.num_train:data.num_train+data.num_val])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
vocab = data.vocab
print(f"词汇表大小: {len(vocab)}")


class RNN(nn.Module):
    def __init__(self,num_inputs,num_hiddens,sigma=0.1):
        super().__init__()
        self.num_inputs=num_inputs
        self.num_hiddens=num_hiddens
        self.sigma=sigma
        self.Wxh=nn.Parameter(torch.randn(num_inputs,num_hiddens)*sigma)
        self.Whh=nn.Parameter(torch.randn(num_hiddens,num_hiddens)*sigma)
        self.bh=nn.Parameter(torch.zeros(num_hiddens))
    def forward(self,inputs,state=None):
        #inputs (num_steps,batch_size,num_inputs)
        #Wxh (num_inputs,h)
        #Whh (h,h)
        #state (batch_size,h)
        if state is None:
            state=torch.zeros((inputs.shape[1],self.num_hiddens),device=inputs.device)
        else:
            #state,=state
            pass
        rnn_outputs=[]
        for X in inputs:
            #print (X.shape)
            state=torch.tanh(torch.matmul(X,self.Wxh)+torch.matmul(state,self.Whh)+self.bh)
            rnn_outputs.append(state)
        return rnn_outputs,state #rnn_outputs是每个中间层的状态矩阵，state是最后一个的中间层状态矩阵(batch_size,h)
    
class RNNLMS(nn.Module):
    def __init__(self,rnn,num_outputs):
        super().__init__()
        self.rnn=rnn
        self.num_outputs=num_outputs
        self.Whq=nn.Parameter(torch.randn(self.rnn.num_hiddens,self.num_outputs)*self.rnn.sigma)
        self.bq=nn.Parameter(torch.zeros(self.num_outputs))
    def output_layers(self,rnn_outputs):
        outputs=[torch.matmul(H,self.Whq)+self.bq for H in rnn_outputs]
        return torch.stack(outputs,1)
        #插在1维度，返回的是(batch_size,num_steps,num_outputs)
    def one_hot(self,X):#这是对输入端的处理
        return F.one_hot(X.T,self.rnn.num_inputs).type(torch.float32)# one_hot函数，第二个参数为要编码的总类个数。在输入张量最后一个维度变为num_inputs（第二个参数），使得原来每个元素变为独热编码
        #返回的是 (num_steps,batch_size,num_inputs)
    def forward(self,X,state=None):
        #X (batch_size,num_steps)
        embs=self.one_hot(X)
        #embs (num_steps,batch_size,num_inputs)
        rnn_outputs,final_state=self.rnn(embs,state)
        #print(final_state.shape)
        return self.output_layers(rnn_outputs),final_state
    #返回(batch_size,num_steps,num_outputs)

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

@torch.no_grad()  
def evaluate_perplexity(model, data_iter, device):
    """
    在指定数据集上评估语言模型的困惑度。
    """
    model.eval() 
    metric=d2l.Accumulator(2)  # [总损失, 总词元数]
    state=None
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        if state is not None and hasattr(state, 'detach_'):
            state=state.detach()
        y_hat,state=model(X, state)
        l = nn.CrossEntropyLoss()(y_hat.reshape(-1, model.num_outputs), y.reshape(-1))
        metric.add(l * y.numel(), y.numel())
        
    # 返回最终的困惑度
    return math.exp(metric[0] / metric[1])

def train(model, data_iter,val_iter, lr, num_epochs, device):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    animator = d2l.StaticPlotter(xlabel='epoch', ylabel='perplexity',
                            legend=['train', 'validation'], xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)  # [总损失, 总词元数]
        timer = d2l.Timer()
        state = None # 初始化每个epoch开始时的隐藏状态
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            if state is not None and hasattr(state, 'detach_'):
                state = state.detach()
            y_hat, state = model(X, state)
            

            input_for_loss = y_hat.reshape(-1, model.num_outputs)
            target_for_loss = y.reshape(-1)
            l = loss_fn(input_for_loss, target_for_loss)
            
            optimizer.zero_grad()
            l.backward()
            
            grad_clipping(model, 1.0)
            optimizer.step()
            metric.add(l * target_for_loss.numel(), target_for_loss.numel())

        train_perplexity = math.exp(metric[0] / metric[1])

        val_perplexity = evaluate_perplexity(model, val_iter, device)

        animator.add(epoch + 1, [train_perplexity, val_perplexity])

    #训练结束
    final_perplexity = math.exp(metric[0] / metric[1])
    speed = metric[1] / timer.stop() # 每秒处理的词元数
    animator.show()
    print(f'Perplexity: {final_perplexity:.1f}, {speed:.1f} tokens/sec on {str(device)}')

my_rnn=RNN(num_inputs=len(vocab),num_hiddens=config['nums_hidden'])
my_model=RNNLMS(my_rnn,num_outputs=len(vocab))
my_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(my_model,train_loader,val_loader,lr=config['lr'],num_epochs=config['num_epochs'],device=my_device)