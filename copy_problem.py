import torch
from torch import nn
from torch.nn import functional, init
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='ATN Copy Problem')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=68)
parser.add_argument('--iterations', type=int, default=4000)
parser.add_argument('--sequence_length', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--max_time', type=int, default=45)
parser.add_argument('--mode', choices=['lstm', 'atn', 'ln'], type=str, default = 'atn')
args = parser.parse_args()
# Fix seed across experiments for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

batch_size  = args.batch_size
hidden_size = args.hidden_size
iterations  = args.iterations
L           = args.sequence_length
device      = torch.device('cuda')

class ATN(nn.Module):
    
  """
  A method for performing Assorted Time Normalization
  """
  def __init__(self, input_size, max_time=5, eps = 1e-5):
    super(ATN, self).__init__()
    self.time_step = 0
    self.eps = eps
    self.k = max_time
    self.input_memory = []
    self.weight = nn.Parameter(torch.FloatTensor(input_size))
    self.bias = nn.Parameter(torch.FloatTensor(input_size))
    self.weight.data = torch.ones_like(self.weight.data)
    self.bias.data = torch.zeros_like(self.bias.data)
    self.reset_in_memory()

  def reset_in_memory(self):
      
    """
    Resets the memory of the inputs after each minibatch
    """
      
    self.input_memory = []
    self.time_step = 0

  def forward(self, new_input): #new_input of the size [batch, input/hidden]
    """
    Args: 
        new_input: A (batch_size, input_size) tensor to be normalized
    Returns:
        A normalized (batch_size, input_size) tensor
    """
    if self.time_step >= self.k:
      self.input_memory.pop(0)
      self.input_memory.append(new_input)
    else:
      self.input_memory.append(new_input)
    mean = torch.mean(torch.stack(self.input_memory), dim=[0,2], keepdim = True)
    var = torch.var(torch.stack(self.input_memory), dim=[0,2], unbiased = False, keepdim = True)
    x_norm = (new_input - mean) * torch.pow(var + self.eps,-0.5)
    self.time_step += 1
    return  (x_norm*self.weight + self.bias).squeeze()
    
class Normalized_LSTMCell(nn.Module):

    """A Normalized-LSTM cell."""

    def __init__(self, input_size, hidden_size):

        super(Normalized_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias_h = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_x = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        if args.mode == 'atn':
            self.norm_hh = ATN(4 * hidden_size, args.max_time)
            self.norm_ih = ATN(4 * hidden_size, args.max_time)
            self.norm_c = ATN(hidden_size, args.max_time)
        else:
            self.norm_hh = nn.LayerNorm(4 * hidden_size)
            self.norm_ih = nn.LayerNorm(4 * hidden_size)
            self.norm_c = nn.LayerNorm(hidden_size)
            
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        with torch.no_grad():
          # The input-to-hidden weight matrix is initialized orthogonally.
          init.orthogonal_(self.weight_ih.data)
          # The hidden-to-hidden weight matrix is initialized as an identity
          # matrix.
          weight_hh_data = torch.eye(self.hidden_size)
          weight_hh_data = weight_hh_data.repeat(1, 4)
          self.weight_hh.set_(weight_hh_data)
          # The bias is initialized to zero vectors.
          init.constant_(self.bias_h.data, val=0)
          init.constant_(self.bias_x.data, val=0)


    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch_h = (self.bias_h.unsqueeze(0)
                      .expand(batch_size, *self.bias_h.size()))
        bias_batch_x = (self.bias_x.unsqueeze(0)
                      .expand(batch_size, *self.bias_x.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        normed_wh = self.norm_hh(wh+bias_batch_h)
        normed_wi = self.norm_ih(wi+bias_batch_x)
        f, i, o, g = torch.split(normed_wh + normed_wi,
                                 split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.norm_c(c_1))
        return h_1, c_1
 

def copying_data(L, K, batch_size):
    seq = np.random.randint(1, high=9, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, L))
    zeros2 = np.zeros((batch_size, K-1))
    zeros3 = np.zeros((batch_size, K+L))
    marker = 9 * np.ones((batch_size, 1))

    x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

    return x, y


class Model(nn.Module):
    def __init__(self, n_classes, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        if args.mode == 'lstm':
            self.rnn = nn.LSTMCell(n_classes + 1, hidden_size)
        else:
            self.rnn = Normalized_LSTMCell(n_classes + 1, hidden_size)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin.weight.data, nonlinearity="relu")
        nn.init.constant_(self.lin.bias.data, 0)

    def forward(self, inputs):
        state = (torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device),
                  torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device))
        outputs = []
        states = []
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input, state)
            state = (out_rnn, state)
            states.append(out_rnn)
            outputs.append(self.lin(out_rnn))
        return torch.stack(outputs, dim=1), torch.stack(states)

    def loss(self, logits, y):
        return nn.functional.cross_entropy(logits.view(-1, 9), y.view(-1))


    def accuracy(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=2), y).float().mean()

def onehot(out, input):
    out.zero_()

    in_unsq = torch.unsqueeze(input, 2)
    out.scatter_(2, in_unsq, 1)

def main():
    # --- Set data params ----------------
    n_classes = 9
    n_characters = n_classes + 1
    K = 10
    n_train = args.iterations + args.batch_size
    n_len = L + 2 * K
    valid_loss = []
    train_loss = []

    model = Model(n_classes, hidden_size).to(device)

    model.train()
    
    optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)

    x_onehot = torch.FloatTensor(batch_size, n_len, n_characters).to(device)

    for step in range(iterations):
        
        model.zero_grad()
        
        train_x, train_y = copying_data(L, K, batch_size)

        batch_x = train_x.to(device)
        onehot(x_onehot, batch_x)
        batch_y = train_y.to(device)
        logits, out_rnn = model(x_onehot)
        loss = model.loss(logits, batch_y)

        loss.backward(retain_graph=True)

        optim.step()

        with torch.no_grad():
            accuracy = model.accuracy(logits, batch_y)
            train_loss.append(loss.item())
        if args.mode == 'atn':
            model.rnn.norm_hh.reset_in_memory()          
            model.rnn.norm_ih.reset_in_memory()          
            model.rnn.norm_c.reset_in_memory()
        print("Iter {}: Loss= {:.6f}, Accuracy= {:.5f}".format(step, loss, accuracy))

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_x, test_y = copying_data(L, K, batch_size)
                test_x, test_y = test_x.to(device), test_y.to(device)
                onehot(x_onehot, test_x)
                logits, out_rnn = model(x_onehot)
                loss = model.loss(logits, test_y)
                if args.mode == 'atn':
                    model.rnn.norm_hh.reset_in_memory()          
                    model.rnn.norm_ih.reset_in_memory()          
                    model.rnn.norm_c.reset_in_memory()
                accuracy = model.accuracy(logits, test_y)
                print("Test result: Loss= {:.6f}, Accuracy= {:.5f}".format(loss, accuracy))
                valid_loss.append(loss.item())
    print("Optimization Finished!")
    return train_loss, valid_loss
    
if __name__ == "__main__":
    main()