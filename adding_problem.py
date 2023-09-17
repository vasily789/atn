import numpy as np
import torch
from torch import nn
from torch.nn import functional, init
import argparse

parser = argparse.ArgumentParser(description='ATN-LSTM Adding Problem')
parser.add_argument('--time_steps', type=int, default=10,
                    help='time steps to keep for ATN')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_steps', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=60)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_time', type=int, default=25)
parser.add_argument('--mode', choices=['lstm', 'atn', 'ln'], type=str, default = 'atn')
args  = parser.parse_args()

class ATN(nn.Module):
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
    self.input_memory = []
    self.time_step = 0

  def forward(self, new_input):
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
        Initialize parameters following the way proposed in the paper.
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
            time: The current timestep value, which is used to
                get appropriate running statistics.
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

# Generates Synthetic Data
def Generate_Data(size, length):
    
    # Random sequence of numbers
    x_random = np.random.uniform(0,1, size = [size, length])

    # Random sequence of zeros and ones
    x_placeholders = np.zeros((size, length))
    firsthalf = int(np.floor((length-1)/2.0))
    for i in range(0,size):
        x_placeholders[i, np.random.randint(0, firsthalf)] = 1
        x_placeholders[i, np.random.randint(firsthalf, length)] = 1

    # Create labels
    y_labels = torch.FloatTensor(np.reshape(np.sum(x_random*x_placeholders, axis=1), (size,1)))
    
    # Creating data with dimensions (batch size, n_steps, n_input)
    data = torch.FloatTensor(np.dstack((x_random, x_placeholders)))
    
    return data, y_labels

class Model(nn.Module):
    def __init__(self, n_classes, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        if args.mode == 'lstm':
            self.rnn = nn.LSTMCell(n_classes + 1, hidden_size)
        else:
            self.rnn = Normalized_LSTMCell(n_classes + 1, hidden_size)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.MSELoss()
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
        return self.loss_func(logits[:,-1,:], y)


    def accuracy(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=2), y).float().mean()

def main():
    epoch = 1
    n_input = 2             
    n_steps = args.n_steps          # Length of sequence
    n_hidden = args.hidden_size          # Hidden layer size
    n_classes = 1  
    training_epochs = args.epochs
    batch_size = args.batch_size
    training_size = 100000   # Training set size
    testing_size = 10000     # Testing set size
    display_step = 100
    train_loss = []
    valid_loss = []
    
    model = Model(n_classes, n_hidden).to('cuda')
    optim = torch.optim.RMSprop(params = model.parameters(), lr = args.lr)
    # Generating training and test data
    x_train, y_train = Generate_Data(training_size, n_steps)
    test_data, test_label = Generate_Data(testing_size, n_steps)
    
    device = torch.device('cuda')
    while epoch <= training_epochs:
        model.train()
    
        step = 1
        # Keep training until reach max iterations
        while step * batch_size <= training_size:
            optim.zero_grad()                      
    
            # Getting input data
            batch_x = x_train[(step-1)*batch_size:step*batch_size,:,:].to('cuda')
            batch_y = y_train[(step-1)*batch_size:step*batch_size].to('cuda')
    
            logits, rnn_out = model(batch_x)
            train_mse = model.loss(logits, batch_y)
            train_mse.backward()
            
            # Printing results
            print('\n')
            print("Epoch:", epoch)
            print("Percent complete:", step*batch_size/training_size) 
            print("Training Minibatch MSE:", train_mse.data)
            train_loss.append(train_mse.data)
    
            optim.step()         
            
            if args.mode == 'atn':
                model.rnn.norm_hh.reset_in_memory()          
                model.rnn.norm_ih.reset_in_memory()          
                model.rnn.norm_c.reset_in_memory()
            step += 1
            if step % 200 == 0:
              v_loss = 0
              test_step = 1
              model.eval()
              while test_step * batch_size <= testing_size:
                optim.zero_grad()
                test_x = test_data[(test_step-1)*batch_size:test_step*batch_size,:,:].cuda()
                test_y = test_label[(test_step-1)*batch_size:test_step*batch_size].cuda()
                test_logits, _ = model(test_x)
                test_mse = model.loss(test_logits, test_y)
                v_loss += test_mse.item()
                test_step += 1
                if args.mode == 'atn':
                    model.rnn.norm_hh.reset_in_memory()          
                    model.rnn.norm_ih.reset_in_memory()          
                    model.rnn.norm_c.reset_in_memory()
              valid_loss.append(v_loss/(testing_size/batch_size))
        epoch += 1

if __name__ == "__main__":
    main()