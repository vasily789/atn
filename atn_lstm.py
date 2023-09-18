import torch
from torch import nn
from torch.nn import init

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

    def forward(self, new_input): #new_input of the size [batch, input/hidden]
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

    def __init__(self, input_size, hidden_size, use_bias=True):

        super(Normalized_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # Initializing Weights and Biases
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias_h = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_x = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        # Initialization of normalization method within LSTM cell
        if args.mode == 'atn':
            self.norm_hh = ATN(4 * hidden_size, args.max_time, use_bias = use_bias)
            self.norm_ih = ATN(4 * hidden_size, args.max_time, use_bias = use_bias)
            self.norm_c = ATN(hidden_size, args.max_time, use_bias = use_bias)
        else:
            self.norm_hh = nn.LayerNorm(4 * hidden_size)
            self.norm_ih = nn.LayerNorm(4 * hidden_size)
            self.norm_c = nn.LayerNorm(hidden_size)
            
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
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
