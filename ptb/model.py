import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import math

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


class LSTM(nn.Module):


    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, max_length=1, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout


        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              max_length = max_length,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = (torch.zeros((input_.size(1), self.hidden_size), device='cuda'),
                 torch.zeros((input_.size(1), self.hidden_size), device='cuda'))
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            
            hx_layer = (hx[0][:,:], hx[1][:,:])
            
            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)

class ATN_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, max_length=5, use_bias=True):

        super(ATN_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias_h = nn.Parameter(torch.FloatTensor(4 * hidden_size))
            self.bias_x = nn.Parameter(torch.FloatTensor(4 * hidden_size))
            
        self.norm_hh = ATN(4 * hidden_size, max_length)
        self.norm_ih = ATN(4 * hidden_size, max_length)
        self.norm_c = ATN(hidden_size, max_length)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
          # The input-to-hidden weight matrix is initialized orthogonally.
          init.orthogonal(self.weight_ih.data)
          # The hidden-to-hidden weight matrix is initialized as an identity
          # matrix.
          weight_hh_data = torch.eye(self.hidden_size)
          weight_hh_data = weight_hh_data.repeat(1, 4)
          self.weight_hh.set_(weight_hh_data)
          # The bias is just set to zero vectors.
          init.constant(self.bias_h.data, val=0)
          init.constant(self.bias_x.data, val=0)


    def forward(self, input_, hx):

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

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0, dropouth=0, dropouti=0, dropoute=0, wdrop=0, tie_weights=False, max_length = 1):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [LSTM(ATN_LSTMCell, ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0, max_length = max_length) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb

        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs, emb
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]