
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import display

x_data = torch.randn(2, 100)
y_data = torch.mm(torch.Tensor([[0.100, 0.200]]), x_data) + 0.300

x_data = Variable(x_data, requires_grad = False)
y_data = Variable(y_data, requires_grad = False)

W = Variable(torch.zeros(1, 2), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

learning_rate = 0.01

state_win = 1
l_stats = []
w_stats = []
b_stats = []

for t in range(500):
    W.grad.data.zero_()
    b.grad.data.zero_()

    y_pred = torch.mm(W, x_data) 
    y_pred += b.unsqueeze(0).expand_as(y_pred)

    loss = ((y_pred - y_data)**2).mean()
    loss.backward()

    W.data -= learning_rate * W.grad.data
    b.data -= learning_rate * b.grad.data

    l_stats.append([t, loss.data[0]])
    w_stats.append([t, W.data[0][0], W.data[0][1]])
    b_stats.append([t, b.data[0]])

    if t % 20 == 0:
        #
        display.plot(l_stats, title="loss", win = state_win)
        display.plot(w_stats, title="weight", width=200, win = state_win + 1)
        display.plot(b_stats, title="bias", win = state_win + 2)

        print("it: #{} loss: {} W: [{}, {}], b: {}".format(
            t, loss.data[0], W.data[0][0], W.data[0][1], b.data[0]))