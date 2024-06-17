import torch
from torch.autograd import Function
import sum_single
import sum_double

class SumSingle(Function):

    @staticmethod
    def forward(ctx, array):
      #pass

    @staticmethod
    def backward(ctx, g_out):
      #pass

class SumDouble(Function):

    @staticmethod
    def forward(ctx, array):
      #pass

    @staticmethod
    def backward(ctx, g_out):
      #pass

sum_single_op = SumSingle.apply
sum_double_op = SumDouble.apply
