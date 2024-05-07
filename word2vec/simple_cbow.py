import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size:int, hidden_size:int) -> None:
        Win=np.random.randn(vocab_size, hidden_size).astype('f')*0.01
        Wout=np.random.randn(hidden_size,vocab_size).astype('f')*0.01

        self.in_layer0=MatMul(Win)
        self.in_layer1=MatMul(Win)
        self.out_layer=MatMul(Wout)
        self.loss_layer=SoftmaxWithLoss()

        self.params, self.grads=[], []

        for layer in [self.in_layer0, self.in_layer1, self.out_layer]:
            self.params+=layer.params
            self.grads+=layer.grads
        
        self.word_vecs=Win
    
    def forward(self,contexts:np.ndarray, targets:np.ndarray):
        h0=self.in_layer0.forward(contexts[:,0])
        h1=self.in_layer1.forward(contexts[:,1])
        h=(h0+h1)*0.5
        score=self.out_layer.forward(h)
        loss=self.loss_layer.forward(score, targets)
        return loss
    
    def backward(self,dout=1):
        ds = self.loss_layer.backward(dout)
        da=self.out_layer.backward(ds)
        da*=0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None