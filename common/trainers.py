import time
import numpy as np
from typing import List
from common.utils import clip_grads
import matplotlib.pyplot as plt

def remove_duplicates(params:List[np.ndarray], grads:List[np.ndarray]):
    """
    Remove duplicate parameters and add up grads corresponding to duplicate parameters
    """
    params, grads = params[:], grads[:]

    while True:
        # becomes True when we encounter duplicate parameters
        find_flag = False
        L = len(params)
        # iterate over individual parameters excluding the last one
        for i in range(0,L-1):
            # iterate over all parameters following the current one
            for j in range(i+1, L):
                # if current parameter and any of parameters following it are identical
                if params[i] is params[j]:
                    # add up gradients corresponding to duplicate parameters
                    grads[i] += grads[j]
                    # set find flag to True
                    find_flag = True
                    # remove jth param and grad from the list of params and grads
                    params.pop(j)
                    grads.pop(j)
                # if current parameter and any of parameters following it are identical after transposing the current parameter
                elif params[i].ndim==2 and params[j].ndim==2 and params[i].T.shape == params[j].shape and np.all(params[i].T==params[j]):
                    # add up grads after transposing grads following the current grads
                    grads[i] += grads[j].T
                    # set find flag to True
                    find_flag=True
                    # remove jth param and grad from the params and grads lists
                    params.pop(j)
                    grads.pop(j)
                # break out first nested loop if found a duplicate
                if find_flag: break
            # break out of first for loop if found a duplicate
            if find_flag: break
        # continue until no duplicates are left
        if not find_flag : break
    return params, grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
    
    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        start_time = time.time()

        for epoch in range(max_epoch):
            #shuffle data
            idx = np.random.permutation(np.arange(data_size))

            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # compute gradients and update parameters
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicates(model.params, model.grads)

                if max_grad is not None:
                    clip_grads(grads, max_grad)
                
                optimizer.update(params, grads)

                total_loss += loss
                loss_count +=1

                if eval_interval is not None and iters%eval_interval==0:
                    avg_loss = total_loss//loss_count
                    elapsed_time = time.time() - start_time
                    print(f"| epoch {epoch+1} | iter {iters} | time {elapsed_time}s | loss {avg_loss:.2f}")
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1
    
    def plot(self,ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f"iterations (x{self.eval_interval})")
        plt.ylabel('loss')
        plt.show()