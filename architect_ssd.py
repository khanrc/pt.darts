""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


class Architect_SSD():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.bad_len = 24 # hard coded, determined by bad_count. represents the
        # number of weights/biases in self.head

    def virtual_step(self, trn_X, trn_y, xi, w_optim, is_multi):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.partial_loss(trn_X, trn_y, is_multi) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, list(self.net.weights())[:-self.bad_len], allow_unused=True)

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            # assert len(list(self.net.weights())) == len(list(self.net.named_parameters())), (len(list(self.net.weights())), len(list(self.net.named_parameters())))
            bad_count = 0
            # for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
            #     m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
            #     vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
            q_len = len(list(self.net.weights())) - self.bad_len
            assert len(list(self.net.weights())) == len(list(self.v_net.weights()))
            assert len(list(self.net.weights()))-self.bad_len == len(gradients), f"{len(list(self.net.weights()))}, {len(gradients)}"
            for q, (w, vw, g, (name, _)) in enumerate(zip(self.net.weights(), self.v_net.weights(), gradients, list(self.net.named_parameters())[8:])):
                if q > q_len:
                    break
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                try:
                    vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
                except TypeError:
                    print(name, m , g)
                    bad_count += 1
                    if bad_count > 50:
                        exit()
            print(bad_count)
            # exit()

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, is_multi):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim, is_multi)

        # calc unrolled loss
        loss = self.v_net.partial_loss(val_X, val_y, is_multi) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())[:-self.bad_len]
        v_weights = tuple(self.v_net.weights())[:-self.bad_len]
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y, is_multi)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y, is_multi):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y, is_multi)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y, is_multi)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
