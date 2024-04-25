import numpy as np
import time
import torch
import torch.nn.functional as F
import cvxpy as cp
import cvxopt

from copy import deepcopy
from scipy.optimize import minimize, Bounds, minimize_scalar

from utils.min_norm_solver import MinNormSolver


def grad2vec(m, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        for mm in m.shared_modules():
            for p in mm.parameters():
                grad = p.grad
                if grad is not None:
                    grad_cur = grad.data.detach().clone()
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    grads[beg:en, task].copy_(grad_cur.data.view(-1))
                cnt += 1

def overwrite_grad(m, newgrad, grad_dims, num_class):

    newgrad = newgrad * num_class # to match the sum loss
    cnt = 0
    for param in m.parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone()
        cnt += 1

def mgd(grads):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([
            grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g


def pcgrad(args, grads, rng = np.random.default_rng()):
    grad_vec = grads.t()
    num_tasks = args['setting']['num_class']

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (
        grad_vec.norm(dim=1, keepdim=True) + 1e-8
    )  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[
            task_indices
        ]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(
            dim=1, keepdim=True
        )  # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g

def cagrad(epoch, args, grads, per_weight, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt()  # norm of the average gradient
    _, num_class = grads.shape
    if args['train']['init_preference'] == 'ave':
        x_start = np.ones(num_class) / num_class
    else:                                 # needed modify here
        x_start = per_weight
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_class).dot(A).dot(b.reshape(num_class, 1)) + c * np.sqrt(x.reshape(1, num_class).dot(A).dot(x.reshape(num_class, 1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)

    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2), num_class
    else:
        return g / (1 + alpha), num_class


def pareto_search(epoch, args, pareto_module, grads, grad_dims, per_weight, y):

    per_weight = per_weight[y] / np.sum(per_weight[y])

    if args['train']['pareto_solver'] == 'cagrad':
        g_w, num_class = cagrad(epoch, args, grads, per_weight)
    elif args['train']['pareto_solver'] == 'mgd':
        g_w = mgd(grads)
    elif args['train']['pareto_solver'] == 'pcgrad':
        g_w = pcgrad(args, grads)

    overwrite_grad(pareto_module, g_w, grad_dims, num_class)

    return


########################
#EPO Search
########################
class EPO_LP(object):

    def __init__(self, m, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a

def epo_search(args, pareto_module, grads, grad_dims, per_weight, y, losses):
    
    num_class = len(y)
    per_weight = per_weight[y] / np.sum(per_weight[y])
    GG = grads.t().mm(grads).cpu()
    epo_lp = EPO_LP(m=num_class, r=per_weight)
    alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
    alpha = torch.Tensor(alpha).to(grads.device)
    g_w = (grads * alpha.view(1, -1).cuda()).sum(1)

    overwrite_grad(pareto_module, g_w, grad_dims, num_class)

    return