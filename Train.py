import argparse
from functools import partial
from itertools import chain
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.ops import MLP
from torchvision import transforms
from tqdm import tqdm
import wandb

from Optimizers import *
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_name(args):
    suffix = f"-{arg.suffix}" if not args.suffix is None else ""
    return f"{args.task}-arch{args.arch}-lr{args.lr}-rho{args.rho}-{args.uid}{suffix}"

class Linear(nn.Module):
    
    def __init__(self):
        super(Linear, self).__init__()
        self.model = nn.Linear(784, 10)
            
    def forward(self, x): return self.model(x.view(-1, 784))

class MnistMLP(nn.Module):

    def __init__(self, args):
        super(MnistMLP, self).__init__()
        self.model = MLP(in_channels=784, hidden_channels=[32, 32, 10])
    
    def forward(self, x): return self.model(x.view(-1, 784))
        

class X2(nn.Module):

    def __init__(self):
        super(X2, self).__init__()
        self.x = nn.Parameter(torch.tensor(5000, dtype=torch.float32))
    
    def forward(self): return self.x ** 2

def x2_task(args):
    """Returns a list where the ith element is the loss on the ith iteration."""
    model = X2().to(device)
    optimizer = wrap_optimizer(model, args)
    
    losses = []
    wandb_log_iter = max(1, args.iterations // 1500)
    for idx in tqdm(range(args.iterations)):

        def closure():
            model.zero_grad(set_to_none=True)
            loss = model()
            loss.backward()
            return loss

        loss = model()
        loss.backward()
        optimizer.step(closure=closure)
        optimizer.zero_grad()

        detached_loss = loss.detach()
        losses.append(detached_loss)
        if idx % wandb_log_iter == 0 or idx == args.iterations - 1:
            wandb.log({"loss/tr": detached_loss, "iteration": idx})
        
    return {"loss/tr": [l.item() for l in losses]}
    

def mnist_task(args):

    model = get_model(args).to(device)
    optimizer = wrap_optimizer(model, args)
    loss_fn = nn.CrossEntropyLoss()

    data_tr = MNIST("mnist_data",
        train=True,
        download=True,
        transform=transforms.ToTensor())
    data_te = MNIST("mnist_data",
        train=False,
        download=True,
        transform=transforms.ToTensor())
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True)
    loader_te = DataLoader(data_te,
        batch_size=args.bs * 2,
        pin_memory=True,
        num_workers=args.num_workers)

    if args.iterations % len(loader_tr) == 0:
        num_passes_over_loader = max(1, args.iterations // len(loader_tr))
    else:
        raise ValueError(f"Number of iterations {args.iterations} should be evenly divided by the DataLoader length {len(loader_tr)}")

    
    loader = chain(*[loader_tr] * num_passes_over_loader)

    losses_tr, losses_te, accs_te = [], [], []
    wandb_log_iter = max(1, args.iterations // 1500)
    for idx,(x,y) in tqdm(enumerate(loader),
        total=len(loader_tr) * num_passes_over_loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        def closure():
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            return loss

        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step(closure=closure)
        optimizer.zero_grad()

        detached_loss = loss.detach()
        losses_tr.append(detached_loss)
        if idx % wandb_log_iter == 0 or idx == args.iterations - 1:
            wandb.log({"loss/tr": detached_loss, "iteration": idx})

        if idx % args.eval_iter == 0 or idx == args.iterations - 1:
            acc, loss = Utils.eval_model(model, loader_te, loss_fn=loss_fn)
            accs_te.append(acc)
            losses_te.append(loss)
            wandb.log({"acc/te": acc,
                "loss/te": loss,
                "iteration": idx})

    return {"loss/tr": [l.item() for l in losses_tr],
        "loss/te": losses_te,
        "acc/te": accs_te}

def get_model(args):
    if args.arch == "x2":
        return X2()
    elif args.arch == "linear":
        return Linear()
    elif args.arch == "mlp":
        return MnistMLP(args)
    else:
        raise NotImplementedError()

def wrap_optimizer(model, args):
    """Returns an optimizer over the parameters of [model] given [args]."""

    # Function mapping from model parameters to SGD optimizer over them
    get_base_optimizer = partial(SGD,
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=args.mm)

    if args.opt == "sgd":
        return SGD(model.parameters(),
            lr=args.lr,
            nesterov=args.nesterov,
            momentum=args.mm)
    elif args.opt == "sam":
        return SAM(model.parameters(), get_base_optimizer,
            rho=args.rho,
            lr=args.lr,
            method=1)
    elif args.opt == "asam":
        return SAM(model.parameters(), get_base_optimizer,
            rho=args.rho,
            lr=args.lr,
            adaptive=True)
    elif args.opt == "sam_no_norm_division":
        return SAM(model.parameters(), get_base_optimizer,
            rho=args.rho,
            lr=args.lr,
            adaptive=True)
    else:
        raise NotImplementedError()

def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online"],
        help="WandB usage")
    P.add_argument("--task", choices=["mnist", "x2"], default="x2",
        help="Task to run")
    P.add_argument("--opt", choices=["sgd", "sam", "asam"], 
        help="Optimizer to use")
    P.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate")
    P.add_argument("--rho", type=float, default=.05,
        help="Rho in SAM and derivatives")
    P.add_argument("--iterations", type=float, default=1000,
        help="Number of gradient steps")
    P.add_argument("--arch", default="x2", choices=["x2", "linear", "mlp"],
        help="WandB usage")
    P.add_argument("--nesterov", dest="nesterov", action="store_true",
        help="Use Nesterov momenetum with inner optimizer")
    P.add_argument("--mm", type=float, default=0,
        help="Inner optimizer momentum")
    P.add_argument("--suffix", default=None, type=str,
        help="Optional suffix")
    P.add_argument("--bs", type=int, default=300,
        help="Batch size")
    P.add_argument("--num_workers", type=int, default=20,
        help="Number of workers")
    P.add_argument("--eval_iter", type=int, default=100,
        help="Number of gradient steps between validations")

    args = P.parse_args()
    args.uid = wandb.util.generate_id()
    args.arch = "x2" if args.task == "x2" else args.arch
    return args

if __name__ == "__main__":
    args = get_args()
    tqdm.write(str(args))

    run = wandb.init(project="ImprovedSAM",
        anonymous="allow",
        config=args,
        id=args.uid,
        name=get_name(args),
        mode=args.wandb)

    if args.task == "x2":
        result = x2_task(args)
    elif args.task == "mnist":
        result = mnist_task(args)


    
    