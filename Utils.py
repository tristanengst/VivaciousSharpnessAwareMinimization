import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_model(model, loader, loss_fn=None):
    """Returns the accuracy of [model] on data in [loader]."""
    losses, correct, total = [], 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            fx = model(x)
            preds = torch.argmax(fx, dim=1)
            correct += torch.sum(preds == y).item()
            total += len(preds)

            if not loss_fn is None:
                losses.append(loss_fn(fx, y).item())

    if loss_fn is None:
        return correct / total
    else:
        return correct / total, np.mean(losses)