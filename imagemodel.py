from math import ceil

import numpy as np
import torch
import torch.nn as nn

eps = 1e-6

def default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class ImageModel:

    def __init__(self, Xs: np.ndarray,
                 network: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int = 10,
                 lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None,
                 augmentation=None,
                 normalization=None,
                 device: torch.device = default_device()):
        """
        Initialize the image model.

        Parameters
        ----------
        Xs : numpy tensor
            a tensor of input images
        network : torch.nn.Module
            a PyTorch module representing the network to be used
        optimizer : torch.optim.Optimizer
            an optimizer that will be used for learning
        batch_size: int
            size of a batch during training
        lr_scheduler=None: torch.optim.lr_scheduler._LRScheduler
            a scheduler that will be used to adjust LR during learning
        augmentation
            a transform that will be used during learning
        normalization
            a transform that will be used for normalization
        device : torch.device
            device used for learning
        """
        # Convert the numpy arrays to torch tensors
        self.Xs = Xs
        self.network = network
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.augmentation = augmentation
        self.normalization = normalization
        self.device = device

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.network.to(self.device)
        self.bce_loss.to(self.device)

        # By default, switch the network to evaluation mode
        self.network.eval()

        self.optimizer_state = optimizer.state_dict()
        self.lr_state = lr_scheduler.state_dict()

        print("Using {}".format(device.type.upper()))

    def compute_prob(self, X: np.ndarray):
        """
        Computes probability for each image X_i in bag X.
        The bag is a tensor of shape (N, C, H, W).
        """
        # assert len(X.shape) == 4
        # assert X.shape[1] == 1 or X.shape[1] == 3
        # assert X.dtype == np.float32

        X = torch.from_numpy(X).to(self.device)

        # We do not need to compute gradient
        # This reduces resource consumption
        with torch.no_grad():
            if self.normalization:
                X = self.normalization(X)

            y = torch.sigmoid(self.network(X))
            y = y.clamp(eps, 1 - eps)
            return y.cpu().numpy().reshape(len(y))

    def fit(self, Phis: np.ndarray, n_epochs=np.inf):
        assert self.Xs.shape[0] == Phis.shape[0]

        # Switch the network to training mode (it affects, e.g., dropouts)
        # We will switch it back at the end of the update
        self.network.train()

        epoch, loss = 0, np.inf
        while epoch < n_epochs:
            loss = self._epoch(Phis)
            epoch += 1

            if self.lr_scheduler:
                self.lr_scheduler.step(loss)

            if epoch % 10 == 0:
                print("│ ~ {} epochs - loss {:.2E}".format(epoch, loss))

        print("│ {} epochs - loss {:.4E}".format(epoch, loss))

        # Switch the network back to eval mode
        self.network.eval()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def _epoch(self, Phis: np.ndarray):
        m = len(self.Xs)

        loss = 0
        # Process all mini-batches & collect the losses
        for X, Phi in self._each_batch(Phis):
            loss += self._batch_update(X, Phi).cpu().detach().numpy()

        return loss / m

    def _batch_update(self, X: torch.Tensor, t: torch.Tensor):
        # Zero the optimizer gradient
        self.optimizer.zero_grad()

        # Do the forward pass
        pred = self.network.forward(X).reshape(len(t))
        # Compute the loss
        l = self.bce_loss(pred, t)
        # Do the backward pass
        l.backward()
        # Perform one optimizer step
        self.optimizer.step()

        return l

    def _each_batch(self, Phis):
        m = self.Xs.shape[0]

        for j in range(ceil(m / self.batch_size)):
            X = self.Xs[j * self.batch_size:(j + 1) * self.batch_size]
            X = torch.from_numpy(X).to(self.device)

            Phi = Phis[j * self.batch_size:(j + 1) * self.batch_size]
            Phi = torch.from_numpy(Phi).to(self.device)

            with torch.no_grad():
                if self.augmentation:
                    X = self.augmentation(X)
                if self.normalization:
                    X = self.normalization(X)

            yield X, Phi

    def __call__(self, X):
        return self.compute_prob(X)

