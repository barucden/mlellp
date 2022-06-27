import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv

from em import EMAlgorithm
from imagemodel import ImageModel

class Accuracy:

    def __init__(self, ts):
        self.ts = ts

    def evaluate(self, em: EMAlgorithm):
        preds = [p >= 0.5 for p in em.rhos]
        tp = sum(np.sum(t == p) for t, p in zip(self.ts, preds))
        l = sum(len(t) for t in self.ts)
        return tp / l

def random_bag():
    return np.random.rand(8, 3, 32, 32).astype('float32')

def random_labels():
    return np.random.rand(8) > 0.5

# Images
Xs = [random_bag(), random_bag(), random_bag()]
# Instance labels (used for computing accuracy, not used in training)
ts = [random_labels(), random_labels(), random_labels()]
# Number of instances per bag
ys = [np.sum(t) for t in ts]

network = tv.models.resnet18(num_classes=1, norm_layer=nn.Identity)
optimizer = t.optim.Adam(network.parameters(), lr=1e-4)
lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
augmentation = tv.transforms.Compose([
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.RandomHorizontalFlip()])

model = ImageModel(
        np.concatenate(Xs),
        network,
        optimizer,
        batch_size=16,
        lr_scheduler=lr_scheduler,
        augmentation=augmentation)
em = EMAlgorithm(Xs, ys, model, 'models')

history = em.fit(8, metrics=[Accuracy(ts)])
print(history)

trained_network = tv.models.resnet18(num_classes=1, norm_layer=nn.Identity)
trained_network.load_state_dict(t.load('models/model.out'))

