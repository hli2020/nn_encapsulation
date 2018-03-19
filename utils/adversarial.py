import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from foolbox.models import PyTorchModel, TensorFlowModel
import foolbox
import torchvision.models as models
from foolbox.criteria import TargetClassProbability, Misclassification

target_class = 22
criterion = TargetClassProbability(target_class, p=0.99)
# criterion = Misclassification()  # default value
# reference:
# https://foolbox.readthedocs.io/en/latest/user/examples.html


fmodel = models.__dict__['resnet101'](pretrained=True)
fmodel = fmodel.cuda()
fmodel.eval()
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = PyTorchModel(fmodel, num_classes=1000, bounds=(0, 255))
attack = foolbox.attacks.FGSM(fmodel, criterion)
# attack = foolbox.attacks.ContrastReductionAttack(fmodel, criterion)

# image, label = foolbox.utils.imagenet_example()
image = np.asarray(Image.open('example.jpg'), dtype=np.float32)
image = image[:, :, ::-1].transpose([2, 0, 1])
image = np.ascontiguousarray(image)
label = np.argmax(fmodel.predictions(image))

adversarial = attack(image, label)

plt.subplot(1, 3, 1)
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.imshow(adversarial)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - image)
