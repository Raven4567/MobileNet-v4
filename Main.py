import torch as t
from torch import nn, optim

from torchvision import datasets, transforms
import kornia.augmentation as K

import numpy as np

from tqdm import tqdm

# from models import mobilenet_v4_custom, mobilenet_v4_nano, mobilenet_v4_small, \
#                    mobilenet_v4_medium, mobilenet_v4_large, mobilenet_v4_hybrid_large

t.backends.cudnn.benchmark = True

TRAIN_EPOCHS=20
TEST_EPOCHS=3
BATCH_SIZE=4

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Аугментация
train_transform = t.nn.Sequential(
	K.RandomHorizontalFlip(),
	K.ColorJitter(0.2, 0.2, 0.2, 0.1),
	K.RandomRotation(15),
	K.RandomCrop((32, 32), padding=4),
	K.Normalize(0.5, 0.5),
	K.Resize((224, 224))
)

test_transform = t.nn.Sequential(
	K.Normalize(0.5, 0.5),
	K.Resize((224, 224))
)

train_dataset = t.utils.data.DataLoader(datasets.CIFAR100('F:/datasets', train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True)
test_dataset = t.utils.data.DataLoader(datasets.CIFAR100('F:/datasets', train=False, download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True)

from modules import Stem, FusedIB, ExtraDW, IB, ConvNext  # noqa: E402
model = nn.Sequential(
	nn.Sequential(
		Stem(3, 32, kernel_size=(3, 3), stride=2, padding=1),
	),
	nn.Sequential(
		FusedIB(32, 32, 32, kernel_size=(3, 3), stride=2, padding=1),
	),
	nn.Sequential(
		FusedIB(32, 64, 96, kernel_size=(3, 3), stride=2, padding=1),
	),
	nn.Sequential(
		ExtraDW(64, 96, 192, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=2, padding_1st=2, padding_2nd=2),
		IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
		IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
		IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
		IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
		ConvNext(96, 96, 384, kernel_size=(3, 3), stride=1, padding=1),
	),
	nn.Sequential(
		ExtraDW(96, 128, 576, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=2, padding_1st=1, padding_2nd=1),
		ExtraDW(128, 128, 512, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
		IB(128, 128, 512, kernel_size=(5, 5), stride=1, padding=2),
		IB(128, 128, 384, kernel_size=(5, 5), stride=1, padding=2),
		IB(128, 128, 384, kernel_size=(3, 3), stride=1, padding=1),
		IB(128, 128, 384, kernel_size=(3, 3), stride=1, padding=1),
	),
	nn.Sequential(
		nn.Conv2d(128, 960, kernel_size=(1, 1), bias=False),
		nn.GroupNorm(960 // 8, 960),
		nn.SiLU(inplace=True),

		nn.AvgPool2d((7, 7)),

		nn.Conv2d(960, 1280, kernel_size=(1, 1), bias=False),
		nn.GroupNorm(1280 // 8, 1280),
		nn.SiLU(inplace=True),

		nn.Dropout(0.3),

		nn.Conv2d(1280, 100, kernel_size=(1, 1)),
		nn.Flatten()
	)
).to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
with tqdm(total=len(train_dataset.dataset) * TRAIN_EPOCHS, unit='img') as pbar:
	for epoch in range(TRAIN_EPOCHS):
		for imgs, labels in train_dataset:
			# imgs = train_transform(imgs.cuda())
			imgs = t.nn.functional.interpolate(
				imgs.cuda().mul_(2.0).sub_(1.0),
				size=(224, 224),
				mode='bilinear',
				align_corners=False
			)
			labels = labels.cuda()

			preds = model(imgs)
			loss = criterion(preds, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			pbar.update(imgs.shape[0])
			pbar.set_description(f'Loss: {loss: .3f}')

accuracy = []

model.eval()
with tqdm(total=len(test_dataset.dataset) * TEST_EPOCHS, unit='img') as pbar:
	for epoch in range(TEST_EPOCHS):
		for imgs, labels in test_dataset:
			with t.no_grad():
				imgs = imgs = t.nn.functional.interpolate(
					imgs.cuda().mul_(2.0).sub_(1.0),
					size=(224, 224),
					mode='bilinear',
					align_corners=False
				)

				preds = t.argmax(model(imgs), axis=1).cpu().numpy()
				labels = labels.cpu().numpy()
				
			[accuracy.append(i) for i in np.array(preds == labels, dtype=np.int32)]

			pbar.update(imgs.shape[0])

accuracy = np.array(accuracy)

print(f'Mean accuracy: {np.mean(accuracy)*100: .1f}%')