import torch as t

from torchvision import datasets, transforms
import kornia.augmentation as K

import numpy as np

from tqdm import tqdm

from MobileNet_v4 import mobilenet_v4_custom, mobilenet_v4_nano, mobilenet_v4_small, mobilenet_v4_medium, mobilenet_v4_large, mobilenet_v4_hybrid_large

t.backends.cudnn.benchmark = True

TRAIN_EPOCHS=20
TEST_EPOCHS=3
BATCH_SIZE=64

# Аугментация
train_transform = t.nn.Sequential(
	K.RandomHorizontalFlip(),
	K.RandomCrop((32, 32), padding=4),
	K.ColorJitter(0.2, 0.2, 0.2, 0.1),
	K.RandomRotation(15),
	K.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	# K.Resize((84, 84))
)

test_transform = t.nn.Sequential(
	K.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	# K.Resize((84, 84))
)

train_dataset = t.utils.data.DataLoader(datasets.CIFAR10('F:/datasets', train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True)
test_dataset = t.utils.data.DataLoader(datasets.CIFAR10('F:/datasets', train=False, download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True)

#model = mobilenet_v4_large(in_channels=3, num_classes=100).cuda()

#from MobileNet_v4_adaptive import MobileNet_v4_adaptive
#model = MobileNet_v4_adaptive(in_channels=1, num_classes=10)

model = mobilenet_v4_nano(in_channels=3, num_classes=10).cuda()

optimizer = t.optim.Adam(params=model.parameters(), lr=0.001)
criterion = t.nn.CrossEntropyLoss()

model.train()
for epoch in (pbar := tqdm(range(TRAIN_EPOCHS))):
	for imgs, labels in train_dataset:
		imgs = train_transform(imgs.cuda())
		labels = labels.cuda()

		preds = model(imgs)
		loss = criterion(preds, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		#model.update(loss)

		pbar.set_description(f'Loss: {loss: .3f}')

accuracy = []

model.eval()
for epoch in tqdm(range(TEST_EPOCHS)):
	for imgs, labels in test_dataset:
		with t.no_grad():
			preds = t.argmax(model(test_transform(imgs.cuda())), axis=1).cpu().numpy()
			labels = labels.cpu().numpy()
			
		[accuracy.append(i) for i in np.array(preds == labels, dtype=np.int32)]

accuracy = np.array(accuracy)

print(f'Mean accuracy: {np.mean(accuracy)*100: .1f}%')