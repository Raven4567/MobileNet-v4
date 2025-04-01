import torch as t
from torch import nn, optim

from MobileNet_v4 import mobilenet_v4_custom, Fused_IB_block, Universval_IB_block, ExtraDW_block, SE_block, MobileMQA_block
from PPO import PPO

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class MobileNet_v4_adaptive(nn.Module):
	def __init__(
			self, in_channels: int=3, num_classes: int=1000
		):
		super().__init__()

		self.MobileNet = mobilenet_v4_custom(
			in_channels=in_channels, num_classes=num_classes,
			base_channels=16, expanded_channels=32, squeezed_channels=4,
			num_heads=2, downsample=True,
			hidden_dim_classifier=64, dropout=0.15
		)
		
		self.ppo = PPO(
			
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)

			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.normal_(m.bias, mean=0, std=0.01)

			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

			elif isinstance(m, nn.LayerNorm):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

		self.blocks = [
			lambda: nn.Sequential(Fused_IB_block(base_channels=16, expanded_channels=32)).to(device),
			lambda: nn.Sequential(Universval_IB_block(base_channels=16, expanded_channels=32), ExtraDW_block(base_channels=16, expanded_channels=32)).to(device),
			lambda: nn.Sequential(SE_block(base_channels=16, squeezed_channels=4), Universval_IB_block(base_channels=16, expanded_channels=32)).to(device),
			lambda: nn.Sequential(SE_block(base_channels=16, squeezed_channels=4), MobileMQA_block(base_channels=16, num_heads=4, downsample=False)).to(device)
		]

		self.optimizer = optim.Adam(self.parameters(), lr=0.001, amsgrad=True)

		self.to(device)

	def forward(self, x: t.Tensor):
		return self.MobileNet(x)

	def update(self, loss: t.Tensor):
		adaptive_loss = loss + 0.05 * sum([len(block) for block in self.MobileNet.features[1:5]])

		self.optimizer.zero_grad()
		adaptive_loss.backward()
		self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)

		probs = self.model(
			self.embedding(
				t.cat([
					adaptive_loss.unsqueeze(-1), 
					t.tensor([len(block) for block in self.MobileNet.features[1:]], dtype=t.float32, device=device)
				], dim=-1)	
			)
		)

		idx_probs, action_probs = self.index_chooser(probs), self.action_chooser(probs)

		idx, action = t.argmax(idx_probs), t.argmax(action_probs)
		
		entropy_loss = t.sum(probs * t.log(probs + 1e-8))  # Encourage randomness
		adaptive_loss += 0.02 * entropy_loss

		if action.item() == 0:
			if sum([len(block) for block in self.MobileNet.features[1:]]) < 20:
				self.MobileNet.features[idx+1].append(self.blocks[idx]())

				self.optimizer.param_groups = [{'params': self.parameters()}]

		elif action.item() == 1:
			if sum([len(block) for block in self.MobileNet.features[1:]]) > 4:
				del self.MobileNet.features[idx+1][-1]

				self.optimizer.param_groups = [{'params': self.parameters()}]

		elif action.item() == 2:
			pass