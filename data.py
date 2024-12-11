import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from env_killweb import Env_Kill_Web
from config import Config, load_pkl, pkl_parser



class Generator(Dataset):
	def __init__(self, cfg, env: Env_Kill_Web):
		self.blue_device = env.get_batch_blue_device(cfg.n_samples)
		self.red_device = env.get_batch_red_device(cfg.n_samples)

	def __getitem__(self, idx):
		return self.blue_device[idx], self.red_device[idx]

	def __len__(self):
		return self.blue_device.size(0)

if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	env = Env_Kill_Web(cfg)
	dataset = Generator(cfg, env)
	blue_device, red_device = next(iter(dataset))
	print(f"len(dataset): {len(dataset)}, batch size: {cfg.batch}")
	print(f"blue_device.size(): {blue_device.size()}, red_device.size(): {red_device.size()}")
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
	for i, (blue_device, red_device) in enumerate(dataloader):
		print(f"batch_blue_device.size(): {blue_device.size()}, batch_red_device.size(): {red_device.size()}")
		if i == 0:
			break
