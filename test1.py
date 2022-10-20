import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from model1 import TransFuser
from data1 import CARLA_Data, CARLA_Data_Test

import matplotlib.pyplot as plt

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=bool, default=False, help='use the test dataset.')
parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=211, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=2, help='Validation frequency (epochs).')
parser.add_argument('--shuffle_every', type=int, default=6, help='Shuffle the dataset frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

writer = SummaryWriter(log_dir=args.logdir)
# class_weights=[174.109375, 3.28508255, 2.20391614, 2.48727679, 2.00125718, 0.90212111, 1.33930288, 1.29932369, 1.85222739, 0.52128555, 0.61960632, 1.27087135, 0.5440918,0.80981105, 1.48811432, 0.32912925, 0.28127524, 1.51399457, 0.61960632, 0.32850825, 0.5260102,0.20876424, 0.69366285, 3.28508255, 1.20909288, 0.58425965, 1.31901042, 0.76700165, 0.56713151, 1.40410786, 1.16072917, 1.19252997, 0.59626498, 0.57085041, 4.35273437, 0.84519114, 0.60037716, 0.59020127, 0.40117368, 1.52727522, 1.37093996, 0.47701199, 0.633125, 0.99491071, 1.64254127, 2.85425205, 2.80821573, 2.20391614, 3.05455044, 5.61643145, 1.65818452, 8.29092262, 4.97455357, 1.77662628, 2.26116071, 5.61643145, 5.27604167, 9.16365132, 5.27604167,10.88183594,12.43638393, 9.16365132,29.01822917, 29.01822917]
# class_weights=torch.tensor(class_weights,dtype=torch.float).cuda()
class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.DBA = []
		self.bestval = 0
		# self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
		self.criterion = torch.nn.CrossEntropyLoss()

	def validate(self):
		model.eval()
		running_acc = 0.0
		with torch.no_grad():
			num_batches = 0
			wp_epoch = 0.
			gt_beam_all = []
			pred_beam_all = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0):
				# create batch and move to GPU
				fronts = []
				lidars = []
				for i in range(config.seq_len):
					fronts.append(data['fronts'][i].to(args.device, dtype=torch.float32))
					lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))
				velocity = torch.zeros((data['fronts'][0].shape[0])).to(args.device, dtype=torch.float32)
				pred_beams = model(fronts, lidars, velocity)
				gt_beam_all.append(data['beam'][0])
				gt_beamidx = data['beam'][0].to(args.device, dtype=torch.long)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				running_acc += (torch.argmax(pred_beams, dim=1) == gt_beamidx).sum().item()
				wp_epoch += float(self.criterion(pred_beams, gt_beamidx))
				num_batches += 1
			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))

			gt_beam_all = np.squeeze(np.concatenate(gt_beam_all, 0))

			curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1, 2, 3])
			DBA_score_val = compute_DBA_score(pred_beam_all, gt_beam_all, max_k=3, delta=5)
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')
			print('Val', curr_acc, running_acc * 100 / test_size, DBA_score_val)

			writer.add_scalar('val_loss', wp_loss, self.cur_epoch)

			self.val_loss.append(wp_loss)
			self.DBA.append(DBA_score_val)
	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0):
				# create batch and move to GPU
				fronts = []
				lidars = []

				for i in range(config.seq_len):
					fronts.append(data['fronts'][i].to(args.device, dtype=torch.float32))
					lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))


				velocity = torch.zeros((data['fronts'][0].shape[0])).to(args.device, dtype=torch.float32)
				pred_beams = model(fronts, lidars, velocity)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			save_pred_to_csv(pred_beam_all, top_k=[1, 2, 3], target_csv='beam_pred.csv')

def save_pred_to_csv(y_pred, top_k=[1, 2, 3], target_csv='beam_pred.csv'):
	"""
    Saves the predicted beam results to a csv file.
    Expects y_pred: n_samples x N_BEAMS, and saves the top_k columns only.
    """
	cols = [f'top-{i} beam' for i in top_k]
	df = pd.DataFrame(data=y_pred[:, np.array(top_k) - 1]+1, columns=cols)
	df.index.name = 'index'
	df.to_csv(target_csv)
def compute_acc(y_pred, y_true, top_k=[1,2,3]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx])
            total_hits[k_idx] += 1 if hit else 0
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true)*100, 4)


def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
	"""
    The top-k MBD (Minimum Beam Distance) as the minimum distance
    of any beam in the top-k set of predicted beams to the ground truth beam.
    Then we take the average across all samples.
    Then we average that number over all the considered Ks.
    """
	n_samples = y_pred.shape[0]
	yk = np.zeros(max_k)
	for k in range(max_k):
		acc_avg_min_beam_dist = 0
		idxs_up_to_k = np.arange(k + 1)
		for i in range(n_samples):
			aux1 = np.abs(y_pred[i, idxs_up_to_k] - y_true[i]) / delta
			# Compute min between beam diff and 1
			aux2 = np.min(np.stack((aux1, np.zeros_like(aux1) + 1), axis=0), axis=0)
			acc_avg_min_beam_dist += np.min(aux2)

		yk[k] = 1 - acc_avg_min_beam_dist / n_samples

	return np.mean(yk)


# Config
config = GlobalConfig()
# Model
model = TransFuser(config, args.device)
model = torch.nn.DataParallel(model, device_ids = [0,1])
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# print([p.requires_grad for p in model_parameters])
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)
# load the pretrained model
model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if not args.test:
	# Load checkpoint
	# optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))
	test_root='/home/tiany0c/Downloads/MultiModeBeamforming/Adaptation_dataset_multi_modal/'
	test_root_csv='ml_challenge_data_adaptation_multi_modal.csv'
	# Data
	test_set = CARLA_Data(root=test_root, root_csv=test_root_csv, config=config)
	test_size=len(test_set)
	dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
	# Log args
	with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)
		trainer.validate()
if args.test:
	test_root = '/home/tiany0c/Downloads/MultiModeBeamforming/Multi_Modal_Test/'
	test_root_csv = 'ml_challenge_test_multi_modal.csv'
	# Data
	test_set = CARLA_Data(root=test_root, root_csv=test_root_csv, config=config)
	dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
	trainer.test()


