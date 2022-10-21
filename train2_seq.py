import argparse
import json
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config_seq import GlobalConfig
from model2_seq import TransFuser
from data2_seq import CARLA_Data,CARLA_Data_Test

import matplotlib.pyplot as plt
import torchvision

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='transfuser_seq', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=70, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
parser.add_argument('--shuffle_every', type=int, default=6, help='Shuffle the dataset frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')	# default=24
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
		self.criterion = torch.nn.CrossEntropyLoss( reduction='mean')
		# self.criterion = torchvision.ops.sigmoid_focal_loss(reduction='mean')

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		running_acc = 0.0
		gt_beam_all = []
		pred_beam_all = []

		# Train loop
		pbar=tqdm(dataloader_train, desc='description')
		for data in pbar:
			
			# efficiently zero gradients
			# for p in model.parameters():
			# 	p.grad = None
			optimizer.zero_grad(set_to_none=True)
			
			# create batch and move to GPU
			fronts = []
			lidars = []
			radars = []
			gps = data['gps'].to(args.device, dtype=torch.float32)

			for i in range(config.seq_len):
				fronts.append(data['fronts'][i].to(args.device, dtype=torch.float32))
				lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))
				radars.append(data['radars'][i].to(args.device, dtype=torch.float32))

			# velocity=torch.zeros((data['fronts'][0].shape[0])).to(args.device, dtype=torch.float32)
			pred_beams = model(fronts, lidars, radars, gps)


			gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
			gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)

			running_acc += (torch.argmax(pred_beams, dim=1) == gt_beamidx).sum().item()
			# print('Pre',torch.argmax(pred_beams, dim=1))
			# print(pred_beams[0,:])
			# print(torch.argmax(pred_beams, dim=1) == gt_beamidx)

			loss = self.criterion(pred_beams, gt_beams)
			gt_beam_all.append(data['beamidx'][0])

			pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())

			loss.backward()
			loss_epoch += float(loss.item())
			pbar.set_description(str(loss.item()))
			num_batches += 1
			optimizer.step()

			writer.add_scalar('train_loss', loss.item(), self.cur_iter)
			self.cur_iter += 1
		pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))

		gt_beam_all = np.squeeze(np.concatenate(gt_beam_all, 0))

		curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1, 2, 3])
		DBA = compute_DBA_score(pred_beam_all, gt_beam_all, max_k=3, delta=5)
		print('Train',curr_acc, running_acc*100/train_size,DBA)
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self):
		model.eval()
		running_acc = 0.0

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			gt_beam_all=[]
			pred_beam_all=[]

			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				# create batch and move to GPU
				fronts = []
				lidars = []
				radars = []
				gps = data['gps'].to(args.device, dtype=torch.float32)

				for i in range(config.seq_len):
					fronts.append(data['fronts'][i].to(args.device, dtype=torch.float32))
					lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))
					radars.append(data['radars'][i].to(args.device, dtype=torch.float32))
				velocity=torch.zeros((data['fronts'][0].shape[0])).to(args.device, dtype=torch.float32)
				pred_beams = model(fronts, lidars, radars, gps)
				gt_beam_all.append(data['beamidx'][0])
				gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
				gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				running_acc += (torch.argmax(pred_beams, dim=1) == gt_beamidx).sum().item()
				wp_epoch += float(self.criterion(pred_beams, gt_beams))
				num_batches += 1

			pred_beam_all=np.squeeze(np.concatenate(pred_beam_all,0))
			gt_beam_all=np.squeeze(np.concatenate(gt_beam_all,0))
			curr_acc31=compute_acc(pred_beam_all[:50,:], gt_beam_all[:50], top_k=[1,2,3])
			DBA_score31=compute_DBA_score(pred_beam_all[:50,:], gt_beam_all[:50], max_k=3, delta=5)
			curr_acc32 = compute_acc(pred_beam_all[50:75, :], gt_beam_all[50:75], top_k=[1, 2, 3])
			DBA_score32 = compute_DBA_score(pred_beam_all[50:75, :], gt_beam_all[50:75], max_k=3, delta=5)
			curr_acc33 = compute_acc(pred_beam_all[75:, :], gt_beam_all[75:], top_k=[1, 2, 3])
			DBA_score33 = compute_DBA_score(pred_beam_all[75:, :], gt_beam_all[75:], max_k=3, delta=5)
			print('31:', curr_acc31, DBA_score31)
			print('32:', curr_acc32, DBA_score32)
			print('33:', curr_acc33, DBA_score33)

			curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1,2,3])
			DBA_score_val = compute_DBA_score(pred_beam_all, gt_beam_all, max_k=3, delta=5)
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')
			print('Val',curr_acc,running_acc*100/val_size,DBA_score_val)




			writer.add_scalar('val_loss', wp_loss, self.cur_epoch)
			
			self.val_loss.append(wp_loss)
			self.DBA.append(DBA_score_val)

	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			pred_beam_confidence = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0):
				# create batch and move to GPU
				fronts = []
				lidars = []
				radars = []
				gps = data['gps'].to(args.device, dtype=torch.float32)

				for i in range(config.seq_len):
					fronts.append(data['fronts'][i].to(args.device, dtype=torch.float32))
					lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))
					radars.append(data['radars'][i].to(args.device, dtype=torch.float32))

				velocity = torch.zeros((data['fronts'][0].shape[0])).to(args.device, dtype=torch.float32)
				pred_beams = model(fronts, lidars, radars, gps)
				# pred_beams = model(fronts, lidars, radars, velocity)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				sm=torch.nn.Softmax(dim=1)
				beam_confidence=torch.max(sm(pred_beams), dim=1)
				pred_beam_confidence.append(beam_confidence[0].cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			pred_beam_confidence = np.squeeze(np.concatenate(pred_beam_confidence, 0))
			save_pred_to_csv(pred_beam_all, top_k=[1, 2, 3], target_csv='beam_pred.csv')
			df = pd.DataFrame(data=pred_beam_confidence)
			df.to_csv('beam_pred_confidence_seq.csv')

	def save(self):
		save_best = False
		# if self.val_loss[-1] <= self.bestval:
		# 	self.bestval = self.val_loss[-1]
		# 	self.bestval_epoch = self.cur_epoch
		# 	save_best = True
		print('best', self.bestval, self.bestval_epoch)

		if self.DBA[-1] >= self.bestval:
			self.bestval = self.DBA[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True

		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'DBA': self.DBA,
		}

		# Save ckpt for every epoch
		# torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))
		#
		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
		#
		# # Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))
		#
		# tqdm.write('====== Saved recent model ======>')

		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')
		# else:
		# 	model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
		# 	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))
		# 	tqdm.write('====== Load the previous best model ======>')



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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# trainval_root='/home/tiany0c/Downloads/MultiModeBeamforming/0Multi_Modal/'
# train_root_csv='ml_challenge_dev_multi_modal1.csv'
# trainval_root= '/home/tiany0c/Downloads/MultiModeBeamforming/Adaptation_dataset_multi_modal/'

# trainval_root= '/efs/data/Multi_Modal/'
# train_root_csv='ml_challenge_dev_multi_modal.csv'

trainval_root='/efs/data/Adaptation_dataset_multi_modal/'
train_root_csv='ml_challenge_data_adaptation_multi_modal.csv'

# val_root='/home/tiany0c/Downloads/MultiModeBeamforming/Adaptation_dataset_multi_modal/'
val_root='/efs/data/Adaptation_dataset_multi_modal/'
val_root_csv='ml_challenge_data_adaptation_multi_modal.csv'

# test_root='/home/tiany0c/Downloads/MultiModeBeamforming/Multi_Modal_Test/'
test_root='/efs/data/Multi_Modal_Test/'
test_root_csv='ml_challenge_test_multi_modal.csv'

# Data
train_set = CARLA_Data(root=trainval_root, root_csv=train_root_csv, config=config)
# train_set = CARLA_Data(root=test_root, root_csv=test_root_csv, config=config)
val_set = CARLA_Data(root=val_root, root_csv=val_root_csv, config=config)
test_set = CARLA_Data_Test(root=test_root, root_csv=test_root_csv, config=config)
train_size, val_size= len(train_set), len(val_set)
# train_size = int(0.01 * len(train_set))
# train_set, _= torch.utils.data.random_split(train_set, [train_size, len(train_set) - train_size])
print(len(train_set),len(val_set) )
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)



# Model

model = TransFuser(config, args.device)
# model = torch.nn.DataParallel(model, device_ids = [2, 3])
model = torch.nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
trainer = Engine()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# print([p.requires_grad for p in model_parameters])
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']
	trainer.DBA = log_table['DBA']

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
	# optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))

	# trainer.validate()
	trainer.test()	# test



# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

for epoch in range(trainer.cur_epoch, args.epochs): 
	
	print('epoch:',epoch)
	# print('lr', scheduler.get_lr())

	# trainer.test()	# test

	# trainer.train()
	# trainer.validate()
	# trainer.save()
	# scheduler.step()

	# torch.save(model.state_dict(), os.path.join(args.logdir, 'current_model.pth'))
	# torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'current_optim.pth'))


	# if epoch % args.val_every == 0:
	# 	trainer.validate()
	# 	trainer.save()


