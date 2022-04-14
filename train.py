import torch
from torch import autograd
from torch import optim
import json
import numpy as np
import pprint
import pickle
import datetime
from collections import OrderedDict
from model import *
from params import BATCH_SIZE, BETA1, BETA2, EPOCHS_PER_SAMPLE, LATENT_DIM, LEARNING_RATE, LMBDA, MODEL_SIZE, NGPUS, OUTPUT_DIR, SAMPLE_SIZE, TEST_RATIO, VALID_RATIO 
from utils import *
from logger import *

cuda = True if torch.cuda.is_available() else False

# =============Logger===============
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

LOGGER.info('Initialized logger.')
init_console_logger(LOGGER)

# ========Hyperparameters===========
args = parse_arguments()
epochs = args['num_epochs']
load_model = args['load_model']
batch_size = BATCH_SIZE
latent_dim = LATENT_DIM
ngpus = NGPUS
model_size = MODEL_SIZE
epochs_per_sample = EPOCHS_PER_SAMPLE
lmbda = LMBDA

audio_dir = AUDIO_DIR
output_dir = OUTPUT_DIR

# =============Network===============
netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)
netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)

# =============Load Model============
if load_model: 
    LOGGER.info("Loading models...")
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")
    netD = torch.load(netD_path)
    netG = torch.load(netG_path) 

if cuda:
    netG = netG.cuda()
    netD = netD.cuda()

# "Two time-scale update rule"(TTUR) to update netD 4x faster than netG.
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

# Sample noise used for generated output.
sample_noise = torch.randn(SAMPLE_SIZE, latent_dim)
if cuda:
    sample_noise = sample_noise.cuda()
sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)

# Load data.
LOGGER.info('Loading audio data...')
audio_paths = get_all_audio_filepaths(audio_dir)
train_data, valid_data, test_data, train_size = split_data(audio_paths, VALID_RATIO,
                                                           TEST_RATIO, batch_size)
TOTAL_TRAIN_SAMPLES = train_size
BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

train_iter = iter(train_data)
valid_iter = iter(valid_data)
test_iter = iter(test_data)

# =============Train===============
history = []
D_costs_train = []
D_wasses_train = []
D_costs_valid = []
D_wasses_valid = []
G_costs = []

start = time.time()
LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))

for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D_cost_train_epoch = []
    D_wass_train_epoch = []
    D_cost_valid_epoch = []
    D_wass_valid_epoch = []
    G_cost_epoch = []
    for i in range(1, BATCH_NUM+1):
        # Set Discriminator parameters to require gradients.
        for p in netD.parameters():
            p.requires_grad = True
  
        one = torch.tensor(1, dtype=torch.float)
        neg_one = one * -1
        if cuda:
            one = one.cuda()
            neg_one = neg_one.cuda()
        #############################
        # (1) Train Discriminator
        #############################
        for iter_dis in range(5):
            netD.zero_grad()

            # Noise
            noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
            if cuda:
                noise = noise.cuda()
            noise_Var = Variable(noise, requires_grad=False)

            real_data_Var = numpy_to_var(next(train_iter)['X'], cuda)

            # a) compute loss contribution from real training data
            D_real = netD(real_data_Var)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_data_Var.data,
                                                     fake.data, batch_size, lmbda,
                                                     use_cuda=cuda)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            #############################
            # (2) Compute Valid data
            #############################
            netD.zero_grad()

            valid_data_Var = numpy_to_var(next(valid_iter)['X'], cuda)
            D_real_valid = netD(valid_data_Var)
            D_real_valid = D_real_valid.mean()  # avg loss

            # b) compute loss contribution from generated data, then backprop.
            fake_valid = netG(noise_Var)
            D_fake_valid = netD(fake_valid)
            D_fake_valid = D_fake_valid.mean()

            # c) compute gradient penalty and backprop
            gradient_penalty_valid = calc_gradient_penalty(netD, valid_data_Var.data,
                                                           fake_valid.data, batch_size, lmbda,
                                                           use_cuda=cuda)
            # Compute metrics and record in batch history.
            D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
            D_wass_valid = D_real_valid - D_fake_valid

            if cuda:
                D_cost_train = D_cost_train.cpu()
                D_wass_train = D_wass_train.cpu()
                D_cost_valid = D_cost_valid.cpu()
                D_wass_valid = D_wass_valid.cpu()

            # Record costs
            D_cost_train_epoch.append(D_cost_train.data.numpy())
            D_wass_train_epoch.append(D_wass_train.data.numpy())
            D_cost_valid_epoch.append(D_cost_valid.data.numpy())
            D_wass_valid_epoch.append(D_wass_valid.data.numpy())

        #############################
        # (3) Train Generator
        #############################
        # Prevent discriminator update.
        for p in netD.parameters():
            p.requires_grad = False

        # Reset generator gradients
        netG.zero_grad()

        # Noise
        noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
        if cuda:
            noise = noise.cuda()
        noise_Var = Variable(noise, requires_grad=False)

        fake = netG(noise_Var)
        G = netD(fake)
        G = G.mean()

        # Update gradients.
        G.backward(neg_one)
        G_cost = -G

        optimizerG.step()

        # Record costs
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy())

        if i % (BATCH_NUM // 5) == 0:
            LOGGER.info("{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                             i, BATCH_NUM,
                                                                                             D_cost_train.data.numpy(),
                                                                                             D_wass_train.data.numpy(),
                                                                                             G_cost.data.numpy()))

    # Save the average cost of batches in every epoch.
    D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
    D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
    D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
    D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs_train.append(D_cost_train_epoch_avg)
    D_wasses_train.append(D_wass_train_epoch_avg)
    D_costs_valid.append(D_cost_valid_epoch_avg)
    D_wasses_valid.append(D_wass_valid_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                "G_cost:{:.4f}".format(time_since(start),
                                       D_cost_train_epoch_avg,
                                       D_wass_train_epoch_avg,
                                       D_cost_valid_epoch_avg,
                                       D_wass_valid_epoch_avg,
                                       G_cost_epoch_avg))

    # Generate audio samples.
    if epoch % epochs_per_sample == 0:
        LOGGER.info("Generating samples...")
        sample_out = netG(sample_noise_Var)
        if cuda:
            sample_out = sample_out.cpu()
        sample_out = sample_out.data.numpy()
        save_samples(sample_out, epoch, output_dir)

    # Save model
    LOGGER.info("Saving models...")
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")

    torch.save(netG, netG_path)
    torch.save(netD, netD_path) 

LOGGER.info('>>>>>>>Training finished !<<<<<<<')

# Plot loss curve.
LOGGER.info("Saving loss curve...")
plot(D_costs_train, D_wasses_train,
          D_costs_valid, D_wasses_valid, G_costs, output_dir)

LOGGER.info("All finished!")