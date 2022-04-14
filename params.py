import os 

##### DATA #################

AUDIO_DIR = "data/"
OUTPUT_DIR = "output/"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

##### HYPER-PARAMETERS #####

EPOCHS = 3 # Number of epochs of training 
BATCH_SIZE = 16 # Batch size of DataLoader 
MODEL_SIZE = 64 # Model size parameter of WaveGAN 
SHIFT_FACTOR = 2 # Maximum shift used by phase shuffle 
PHASE_SHUFFLE_BATCHWISE = 'store_true' # If true, apply phase shuffle to entire batches rather than individual samples
POST_PROC_FILT_LEN = 512 # Length of post processing filter used by generator. Set to 0 to disable.
ALPHA = 0.2 # Slope of negative part of LReLU used by discriminator
VALID_RATIO = 0.1 # Ratio of audio files used for validation
TEST_RATIO = 0.1 # Ratio of audio files used for testing
NGPUS = 4 # Number of GPUs to use for training
LATENT_DIM = 128 # Size of latent dimension used by generator

EPOCHS_PER_SAMPLE = 1 # Generate audio samples every 1 epoch.
SAMPLE_SIZE = 10  # Generate 10 samples every sample generation.
LMBDA = 10.0 # Gradient penalty regularization factor
LEARNING_RATE = 1e-4 # Initial ADAM learning rate
BETA1 = 0.5 # beta_1 ADAM parameter
BETA2 = 0.9 # beta_2 ADAM parameter



