from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

class ImageCheckpointCallback(Callback):
    def __init__(self, save_frequency, batch_size):
        self.save_frequency = save_frequency
        self.batch_size = batch_size
        self.image_count = 0

    def on_batch_end(self, trainer, pl_module):
        # Increment image count by the batch size used during training
        self.image_count += self.batch_size
        # Check if it's time to save a checkpoint
        if self.image_count % self.save_frequency == 0:
            checkpoint_path = f"checkpoint_{self.image_count}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at {self.image_count} images.")



# Configs
resume_path = './models/control_sd21_ini.ckpt'

# was 4
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True

# What is this param and where is it used 
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# why is this in env is is python lightning method 
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
# Modify the Trainer instantiation
# Update the instantiation of your trainer to include this custom callback
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    callbacks=[
        logger,
        ImageCheckpointCallback(save_frequency=250, batch_size=batch_size)
    ]
)



# Train!
trainer.fit(model, dataloader)

#where do I see the model architecture, where can I see the training like loss etc 
