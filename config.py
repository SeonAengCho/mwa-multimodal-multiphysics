# All settings used by Train_Temp.py, Train_Dmg.py, Test_Temp.py and Test_Dmg.py.

# ---- Model selection (one model per task) ----
# Train_* trains this model and saves weight/<model name>.pth;
# Test_* builds the same model and loads that file.
#
# Available names: AffineUNet1, AffineUNet2, AffineUNet3, ConcatNet,
#                  CrossAttenUNet8, CrossAttenUNet12, CrossAttenUNet16
# CrossAttenUNet* also take: type="skipall", proj=0, encoder="CNN" or "ViT"
#   e.g. {"name": "CrossAttenUNet12", "type": "skipall", "proj": 0, "encoder": "CNN"}
temp_model = {"name": "AffineUNet3"}
dmg_model = {"name": "AffineUNet3"}

# ---- Paths (relative to the folder you run the scripts from) ----
data_dir = "Dataset"      # input / label .npy files
weight_dir = "weight"     # trained weights (.pth)
result_dir = "Results"    # example prediction images from Test_*

# ---- Training hyperparameters ----
learning_rate = 0.0001    # same for the temperature and damage models
epoch_num = 100
batch_size = 16
device = 'cuda:0'

# ---- Random seeds ----
train_seed = None                     # Train_*: None = random each run (the seed used is printed);
                                      #          set an integer (e.g. 42) to fix it
valid_seed = None                     # Train_*: which 1,080 validation samples are used;
                                      #          None = random each run (printed), integer = fixed
test_seed = {"temp": 42, "dmg": 42}   # Test_*: which 1,080 unforeseen test samples are used
