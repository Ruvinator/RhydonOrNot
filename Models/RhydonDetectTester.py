import numpy as np
from RhydonDetect import train_rhydon_detect, is_this_rhydon

# Training model
# accuracy = train_rhydon_detect(
#     _test_pct=0.2,
#     _checkpoint_name='2cnn_tr80',
#     _checkpoint_dir='RhydonMate/Models/training/ckpt/ckpt',
#     )

# Inferencing model
inf_result = is_this_rhydon(
    _load_model_dir='RhydonMate/Models/training/ckpt/ckpt_20191214-0116/2cnn_tr80.ckpt',
    _image_dir='RhydonMate/Data/UE4/Charizard_Pose1_1_191_g.png',
    _use_greyscale=True,
    _normalize=True)

is_rhydon = inf_result[1] > inf_result[0]
print('There is', 'a' if is_rhydon else 'no', 'Rhydon in that image (%.1f %% confidence).' 
      % (inf_result[int(is_rhydon)] * 100))
