# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

#%% [markdown]
# To improve upon the baseline CNN image classification model, we will be 
# using Transfer Learning. We will be using an already trained model (trained 
# on ImageNet), and adapt it to our current image dataset. 
