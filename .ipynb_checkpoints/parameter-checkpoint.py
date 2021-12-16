pretrain_epochs = 50
posttrain_epochs = 80
workers=1
log_interval = 50
batch_size_train = 256
batch_size_test = 2000
batch_size_class = 256
use_bias=True
n_classes = 10
num_classes = n_classes
non_binary_models =[]
mnist ='mnist'
cifar10 ='cifar10'
fmnist = 'fmnist'
we = 0.001


fmnist_p = {'n_epochs_post': 50,
    'lr_post': 0.01,
    'reg_weight': 4e-05,
    'lr_prox_64': 0.05,
    'lr_prox_512': 0.02,
    'lr_prox_1152': 0.01,
    'margin_64': 0.9,
    'margin_512': 0.6,
    'margin_1152':0.4,
    'sl_64': 0.24,
    'sl_512': 0.044,
    'sl_1152': 0.01,
    'lamda_64': 0.04,
    'lamda_512': 0.008,
    'lamda_1152': 0.001}





mnist_p = {'n_epochs_post': 300,
    'lr_post': 0.001,
    'reg_weight': 4e-5,
    'lr_prox_64': 0.1,
    'lr_prox_512': 0.1,
    'lr_prox_1152': 0.1,
    'lr_conprox_64': 0.001,
    'lr_conprox_512': 0.001,
    'lr_conprox_1152': 0.001,
    'margin_64': 0.95,
    'margin_512': 0.95,
    'margin_1152': 0.95,
    'sl_64': 0.24,
    'sl_512': 0.044,
    'sl_1152': 0.01,
    'lamda_64': 0.04,
    'lamda_512': 0.008,
    'lamda_1152': 0.001,
    'lr_br_64':0.1,
    'lr_br_512':0.1,
    'lr_br_1152':0.1,}

