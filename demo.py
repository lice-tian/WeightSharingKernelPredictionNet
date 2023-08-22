from network_train import DataHandler, init_seed

init_seed(2022)

data_dir = 'data'
scenes = ['scene1', 'scene2', 'scene3', 'scene4', 'scene5', 'scene6']

print('加载训练集')
train_data = DataHandler(data_dir=data_dir,
                         subset='train',
                         scene_list=scenes,
                         patch_per_img=80,
                         patch_width=128,
                         patch_height=128)

print('加载验证集')
valid_data = DataHandler(data_dir=data_dir,
                         subset='valid',
                         scene_list=scenes,
                         patch_per_img=20,
                         patch_width=128,
                         patch_height=128)
