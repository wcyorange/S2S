from easydict import EasyDict

cfgs = {
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'z_dim': 32,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': True,
    # dataset
    'anime': False,
    'dirA': '/root/wcy/Morph-UGATIT-self/dataset/skull2skinall/trainA_paired',
    'dirB': '/root/wcy/Morph-UGATIT-self/dataset/skull2skinall/trainB_paired',
    'direction': 'AtoB',
    'load_size': 256,
    'batchsize': 1,
    'worker': 0,
    # training
    'total_epoch': 101,
    'tensorboard': '/root/wcy/Morph-UGATIT-self/log',
    'saved_dir': '/root/wcy/Morph-UGATIT-self/ckpt',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 2e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,
    'lambda_similarity': 1.0,  # 或许可以更小
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'z_dim': 32,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': False,
    # 'saved_dir': '/share/yangjie08/result_ugatit',


})