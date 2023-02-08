_base_ = [
    '../_base_/datasets/cifar100_bs16.py',
]

# model settings
model = dict(
    type="Distiller",
    teacher=dict(type='AlexNet', num_classes=10),
    student=dict(type='AlexNet', num_classes=10),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

work_dir = "work_dirs"

# dataset settings
# data这里直接继承cifar100试一下

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR',  # learning policy, decay on several milestones.
    by_epoch=True,  # update based on epoch.
    milestones=[15],  # decay at the 15th epochs.
    gamma=0.1,  # decay to 0.1 times.
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()

# runtime settings
default_scope = 'mmcls'

default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 150 iterations.
    logger=dict(type='LoggerHook', interval=150),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    # disable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume the training of the checkpoint
resume_from = None

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (1 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
