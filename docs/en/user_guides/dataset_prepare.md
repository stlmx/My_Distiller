# Prepare Dataset

MMClassification supports following datasets:

- [CustomDataset](#customdataset)
- [ImageNet](#imagenet)
- [CIFAR](#cifar)
- [MINIST](#mnist)
- [OpenMMLab 2.0 Standard Dataset](#openmmlab-20-standard-dataset)
- [Other Datasets](#other-datasets)
- [Dataset Wrappers](#dataset-wrappers)

If your dataset is not in the abvove list, you could reorganize the format of your dataset to adapt to **`CustomDataset`**.

## CustomDataset

[`CustomDataset`](mmcls.datasets.CustomDataset) is a general dataset class for you to use your own datasets. To use `CustomDataset`, you need to organize your dataset files according to the following two formats:

### Subfolder Format

The sub-folder format distinguishes the categories of pictures by folders. As follows, class_1 and class_2 represent different categories.

```text
data_prefix/
├── class_1     # Use the category name as the folder name
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
├── class_2
│   ├── 123.png
│   ├── 124.png
│   └── ...
```

Assume you want to use it as the training dataset, and the below is the configurations in your config file.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CustomDataset',
        data_prefix='path/to/data_prefix',
        pipeline=...
    )
)
```

```{note}
Do not specify `ann_file`, or specify `ann_file=None` if you want to use this method.
```

### Text Annotation File Format

The text annotation file format uses text files to store path and category information. All the images are placed in the folder of `data_prefix`, and `ann_file` contaions all the ground-truth annotation.

In the following case, the dataset directory is as follows:

```text
data_root/
├── meta/
│   ├── train_annfile.txt
│   ├── val_annfile.txt
│   └── ...
├── train/
│   ├── folder_1
│   │   ├── xxx.png
│   │   ├── xxy.png
│   │   └── ...
│   ├── 123.png
│   ├── nsdf3.png
│   └── ...
├── val/
└── ...
```

Assume you want to use the training dataset, and the annotation file is `train_annfile.txt` as above. The annotation file contains ordinary text, which is divided into two columns, the first column is the image path, and the second column is the **index number** of its category:

```text
folder_1/xxx.png 0
folder_1/xxy.png 1
123.png 1
nsdf3.png 2
...
```

```{note}
The index numbers of categories start from 0. And the value of ground-truth labels should fall in range `[0, num_classes - 1]`.
```

In the annotation file, we only specified the category index of every sample, you also need to specify `classes` field in the dataset config to record the name of every category:

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',
        ann_file='meta/train_annfile.txt',
        data_prefix='train',
        classes=['A', 'B', 'C', 'D', ...],
        pipeline=...,
    )
)
```

```{note}
If the `ann_file` is specified, the dataset will be generated by the the ``ann_file``. Otherwise, try the first way.
```

## ImageNet

ImageNet has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). It can be accessed with the following steps.

1. Register an account and login to the [download page](http://www.image-net.org/download-images).
2. Find download links for ILSVRC2012 and download the following two files
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. Untar the downloaded files
4. Download and untar the meta data from this [link](https://download.openmmlab.com/mmclassification/datasets/imagenet/meta/caffe_ilsvrc12.tar.gz).
5. Re-organize the image files according to the path in the meta data, and it should be like:

```text
   imagenet/
   ├── meta/
   │   ├── train.txt
   │   ├── test.txt
   │   └── val.txt
   ├── train/
   │   ├── n01440764
   │   │   ├── n01440764_10026.JPEG
   │   │   ├── n01440764_10027.JPEG
   │   │   ├── n01440764_10029.JPEG
   │   │   ├── n01440764_10040.JPEG
   │   │   ├── n01440764_10042.JPEG
   │   │   ├── n01440764_10043.JPEG
   │   │   └── n01440764_10048.JPEG
   │   ├── ...
   ├── val/
   │   ├── ILSVRC2012_val_00000001.JPEG
   │   ├── ILSVRC2012_val_00000002.JPEG
   │   ├── ILSVRC2012_val_00000003.JPEG
   │   ├── ILSVRC2012_val_00000004.JPEG
   │   ├── ...
```

And then, you can use the [`ImageNet`](mmcls.datasets.ImageNet) dataset with the below configurations:

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        ann_file='meta/train.txt',
        data_prefix='train/',
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        ann_file='meta/val.txt',
        data_prefix='val/',
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## CIFAR

We support downloading the [`CIFAR10`](mmcls.datasets.CIFAR10) and [`CIFAR100`](mmcls.datasets.CIFAR100) datasets automatically, and you just need to specify the
download folder in the `data_root` field. And please specify `test_mode=False` / `test_mode=True`
to use training datasets or test datasets.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=False,
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=True,
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## MNIST

We support downloading the [MNIST](mmcls.datasets.MNIST) and [Fashion-MNIST](mmcls.datasets.FashionMNIST) datasets automatically, and you just need to specify the
download folder in the `data_root` field. And please specify `test_mode=False` / `test_mode=True`
to use training datasets or test datasets.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=False,
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=True,
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## OpenMMLab 2.0 Standard Dataset

In order to facilitate the training of multi-task algorithm models, we unify the dataset interfaces of different tasks. OpenMMLab has formulated the **OpenMMLab 2.0 Dataset Format Specification**. When starting a trainning task, the users can choose to convert their dataset annotation into the specified format, and use the algorithm library of OpenMMLab to perform algorithm training and testing based on the data annotation file.

The OpenMMLab 2.0 Dataset Format Specification stipulates that the annotation file must be in `json` or `yaml`, `yml`, `pickle` or `pkl` format; the dictionary stored in the annotation file must contain `metainfo` and `data_list` fields, The value of `metainfo` is a dictionary, which contains the meta information of the dataset; and the value of `data_list` is a list, each element in the list is a dictionary, the dictionary defines a raw data, each raw data contains a or several training/testing samples.

The following is an example of a JSON annotation file (in this example each raw data contains only one train/test sample):

```json
{
    'metainfo':
        {
            'classes': ('cat', 'dog'), # the category index of 'cat' is 0 and 'dog' is 1.
            ...
        },
    'data_list':
        [
            {
                'img_path': "xxx/xxx_0.jpg",
                'img_label': 0,
                ...
            },
            {
                'img_path': "xxx/xxx_1.jpg",
                'img_label': 1,
                ...
            },
            ...
        ]
}
```

Assume you want to use the training dataset and the dataset is stored as the below structure:

```text
data
├── annotations
│   ├── train.json
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

Build from the following dictionaries:

```python
train_dataloader = dict(
    ...
    dataset=dict(
        type='BaseDataset',
        data_root='data',
        ann_file='annotations/train.json',
        data_prefix='train/',
        pipeline=...,
    )
)
```

## Other Datasets

To find more datasets supported by MMClassification, and get more configurations of the above datasets, please see the [dataset documentation](mmcls.datasets).

## Dataset Wrappers

The following datawrappers are supported in MMEngine, you can refer to {external+mmengine:doc}`MMEngine tutorial <advanced_tutorials/basedataset>` to learn how to use it.

- [ConcatDataset](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md#concatdataset)
- [RepeatDataset](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md#repeatdataset)
- [ClassBalanced](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md#classbalanceddataset)

The MMClassification also support [KFoldDataset](mmcls.datasets.KFoldDataset), please use it with `tools/kfold-cross-valid.py`.
