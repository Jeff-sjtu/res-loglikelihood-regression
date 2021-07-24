# Human Pose Regression with Residual Log-likelihood Estimation

[[`Paper`]()]
[[`Project Page`](https://jeffli.site/rle/)]

> [Human Pose Regression with Residual Log-likelihood Estimation]()  
> Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu  
> ICCV 2021 Oral  


### Installation
1. Install pytorch >= 1.1.0 following official instruction.
2. Install `rlepose`:
``` bash
pip install cython
python setup.py develop
```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi).
``` bash
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
4. Init `data` directory:
``` bash
mkdir data
```
5. Download [COCO](https://cocodataset.org/#download) data:
```
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Train from scratch
``` bash
./scripts/train.sh ./configs/256x192_res50_regress-flow.yaml train_rle
```

### Evaluation
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1YBHqNKkxIVv8CqgDxkezC-4vyKpx-zXK/view?usp=sharing).
``` bash
./scripts/validate.sh ./configs/256x192_res50_regress-flow.yaml ./coco-laplace-rle.pth
```

### Citing
If our code helps your research, please consider citing the following paper:
```
@inproceedings{li2021human,
    title={Human Pose Regression with Residual Log-likelihood Estimation},
    author={Li, Jiefeng and Bian, Siyuan and Zeng, Ailing and Wang, Can and Pang, Bo and Liu, Wentao and Lu, Cewu},
    booktitle={ICCV},
    year={2021}
}
```