# TG-Pose (TMM 2024)
Pytorch implementation of TG-Pose: Delving into Topology and Geometry for Category-level Object Pose Estimation.
([Paper](https://ieeexplore.ieee.org/abstract/document/10539316/), [Project](https://sites.google.com/view/tg-pose/))


## Required environment
- Ubuntu 18.04
- Python 3.8 
- Pytorch 1.11.0
- CUDA ≥11.7


## Data Preparation
To generate your own dataset, use the data preprocess code provided in this [git](https://github.com/mentian/object-deformnet/blob/master/preprocess/pose_data.py). Download the detection results in this [git](https://github.com/Gorilla-Lab-SCUT/DualPoseNet). Change the `dataset_dir` and `detection_dir` to your own path.

Since the handle visibility labels are not provided in the original NOCS REAL275 train set, please put the handle visibility file `./mug_handle.pkl` under `YOUR_NOCS_DIR/Real/train/` folder. The `mug_handle.pkl` is mannually labeled and originally provided by the [GPV-Pose](https://github.com/lolrudy/GPV_Pose).


## Training
```shell
python -m engine.train --loss_list='TDA_loss' --run_stage='RL_TDA' --dataset_dir YOUR_DATA_DIR --model_save SAVE_DIR
```

Detailed configurations are in `config/config.py`.

## Evaluation
```shell
python -m evaluation.evaluate --dataset_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR --resume 1 --resume_model MODEL_PATH --model_save SAVE_DIR
```

|Metrics| IoU25 | IoU50 | IoU75 | 5d2cm | 5d5cm | 10d2cm | 10d5cm | 10d10cm |
|:------|:------|:------|:------|:------|:------|:-------|:-------|:--------|
|Scores | 84.3  | 82.6  | 76.2  | 49.8  | 59.0  | 71.7   | 86.6   | 87.7    |




## Citation
Cite us if you found this work useful.
```
@article{zhan2024tg,
  title={TG-Pose: Delving into Topology and Geometry for Category-level Object Pose Estimation},
  author={Zhan, Yue and Wang, Xin and Nie, Lang and Zhao, Yang and Yang, Tangwen and Ruan, Qiuqi},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```
