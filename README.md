# SR_Tensor
An implementation of 3D brain MRI super-resolution method by image gradient-tensor distance based patch clustering

### Conference Abstract
 - 3D Brain MRI Super-Resolution with Image Gradient Tensor Feature Clustering [[Poster]](https://github.com/Snailpong/SR_Tensor/files/7373293/ohbm.pdf)
  
 - Organization for Human Brain Mapping Annual Meeting, 2021


## Requirments

 - Both Linux and Windows are supported.

 - Package Required: numpy, numba, scipy, skimage, nibabel



## Prepare datasets

 - We got young-adult T1-weighted masked MRI brain Dataset from 'Human Connectome Project' (https://www.humanconnectome.org/study/hcp-young-adult)

 - We used HCP-900 dataset with 30 images to train, 867 images to estimate metrics.

 - Store your HR train data to 'train' folder, HR test data to 'test' folder. In test stage, data will be downscaled to estimate HR image.



## Folder Structure Example
```
.
├── train
|   ├── T1w_acpc_dc_restore_brain_id1.nii.gz
|   └── T1w_acpc_dc_restore_brain_id2.nii.gz
├── test
|   ├── T1w_acpc_dc_restore_brain_id3.nii.gz
|   └── T1w_acpc_dc_restore_brain_id4.nii.gz
├── result
|   └── 110521
|       ├── T1w_acpc_dc_restore_brain_id3.nii.gz
|       ├── T1w_acpc_dc_restore_brain_id4.nii.gz
├── arrays
|   ├── h_2x_1023.npy
|   └── space_2x_1023.km
├── train.py
├── test.py
├── feature_model.py
├── filter_constant.py
├── kmeans_vector.py
├── preprocessing.py
├── filter_func.py
├── matrix_compute.py
└── util.py
```


## Running
 - Run following commmand to start training.

```
git clone https://github.com/Snailpong/SR_Tensor.git
cd SR_Tensor
python train.py
```
 - Run following commmand to get upscaled images.

```
python test.py
```


## Result Visualization
![Result](https://user-images.githubusercontent.com/11583179/113809609-772f8b00-97a3-11eb-89a0-4bcf40294e72.png)


## Code References
  - https://github.com/Snailpong/MRI_Super_Resolution_3D_Gradient_Filter_Learning.git



## License
GNU General Public License 3.0 License
