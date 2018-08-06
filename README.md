# Recursive Chaining of Reversible Image-to-image Translators

![alt text](samples/age_15_id_2B.jpg)
![alt text](samples/age_25_id_2B.jpg)
![alt text](samples/age_35_id_2B.jpg)
![alt text](samples/age_45_id_2B.jpg)
![alt text](samples/age_55_id_2B.jpg)
![alt text](samples/age_65_id_2B.jpg)

Code for the paper ["Recursive Chaining of Reversible Image-to-image Translators For Face Aging"](https://arxiv.org/abs/1802.05023) [1].

Implementation by Ari Heljakka. The foundation for the main script was adapted from https://github.com/LynnHo/CycleGAN-Tensorflow-Simple which implements the paper [2].


## Summary

The primary code base allows you to train image-to-image transformers sequentially, and supports certain experimental modes not mentioned in the paper.
The code is agnostic as to the actual type of transformation, but many of the extensions are useful only if you work on sequential transformations.

In order to reproduce the face aging results in the paper, you also have to run the data pre-processing steps.

The focus of the paper was in evaluating the recursive chaining of the transformers, and there is much room to optimize the actual image quality.

## Dependencies

The separate tools described in their own subsections below may need to be isolated to separate virtual environments.

Ubuntu 16.04 and Tensorflow 1.1.0-rc0 was used for all scripts.

Python 2.7 is used for the main script and "Apparent age estimator" [6]

Python 3.5.2 is recommended for the FaceNet [5] version used here (other Python 3.5 versions may work as well)

## Data pre-processing
### Original Datasets

For training data [3], download CACD dataset from http://bcsiriuschen.github.io/CARC/  - select the "Original face images (detected and croped by openCV face detector".

For test data [4], download IMDB-Wiki dataset (actually only IMDB) from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ - from "IMDB", select "Download Faces only"

The main script expects all datasets to reside under `datasets/` directory. E.g. for dataset called `celeb2`, with two age groups `age25` and `age35`, there should be the following directories:
```
datasets/celeb2/train_age25
datasets/celeb2/train_age35
datasets/celeb2/test_age25
datasets/celeb2/test_age35
```
We show how to preprocess the CACD data. You can apply the exact same method to the IMDB data. You only need a small subset of the latter, of course.

Be warned that the process is lengthy. On the positive side, you can use this data pipeline to categorize any face dataset by the apparent age.

### Face alignment & cropping

Here, we install FaceNet [5] for face alignment and cropping to 132x132. The instructions apply for davidsandberg implementation, commit [60e34476a](https://github.com/davidsandberg/facenet/tree/60e34476a676fb47cb7b93d6d25fe6a919b3543d). (You can use any modern face alignment tool. E.g. the [Adience](https://github.com/eranid/adience_align) face alignment tool has a rather difficult dependency chain.)

1. Get FaceNet:
```
git clone https://github.com/davidsandberg/facenet.git
cd facenet
git checkout 60e34476a
```
2. Remove tensorflow version specification from requirements.txt if you run this at the same environment as the other components
3. Add dlib to requirements
4. In Ubuntu, installing dlib may require: `apt-get install -y libboost-python-dev cmake`
5. Run:
```
pip3 install -r requirements.txt
cp src/align/align_*.py .
# Download the landmarks data file separately, e.g. as follows:
wget "https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true" -O data/shape_predictor_68_face_landmarks.dat
```

Run the alignment for the whole training (and testing) dataset as follows. Please note that the images must reside on a subdirectory under `original-images-directory-root`, so that e.g. if your images are under img/celeb2, then your `original-images-directory-root` is `img`.
```
python3 align_dataset.py --face_size=115 --image_size=132 [original-images-directory-root] [aligned-images-directory]
```

### Auxiliary age estimator

The Apparent age estimator or AAE [6] is needed to split your dataset according to each subject's age, and/or to evaluate ages of transformed faces. You need to install the estimator, wrap it with utility scripts, and run it on the input images. Note: AAE itself takes a while to run, and with our scripts, it only takes 50 images at a time (as higher numbers may run out of memory).

We provide you with our utility scripts to cover the whole pipeline, but note that it is fragile and, in practise, it is definitely not guaranteed to work for every environment as-is. Still, you should be able to use it with reasonable amount of adjustments by meticulously following these instructions.

To re-annotate the whole dataset, img_age_mover.sh script will have to read in the images one by one, measure the apparent age, create the appropriate sub-directory for each age, and copy the file over to the proper directory based on the age. Also, for compatibility between the tools, the images have to be resized in the process.

Set up our toolchain:

1. Request access from the authors of "Apparent age estimator" [6], as described in https://cactus.orange-labs.fr/apparent-age-estimation/index.php?p=3
and extract the files, resulting in directory AAE_CVPR2016_OrangeLabs.
2. Copy our changes on top of the original estimator, as (assuming you extracted the files at current directory):
```
cp -R src/utils/apparent_age_estimator/AAE_CVPR2016_OrangeLabs/* AAE_CVPR2016_OrangeLabs/
```
3. Add the directory under which the resulting images should be stored directly into the `maindir` configuration variable in the script `src/utils/apparent_age_estimator/AAE_CVPR2016_OrangeLabs/relocate_by_ages.py` 
4. Add the full path of your AAE_CVPR2016_OrangeLabs directory in the `AAE_ROOT` variable at the beginning of both `img_age_estimator.sh` and `img_age_mover.sh`


Set up dependencies for [6]:
```
apt-get install -y imagemagick libgoogle-glog-dev liblmdb-dev

# Install Caffe from sources, following the standard instructions. (Confirmed to work with commit 7e970675c41f897a05fe1a944d754bd899fbff0e)
# If your 'make all' step fails with error "/usr/bin/ld: cannot find -lhdf5_hl", you can try these steps in Ubuntu (modify the exact paths and version numbers to suit your system):
cd /usr/lib/x86_64-linux-gnu
ln -s libhdf5_serial.so.10 libhdf5.so
ln -s libhdf5_serial_hl.so.10 libhdf5_hl.so
# remember to update your PYTHONPATH to find the caffe module

# Install hdf5 and h5py. In Ubuntu 16.04, as follows:
apt-get update
apt-get install -y python-gflags libgflags-dev  libhdf5-dev libgoogle-glog-dev python-h5py
cd caffe
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;

# Install OpenCV:

apt-get install python-skimage libopencv-dev python-opencv

```

To re-annotate the dataset according to the apparent age, run the following. (The re-annotation takes place by subdirectory structure, so that a face with apparent age of e.g. 32 years is moved into the directory `age32`.)
```
cd src/utils/apparent_age_estimator
./img_age_mover.sh [aligned-images-directory]
```

To only evaluate the ages in images-directory and place the results in mean_ages.txt, run
```
cd src/utils/apparent_age_estimator
./img_age_estimator.sh [images-directory]
```

In order to divide the dataset into only 6 separate bins as in the paper, you then simply create a directory for each bin (e.g. `train_age35` for the range 30-39) and just populate it with a random subset of images on that range (i.e. under `age30`, `age31`, etc. until `age39`, so that the mean age remains at 35). For ranges 2-18 and 60+ you should use as many images as possible. For others, at least 5000 should be used.


## Main Script

### Dependencies

```
sudo apt-get install python-tk
cd temporal-generative-nets/src
pip install -r requirements.txt
```

### Usage


The design of the script assumes that you will be creating multiple transformer networks on the same dataset, here called 'subnets'.  For the first transformation, you train a mapping from one stage (e.g. age15) to another (e.g. age25). Once you try to re-train the model for the same transformation, you need a "triplet" of stages (e.g. also age35).

For testing a recursive transformation N times, you actually have to run the transformer N times, simply setting the output of the previous run as the input of the next run. (In the 6 age domains, you can have 5 transformers, so your maximum N = 4.)

To see all command-line parameters, run
```
python train.py -h
```

The script supports also two special experimental modes not used for the main results in the paper:
* `double_cycle` adds a loss constraint to ensure that if you run a transformer twice in succession, and then twice in reverse, you get back the original
* `transform_twice` during training, run the transformer twice for each transform step, enforcing robustness for recursive usage

Note that train.py loads the latest checkpoint of your subnet by default, if it exists.

### Training Examples

Train the first transformer, ages 15->25:

```
python train.py --save=1 --dataset celeb2_ref --stage1=_age15 --stage2=_age25 --max_steps=600000 --subnet tf1525
```

Note: You may want to make a backup of the final checkpoint of the trained model, if you will be retraining that transformer! You may also want to feed its filename later explicitly with `--prev_checkpoint=[filename]` when you retrain it, to keep track.

Train the second transformer, ages 25->35:

```
python train.py --save=1 --dataset celeb2_ref --stage1=_age25 --stage2=_age35 --max_steps=600000 --subnet tf2535
```

Retrain the 15->25 transformer on 25->35 transformation (Note that epochs are counted on top of the training epochs already done, so that if you started with 60 epochs, adding 30 more epochs requires setting `max_epochs=90`. Also, note that 1 epoch now trains both transformations, so that the number of training steps for these 30 new epochs is the same as the number of steps for the previous 60 epochs):

```
python train.py --save=1 --dataset celeb2_ref --stage1=_age15 --stage2=_age25 --stage3=_age35 --max_steps=900000 --subnet tf1525 --triplet
```

### Testing Examples

Test the single-trained 15->25 transformer on 15->25 transform (give your checkpoint file name as `prev_checkpoint` argument):

```
python train.py --save=0 --dataset celeb2_ref --stage1=_age15 --stage2=_age25 --subnet tf1525 --singletestdir /data/imdb_aed/age_15 --singletestdir_out tf1525_from15 --singletestN 100 --prev_checkpoint="Epoch_(113)_(2117of5291)_step_(599999)"
```

(There is a known minor issue that the output images randomly get horizontally flipped during testing.)

Test the single-trained 25->35 transformer on 25->35 transform (you have only one such model, so you don't have to specify the `prev_checkpoint`):

```
python train.py --save=0 --dataset celeb2_ref --stage1=_age25 --stage2=_age35 --subnet tf2535 --singletestdir /data/imdb_aed/age_25 --singletestdir_out tf2535_from25 --singletestN 100
```

Test the double-trained 15->25 transformer on 15->35 transform:

```
python train.py --save=0 --dataset celeb2_ref --stage1=_age15 --stage2=_age25 --subnet tf1525 --singletestdir /data/imdb_aed/age_15 --singletestdir_out tf1525_N1_from15 --singletestN 100 --prev_checkpoint="Epoch_(113)_(2117of5291)_step_(599999)"
python train.py --save=0 --dataset celeb2_ref --stage1=_age25 --stage2=_age35 --subnet tf1525 --singletestdir tf1525_N1_from15      --singletestdir_out tf1525_N2_from15 --singletestN 100 --prev_checkpoint="Epoch_(113)_(2117of5291)_step_(899999)"
```



## References

[1] A. Heljakka, A. Solin, J. Kannala, "Recursive Chaining of Reversible Image-to-image Translators For Face Aging," in Advanced Concepts for Intelligent Vision Systems, 2018, to appear.

[2] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks," in ICCV, 2017.

[3] B.-C. Chen, C.-S. Chen, and W. H. Hsu, "Cross-age reference coding for age-invariant face recognition and retrieval," in ECCV, 2014, pp. 768–783.

[4] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in ICCV, Looking at People Workshop, 2015.

[5] F. Schroff, D. Kalenichenko, and J. Philbin. "Facenet: A unified embedding for face recognition and clustering," arXiv preprint arXiv:1503.03832, 2015.

[6] G. Antipov, M. Baccouche, S. Berrani, and J. Dugelay, "Apparent age estimation from face images combining general and children-specialized deep learning models," in CVPR Workshops, 2016.

## License

This software is distributed under the MIT License; please refer to the file LICENSE, included with the software, for details.
