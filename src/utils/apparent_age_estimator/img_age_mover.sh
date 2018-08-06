#!/bin/bash

# Replace the following line:
AAE_ROOT=/src/misc/AAE_CVPR2016_OrangeLabs/
RESIZED_IMG_DIR=$1/imgs_224x224

echo "Img target is $1. Ensure there is NO trailing /. If incorrect, Ctrl+C now. Ensure imagemagick is installed. NOTE: All images will be resized."

# Comment the next row out for the final version
read -p "Press enter"

echo "Create a resized set under $1..."
cd $1
mkdir imgs_224x224/
rm imgs_224x224/*.png
rm imgs_224x224/*.jpg
cp *.png  imgs_224x224/
cp *.jpg  imgs_224x224/
cd  imgs_224x224/

for f in *.png; do
  convert $f -resize 224x224 "$f"_224.png 
  rm $f ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVES ORIGS
done
for f in *.jpg; do
  convert $f -resize 224x224 "$f"_224.png
  rm $f ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVES ORIGS
done

echo "Resize done. Create input metadata for the estimator... (remove old input files first)"
#read -p "Press enter"

cd $AAE_ROOT
rm img_pointers_*
rm img_names_*
cd General_age_estimation
python3 make_input_files.py $RESIZED_IMG_DIR

echo "Metadata created. Remove old result file..."
#read -p "Press enter"
rm result_model_1.txt
echo "Run the estimator..."
#read -p "Press enter"
cd ..
python ESTIMATE_AGE_small.py

echo "Estimator complete. Find mean..."
#read -p "Press enter"

python3 relocate_by_ages.py -f AAE_CVPR2016_OrangeLabs/General_age_estimation/result_model_1.txt -imgdir $RESIZED_IMG_DIR

echo "If you want to remove the resized files, run:"
echo "rm -rf $RESIZED_IMG_DIR; rmdir $RESIZED_IMG_DIR"
echo "All done."
