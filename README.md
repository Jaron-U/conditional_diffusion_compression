# Conditional Diffusion Compresion

A Reporece of the project "Conditional Diffusion Compression"
Original project Link: https://github.com/buggyyang/CDC_compression

## requirements
```bash
conda env create -f environment.yml
```

## Dataset
The dataset is from vimeo-90k dataset. http://toflow.csail.mit.edu/

Due to the server's file number limitation, I converted the images to NumPy format and saved them in an h5 file while extracting the dataset. As a result, all the images are stored in `train.h5` and `val.h5`.
```bash
# Download the dataset
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip

# unzip the dataset
# need to change the path variable in the unzip_vimeo.py
python data/pre_process_dataset/unzip_vimeo.py
```

## Training
```bash
python train.py
```

## Testing
```bash
python test_xparam.py
```



