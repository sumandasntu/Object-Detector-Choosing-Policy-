# Object-Detector-Choosing-Policy
Clone the directiory with *git clone git@github.com:sumandasntu/Interpretable-Meteorological-OOD.git*

Change Directory with *cd Interpretable-Meteorological-OOD*
## Preparation of Customize Dataset
Put all the image data in a single folder. Then run the scripts for data-splitting and data augmentation (rain-I and lightness). 
### Data splitting 
Run the script *python Data_splitting.py --path (Path of the image dataset e.g.,/home/Data_set_for_WVAE/CARLA/image_data)* from the folder Data Splitting and Augmentation.
### Data augmentation
Augment the training and test data with different levels of rain-I(or rain-II) and lightness with the following command. 
Run the script *python Data_augmentation_I.py/Data_augmentation_II.py --path (path of the parent folder containing Train and test data, e.g., /home/Data_set_for_WVAE/CARLA) --percentage (percentage of labelled data e.g., 25)* from the folder Data Splitting and Augmentation.
## Meteorological Intensity Predictor (MIP)
For the MIP please run the Jupyter notebook: Intensity_predictor_full_latent_dimension-Working-Final.ipynb
