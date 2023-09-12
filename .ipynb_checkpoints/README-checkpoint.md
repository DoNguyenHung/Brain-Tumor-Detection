# CS615: Final Project - Brain Tumor Detection with MultiClass CNN's

Kenneth Pan <kp3258@drexel.edu> <br>
Amira Bendjama <ab4745@drexel.edu> <br>
Ramona Rubalcava: <rlr92@drexel.edu> <br>
Hung Do <hd386@drexel.edu> <br>

---

## Code Usage:

This code was created to satisfy the work necessary for CS615 Final Project

#### - Data:

The data used in this project was retrieved from Kaggle at this link:<br>
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset <br>

Due to the size of the dataset, the dataset was unable to be attached to this submission. To recreate the folder structure of the dataset used for training and testing:

1. Download the data from Kaggle
2. Unzip from the archive folder 
3. Place the unzipped folders _Training_ and _Testing_ in a new folder named _full_
4. place this folder in the same directory as the Preprocessing.py script
5. Run the Preprocessing.py script 
6. Verify that the dataset results in trainingfull.csv, trainingfull_labels.csv, testingfull.csv, testingfull_labels.csv

#### - Folder structure:

The file strucutre is as follows:

> final_submission <br>
> ├── Preprocessing.py - the script used to resize the images in the brain tumor MRI dataset <br>
> ├── layers.py - the layers file used for construction of the MLP and CNN architectures <br>
> ├── MLP.ipynb - the noteboook used to run the MLP network and generate graphs <br>
> ├── CNN.py - the script used to train and validate the CNN architecture <br>
> ├── CNN-2.py - the script used to train and validate the CNN architecture with 2 convolutional layers <br>
> ├── if needed <br>
> └── if needed <br>

#### - Running the code:

The only packages necessary for running the scripts are below:

pip install pandas <br>
pip install numpy <br>
pip install matplotlib <br>
pip install cv2 <br>
pip install imutils <br>
pip install tqdm <br>



---