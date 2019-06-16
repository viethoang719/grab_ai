# grab_ai
Repository to submit Grab AI for SEA June 2019

1.  Submission.txt is result for prediction of all test images in cars_test_annos.mat
1.  How to run predict.
    * Instal following libraries:\
    pip install -q tensorflow==2.0.0-beta0\
    pip install -q scipy
    * populate folder cars_test with all images listed in cars_test_annos.mat
    
    * Run predict.py  
      predict.py --cars_test=path/to/cars/test/folder --cars_test_annos=path/to/cars/test/annos.mat/ 
      
      Example: predict.py --cars_test=.\\cars_test\\ --cars_test_annos=.\\devkit\\cars_test_annos.mat
    * Check out result  in file submission.txt 
1.  How did I train model for Grab challenge.
    * I was inspired by transfer learning as example from Tensorflow.
    https://www.tensorflow.org/alpha/tutorials/images/transfer_learning
    * My code for training is in jupyter notebook GrabAI.ipynb. In order to run this trainning, you have to install tensorflow tensorflow-gpu==2.0.0-beta1.
