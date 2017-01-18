# split image dataset into test, train, validation and sample directories for consumption into keras deep learning library

'''
understand the dataset

ls test/ | wc -l   # 12500 images in test set
ls train/ | wc -l  # 25000 images in training set

ls kaggle_dogscats/train/ | less  
ls kaggle_dogscats/train/ | tail   # i see that in the training set, images are in the format class(dog/cat).image_id(986660).jpg

ls kaggle_dogscats/test/ | tail    # in the test set though, images are in the format image_id(986660).jpg

ls kaggle_dogscats/train/ | grep 'dog' | wc -l    # 12500 images of dog in the training set
ls kaggle_dogscats/train/ | grep 'cat' | wc -l    # 12500 images of cat in the training set

so I'll use uniform distribution for shuffling and sampling the training data. 
'''
