#This is my attempt at file handling. Should be fun! 
#Prototype the construction of the function that splits an image dataset into test, train, validation and sample datasets 
# in the following format 
#[thanks VedAustin](https://gist.githubusercontent.com/VedAustin/a6bdc1ea5dcc1c363053b1dd1b16e88c/raw/960b61ea0a5eec73563526afbaa17bbb9f1ecd08/create_folders.py):
#
#                    					    data  
#                 ____________________________||__________________
#                 |	      |	                 |	                 |
#               train	test  	          valid              sample		 
#	          ____|___                   _____|___          ______|_______     
#              |       |                 |         |        |              |
#           cats    dogs               cats     dogs    train            valid
#                                               	     ___|___          ____|____    
#                                               	    |       |        |         |
#                                             	    cats   dogs     cats      dogs
#
#
# admin stuff
from __future__ import print_function
import shutil
import os
from random import shuffle

def split_inputs_into_response_classes(sourcepath, response_classes):
    '''
    Define classes in this case as a list of "thumb" and "index". 
    The big idea here is to pass this list of classes into the function, 
    and it searches each file name within the directory to classify. 
    Finally, the function will also create directories with the names 
    of the classes and move the images into these newly created directories accordingly.
    '''
    # Dynamically create child directories named as per the response classes. These will be your destination paths.
    destpath={}
    for cl in response_classes:
        destpath[cl] = os.path.join(sourcepath,cl)
        # to move files into directories, we need to first create the destination directories. Create new only if doesnt 
        # exist, otherwise it throws an 'os' package error
        if not os.path.isdir(destpath[cl]):
            os.mkdir(destpath[cl])

    # look into each file name. then classify them into index or thumb and then move them into appropriate destinations.
    # look into each of them and classify. Obviously move only the ones that belong to one or the other classes. 
    # Print out the number of exceptions so that I can go and manually look at it. If it is a file like .DS_Store, then
    # I can ignore it.
    # IMPORTANT NOTE: BEFORE MOVING THE FILES, ENSURE THAT THE DESTINATION DIRECTORIES ARE PRESENT
    for f in os.listdir(sourcepath):
        for item in destpath:
            if item in f:
                shutil.move(os.path.join(sourcepath,f), destpath[item])
            
    # Check if any files have been left unprocessed. This should not happen unless there are some data issues. 
    # So best to point it out and leave it to your friendly data scientist to figure out.
    unprocessed_files = os.listdir(sourcepath)
    for item in destpath:
        unprocessed_files.remove(item)
    
    return len(unprocessed_files)

def split_into_training_validation_datasets(sourcepath, response_classes, perc_split=0.8):
    '''
    Split files in any directory randomly into training and validation datasets based on 
    the percentage split. Usually, we'll keep it as 80% split.
    
    requires 'shuffle' from the random package: from random import shuffle
    '''
    # Get a list of all the filenames in the directory into a list and randomly shuffle them
    files = os.listdir(sourcepath)
    shuffle(files)
    shuffle(files)
    
    # split the filenames into train and validation based on the percentage split and 
    # store the filenames in a dict
    filesplit={}
    filesplit['train'] = files[:int(len(files)*perc_split)]
    filesplit['valid'] = files[int(len(files)*perc_split):]
    
    # Create child directories for training and validation if it doesnt already exist.
    destpath=['train','valid']

    for folders in destpath:
        # to move files into directories, we need to first create the destination directories. Create new only if doesnt 
        # exist, otherwise it throws an 'os' package error
        if not os.path.isdir(os.path.join(sourcepath,folders)):
            os.mkdir(os.path.join(sourcepath,folders))
    
        # interesting way to move files from directory A to directory B using OS.RENAME method 
        # I got this idea from http://stackoverflow.com/questions/39210765/randomly-distribute-files-into-train-test-given-a-ratio
        for files in filesplit[folders]:
            os.rename(sourcepath+'/'+files , sourcepath+'/'+folders+'/'+files)
    
        # call the function to split into response classes
        num_unprocessed_files = split_inputs_into_response_classes(os.path.join(sourcepath,folders), response_classes)
        print ("%s files were unable to be classified into response classes in the '%s' sub-directory within the '%s' directory" %(num_unprocessed_files,folders,sourcepath))
        
    return None

def copy_one_percent_sample(sourcepath):
    '''
    Image processing is resource intensive. Best practice is to create a sample directory for
    training your models with the same directory structure as that of your actual dataset.
    
    By creating a 1% sample of the entire image dataset (or 100 images which ever is smaller), 
    yu can build and optimise your models in your laptop without having to run a GPU server.
    
    Once you're happy with the code, feel free to change the directory your neural net points to.
    
    That's why its good practice to build a sample dataset. Hope this shit wasnt too patronising! 
    Its new for me. :)
    '''
    # the big idea is to copy from the train folder itself. Thats how kaggle is aligning these folders
    # Get a list of all the filenames in the directory into a list and randomly shuffle them
    # one can see that kaggle puts bulk of the images in the training directory
    files = os.listdir(os.path.join(sourcepath,'train'))
    
    # shuffle them twice so as to get a random split of images in your sample
    shuffle(files)
    shuffle(files)

    # build the list of images that we want to push into our sample dataset
    if int(len(files)*0.01)<100:
        split_sample = files[:int(len(files)*0.01)]
    else:
        split_sample = files[:100]
    
    # create a sample sub-directory if it doesnt already exist. IT SHOULDN'T. CHECK IF IT IS!
    if not os.path.isdir(os.path.join(sourcepath,'sample')):
        os.mkdir(os.path.join(sourcepath,'sample')) 

    # use shutil.copy2 method to copy image as well as image metadata into sample directory
    for files_to_copy in split_sample:
        shutil.copy2(os.path.join(sourcepath,'train',files_to_copy), os.path.join(sourcepath,'sample'))
        
    return None

def split_image_dataset_into_train_test_validation_sample(sourcepath,
                                                           response_classes,
                                                           perc_split = 0.8):
    '''
    creates a directory structure in the format:
                        					    data  
                 ____________________________||__________________
                 |	      |	                |	                    |
               train	test  	          valid              sample		 
	          ____|___                   _____|___          ______|_______     
              |       |                 |         |        |              |
           cats    dogs               cats     dogs    train            valid
                                               	     ___|___          ____|____    
                                               	    |       |        |         |
                                             	    cats   dogs     cats      dogs
    
    sourcepath is the location that points to the original data download and unzip folder into test and train dataset
    '''
    
    # step1: copy 1% of images or 100 images (whichever is smaller) into a sample folder for initial processing
    copy_one_percent_sample(sourcepath)
    
    # step2: split sample into training and validation datasets (default value 80%:20%)
    #        sample step also calls the function to classify into separate folders for each constituent response class
    split_into_training_validation_datasets(sourcepath=os.path.join(sourcepath,"sample"),
                                            response_classes=response_classes, 
                                            perc_split=perc_split)
    
    # step3: split train into training and validation datasets (default value 80%:20%)
    #        this creates train and valid directories within the train foler itself. need to move it one level up
    split_into_training_validation_datasets(sourcepath=os.path.join(sourcepath,"train"),
                                            response_classes=response_classes, 
                                            perc_split=perc_split)
    
    # step4: move train and valid directories within the train folder one level up
    os.rename(os.path.join(sourcepath,"train","valid"), os.path.join(sourcepath,"valid"))
    os.rename(os.path.join(sourcepath,"train"), os.path.join(sourcepath,"unprocessed"))
    os.rename(os.path.join(sourcepath,"unprocessed","train"), os.path.join(sourcepath,"train"))

    return ("Success")