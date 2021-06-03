
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, AveragePooling2D, Dropout, Input
from keras import regularizers, optimizers, models, layers, losses, metrics
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from skimage import img_as_float
from random import shuffle

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


def load_data(file_path):
    data = cv2.imread(file_path)
    data = img_as_float(data)*255
    return data


# divide each channel of the painting to patches then concatenate channels, also get the list of
# corresponding labels (painter id)
def get_patches(data, patch_size, painter_id):
    pc1 = extract_patches(data[:,:,0], patch_shape = patch_size, extraction_step = patch_size)
    pc1 = pc1.reshape(-1, patch_size, patch_size)
    pc2 = extract_patches(data[:,:,1], patch_shape = patch_size, extraction_step = patch_size)
    pc2 = pc2.reshape(-1, patch_size, patch_size)
    pc3 = extract_patches(data[:,:,2], patch_shape = patch_size, extraction_step = patch_size)
    pc3 = pc3.reshape(-1, patch_size, patch_size)
    pc1_reshaped = pc1.reshape(*pc1.shape,1)
    pc2_reshaped = pc2.reshape(*pc2.shape,1)
    pc3_reshaped = pc3.reshape(*pc3.shape,1)
    #CHANGED FOR HEIGHT BELOW
    patches = np.concatenate((pc1_reshaped,pc2_reshaped,pc3_reshaped),axis=3)
    
    labels = []
    def get_label(painter_id, patch_len):
        labels.clear()
        labels.append(painter_id * patch_len)
        return labels

    list_len = np.ones(len(patches))
    y_list = get_label(painter_id, list_len)
    y_list = np.reshape(y_list,(len(patches),1)) 
                        
    return patches, y_list  # use this when shuffle=False


# preprocess each patches to prepare for transfer learning by subtracting the mean [103.939, 116.779, 123.68]
def preprocess_patches(patch_list):
    #patches = preprocess_input(patch_list)
    patches = patch_list
    return patches

# resize patches to 224*224
def resize_patches(patch_list):   
    resize_patches = [None]*len(patch_list)
    for i in range(len(patch_list)):
        resize_patches[i] = cv2.resize(patch_list[i],(224, 224))
    new_list = np.asarray(resize_patches, dtype=np.float64)
    return new_list
    

def process_pipeline(file_path, patch_size, painter_id):
    data = load_data(file_path)
    patch_list, labels = get_patches(data, patch_size, painter_id)
    #preprocessed_patches = preprocess_patches(patch_list)
    #resized_patches = resize_patches(preprocessed_patches)
    return patch_list, labels


# In[ ]:


psizes = [120]

for patch_size in psizes:
    
    foldnum = 60
    for fold in range(0, foldnum):
        print('PATCH SIZE: '+repr(patch_size))
    
        p1a_x, p1a_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp1.png', patch_size, 0)
        p1b_x, p1b_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp2.png', patch_size, 0)
        p1c_x, p1c_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp3.png', patch_size, 0)
        p2a_x, p2a_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp4.png', patch_size, 1)
        p2b_x, p2b_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp5.png', patch_size, 1)
        p2c_x, p2c_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp6.png', patch_size, 1)
        p3a_x, p3a_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp7.png', patch_size, 2)
        p3b_x, p3b_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp8.png', patch_size, 2)
        p3c_x, p3c_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp9.png', patch_size, 2)
        p4a_x, p4a_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp10.png', patch_size, 3)
        p4c_x, p4c_y = process_pipeline('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp12.png', patch_size, 3)

        p4_b = cv2.imread('/home/gxs372/ml_art_hist/bkgrd_rmd_imgs/fgp11.png',cv2.IMREAD_UNCHANGED)  
        p4_b = img_as_float(p4_b)*255
        p4_b = cv2.rotate(p4_b, cv2.ROTATE_180)
        p4b_x, p4b_y = get_patches(p4_b, patch_size, 3)
        #p4b_x = resize_patches(preprocess_patches(p4b_x))

        del p4_b

        p_x = (p1a_x, p1b_x, p1c_x, p2a_x, p2b_x, p2c_x, p3a_x, p3b_x, p3c_x, p4a_x, p4b_x, p4c_x)
        p_y = (p1a_y, p1b_y, p1c_y, p2a_y, p2b_y, p2c_y, p3a_y, p3b_y, p3c_y, p4a_y, p4b_y, p4c_y)

        del (p1a_x, p2a_x, p3a_x, p4a_x, p1b_x, p2b_x, p3b_x, p4b_x, p1c_x, p2c_x, p3c_x, p4c_x)
        del (p1a_y, p2a_y, p3a_y, p4a_y, p1b_y, p2b_y, p3b_y, p4b_y, p1c_y, p2c_y, p3c_y, p4c_y)


        pick_test = np.array([1,4,7,10])

        ind = 0;
        for img in range(0,len(pick_test)//4):
            for ptch in range(0,len(p_x[pick_test[0]] ) ):
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/test/art1/ptch'+repr(ind)+'.png', p_x[pick_test[img]][ptch])
                ind = ind+1

        ind = 0;
        for img in range(0+1,len(pick_test)//4+1):
            for ptch in range(0,len(p_x[pick_test[0]] ) ):
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/test/art2/ptch'+repr(ind)+'.png', p_x[pick_test[img]][ptch])
                ind = ind+1


        ind = 0;
        for img in range(0 +2,len(pick_test)//4+2):
            for ptch in range(0,len(p_x[pick_test[0]] ) ):
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/test/art3/ptch'+repr(ind)+'.png', p_x[pick_test[img]][ptch])
                ind = ind+1                

        ind = 0;
        for img in range(0+3,len(pick_test)//4 +3):
            for ptch in range(0,len(p_x[pick_test[0]] ) ):
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/test/art4/ptch'+repr(ind)+'.png', p_x[pick_test[img]][ptch])
                ind = ind+1             

        pick_train_val = np.array([0,2,3,5,6,8,9,11])     

        a = np.arange(len(p_x[pick_train_val[0]] ) )          
        perc_train = 0.9

        shuffle(a)  
        ind = 0;
        for img in range(0,len(pick_train_val)//4):
            for ptch in a[0:int(perc_train*len(a))]:
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/train/art1/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                ind = ind+1

        ind = 0 
        for img in range(0,len(pick_train_val)//4):
            for ptch in a[int(perc_train*len(a)):]:
                cv2.imwrite('/home/gxs372/ml_art_hist/data2/val/art1/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                ind = ind+1

        shuffle(a)              
        ind = 0;
        for img in range(2,len(pick_train_val)//4+2):
            for ptch in a[0:int(perc_train*len(a))]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/train/art2/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1
        ind = 0
        for img in range(2,len(pick_train_val)//4+2):
            for ptch in a[int(perc_train*len(a)):]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/val/art2/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1

        shuffle(a)              
        ind = 0;
        for img in range(4,len(pick_train_val)//4+4):
            for ptch in a[0:int(perc_train*len(a))]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/train/art3/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1
        ind = 0 
        for img in range(4,len(pick_train_val)//4+4):
            for ptch in a[int(perc_train*len(a)):]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/val/art3/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1                    


        shuffle(a)              
        ind = 0;
        for img in range(6,len(pick_train_val)//4+6):
            for ptch in a[0:int(perc_train*len(a))]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/train/art4/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1
        ind = 0
        for img in range(6,len(pick_train_val)//4+6):
            for ptch in a[int(perc_train*len(a)):]:
                    cv2.imwrite('/home/gxs372/ml_art_hist/data2/val/art4/ptch'+repr(ind)+'.png', p_x[pick_train_val[img]][ptch])
                    ind = ind+1                    


        del (p_x, p_y)

        
        dir_path = "/home/gxs372/ml_art_hist/height_SI_fig/PS"+repr(patch_size)+"/"

        btch_sz = 32

        datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

        # load and iterate training dataset
        train_it = datagen.flow_from_directory('/home/gxs372/ml_art_hist/data2/train/', class_mode='categorical', batch_size=btch_sz,target_size=(224, 224))
        # load and iterate validation dataset
        val_it = datagen.flow_from_directory('/home/gxs372/ml_art_hist/data2/val/', class_mode='categorical', batch_size=btch_sz, target_size=(224, 224))
        # load and iterate test dataset
        test_it = datagen.flow_from_directory('/home/gxs372/ml_art_hist/data2/test/', class_mode='categorical', batch_size=1, target_size=(224, 224), shuffle = False)

        # define model

        baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

        model = models.Sequential()
        model.add(baseModel)
        model.add(layers.AveragePooling2D(pool_size=(3, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(4, activation="softmax"))

        for layer in baseModel.layers[:]:
            layer.trainable = False

        model.compile(optimizer=optimizers.Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        filepath=dir_path + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Short training ONLY top layers 
        #... so the conv_base weights will not be destroyed by the random intialization of the new weights
        history = model.fit_generator(train_it, steps_per_epoch=train_it.n//btch_sz, epochs = 25, validation_data=val_it, validation_steps=val_it.n//btch_sz ,shuffle=True,callbacks=callbacks_list, verbose=2)

        #load the best top model
        model.load_weights(filepath)
            #model.compile(optimizer=optimizers.Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            # Make last block of the conv_base trainable:
        for layer in baseModel.layers[:11]:
            layer.trainable = False
        for layer in baseModel.layers[11:]:
            layer.trainable = True


            # Compile frozen conv_base + UNfrozen top block + my top layer ... slower learning rate
        model.compile(optimizer=optimizers.Adam(lr = 0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

            #train the unfrozen model
        history = model.fit_generator(train_it, steps_per_epoch=train_it.n//btch_sz, epochs = 25, validation_data=val_it, validation_steps=val_it.n//btch_sz ,shuffle=True,callbacks=callbacks_list, verbose=2)
        
        model.load_weights(filepath)

        model.compile(optimizer=optimizers.Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        _, test_acc = model.evaluate_generator(test_it, steps=test_it.n)
        #test_acc

        pred = model.predict_generator(test_it, steps=test_it.n)
        ypred = np.argmax(pred,axis=1)

        ytest = np.repeat(np.array([0,1,2,3]), test_it.n//4)
        cm = confusion_matrix(ytest, ypred)
        
        test_idx = np.arange(len(pred[ytest==0])).reshape(len(pred[ytest==0]),1)
        p1_predict = np.concatenate([test_idx,pred[ytest==0]], axis=1)
        p2_predict = np.concatenate([test_idx,pred[ytest==1]], axis=1)
        p3_predict = np.concatenate([test_idx,pred[ytest==2]], axis=1)
        p4_predict = np.concatenate([test_idx,pred[ytest==3]], axis=1)        


        print(classification_report(ytest, ypred))
        
        #test_accuracy = (np.trace(cm))/len(ytest)
        size_update = patch_size
        
        report = classification_report(ytest, ypred,output_dict=True)
        
        p1_report = np.asarray([report['0']['f1-score']])
        p1_report = np.insert(p1_report,0,patch_size)
        p1_report = p1_report.reshape(1,2)

        p2_report = np.asarray([report['1']['f1-score']])
        p2_report = np.insert(p2_report,0,patch_size)
        p2_report = p2_report.reshape(1,2)

        p3_report = np.asarray([report['2']['f1-score']])
        p3_report = np.insert(p3_report,0,patch_size)
        p3_report = p3_report.reshape(1,2)

        p4_report = np.asarray([report['3']['f1-score']])
        p4_report = np.insert(p4_report,0,patch_size)
        p4_report = p4_report.reshape(1,2)

        overall = np.asarray([report['accuracy'],report['macro avg']['f1-score'],report['weighted avg']['f1-score'] ])
        overall = np.insert(overall,0,patch_size)
        overall = overall.reshape(1,4)

        cm_flatten = cm.flatten()
        ps_cm = np.insert(cm_flatten,0,patch_size)
        ps_cm = ps_cm.reshape(1,17)

        with open(dir_path +'p1_report_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, p1_report, fmt='%s')
        with open(dir_path +'p2_report_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, p2_report, fmt='%s')
        with open(dir_path +'p3_report_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, p3_report, fmt='%s')
        with open(dir_path +'p4_report_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, p4_report, fmt='%s')
        with open(dir_path +'overall_report_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, overall, fmt='%s')
        with open(dir_path +'cm_height_vgg16_ps'+repr(patch_size)+'.csv','a') as f:
            np.savetxt(f, ps_cm, fmt='%s')
            
        
        print('RESULT: '+repr(size_update)+', '+repr(test_acc)+'\n')
        with open(dir_path +"vgg_main_acc_ps"+repr(patch_size)+".csv", "a") as myfile:
            myfile.write(repr(size_update)+', '+repr(test_acc)+'\n')
            
                    
        with open(dir_path + 'heapmap_p1_ps'+repr(patch_size)+'_9010.csv','a') as f:
            np.savetxt(f, p1_predict, fmt='%s')
        with open(dir_path + 'heapmap_p2_ps'+repr(patch_size)+'_9010.csv','a') as f:
            np.savetxt(f, p2_predict, fmt='%s')
        with open(dir_path + 'heapmap_p3_ps'+repr(patch_size)+'_9010.csv','a') as f:
            np.savetxt(f, p3_predict, fmt='%s')
        with open(dir_path + 'heapmap_p4_ps'+repr(patch_size)+'_9010.csv','a') as f:
            np.savetxt(f, p4_predict, fmt='%s')
        with open(dir_path + 'cm_height_vgg16_ps'+repr(patch_size)+'_9010.csv','a') as f:
            np.savetxt(f, ps_cm, fmt='%s') 
        with open(dir_path +"heatmap_vgg_main_acc_ps"+repr(patch_size)+".csv", "a") as myfile:
            myfile.write(repr(size_update)+', '+repr(test_acc)+'\n')
                            
            
        del model    
   


