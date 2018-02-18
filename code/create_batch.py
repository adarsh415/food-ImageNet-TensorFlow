import tensorflow as tf
import numpy as np

import food_input;


BATCH_SIZE=128


def  create_image_label_batch(filepath,batch_Size,epoch):

    image,labels,names=food_input.read_data(filepath)
    image=np.asarray(image,'float32')
    labels=np.asarray(labels,'float32')
    image_size=image.shape[0]
    label_size=labels.shape[0]
    #print (image_size)

    assert image_size==label_size

    while epoch != 0:
        for i in range(0,image_size):
            pos=(i% int(image_size/batch_Size))*batch_Size
            #print(pos)
            #print (image[pos:pos+batch_Size].shape,labels[pos:pos+batch_Size].shape)
        epoch=epoch-1




#create_image_label_batch("C:\\Users\\padarsh\\Documents\\Food_images\\Train\\_food_c101_n10099_r32x32x3.h5",128,10)




