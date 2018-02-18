import tensorflow as tf
import numpy as np
import h5py
import os





IMAGE_SIZE=32
image_shape={}
def read_data(Pathfilename):
    h_file = h5py.File(Pathfilename, 'r')
    features=h_file['images']
    labels=h_file['category']
    names=h_file['category_names']

    image_shape['W']=features.shape[1]
    image_shape['H']=features.shape[2]
    image_shape['C']=features.shape[3]
    image_shape['S']=features.shape[0]
    image_shape['L']=labels.shape[1]
    #NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = image_shape['S']

    return features,labels,names



def _create_queue(fileName,batch_Size):

    images,labels,_=read_data(fileName)
    images_array=np.asarray(images,dtype='float32')
    labels_array=np.asarray(labels,dtype='float32')

    queue=tf.FIFOQueue(capacity=batch_Size,dtypes=[tf.float32,tf.float32],shapes=[[32,32,3],[101]])
    enqueue_op=queue.enqueue_many([images_array,labels_array])
    qr=tf.train.QueueRunner(queue,[enqueue_op]*1)
    tf.train.add_queue_runner(qr)
    image,label=queue.dequeue()

    #with tf.Session() as sess:
        #coord=tf.train.Coordinator()
        #threads=tf.train.start_queue_runners(coord=coord)


        #image,label=queue.dequeue()
        #for _ in range(1):
            #images,labels=sess.run([image,label])
            #print (images.shape)
            #print (labels.shape)
        #coord.request_stop()
        #coord.join(threads)
    ##
    return image,label



def _create_batch_from_data(image,label,min_queue_size,batch_Size,shuffle):

    num_preprocess_threads=4

    if shuffle:
        images,labels=tf.train.shuffle_batch([image,label],
                                             num_threads=num_preprocess_threads,
                                             batch_size=batch_Size,
                                             capacity=min_queue_size + 3*batch_Size,
                                             min_after_dequeue=min_queue_size
                                             )
    else:
        images,labels=tf.train.batch([image,label],
                                     num_threads=num_preprocess_threads,
                                     batch_size=batch_Size,
                                     capacity=min_queue_size + 3*batch_Size
                                     )

    return images, labels





def distored_inputs(fileName,batch_size):

    image,label=_create_queue(fileName,batch_size)
    distorted_image=tf.image.random_flip_left_right(image)
    distorted_image=tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image=tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)

    float_image=tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([image_shape.get('W'),image_shape.get('H'),image_shape.get('C')])
    label.set_shape(image_shape.get('L'))

    min_fraction_of_examples_in_queue = 0.4
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=image_shape.get('S')
    min_queue_examples=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)

    return _create_batch_from_data(float_image,label,min_queue_examples,batch_size,shuffle=True)




#batch1=distored_inputs("C:\\Users\\padarsh\\Documents\\Food_images\\Train\\_food_c101_n10099_r32x32x3.h5",32)
#with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #coord=tf.train.Coordinator()
    #threads=tf.train.start_queue_runners(coord=coord)
    #img,lab=_create_queue("C:\\Users\\padarsh\\Documents\\Food_images\\Train\\_food_c101_n10099_r32x32x3.h5",32)



    #nextBatch=sess.run([batch1])
    #print(nextBatch)



    #coord.request_stop()
    #coord.join(threads)


