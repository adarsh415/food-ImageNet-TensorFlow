import food_Model
import tensorflow as tf
import numpy as np

import math
from datetime import datetime

import food_input


ckpt_dir='D:\\Food_check'
TOTAL_EAMPLES=1000
BATCH_SIZE=64
MOVING_AVERAGE_DECAY=0.9999


def eval_model(saver,top_k_ops):



    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(ckpt_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print ('no checkpoints found')
            return
        coord=tf.train.Coordinator()
        try:
            threads=[]
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess=sess,coord=coord,daemon=True,start=True))

            num_iterations = int(math.ceil(TOTAL_EAMPLES/BATCH_SIZE))
            true_count=0
            total_sample_count=TOTAL_EAMPLES
            step=0

            while step<num_iterations and not coord.should_stop():
                predictions=sess.run([top_k_ops])
                true_count +=np.sum(predictions)
                step +=1

            precision=true_count/TOTAL_EAMPLES
            print('True Count',true_count)
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        except Exception as e :
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)



def evaluate():
    images, labels = food_input.distored_inputs("C:\\Users\\padarsh\\Documents\\Food_images\\Test\\food_test_c101_n1000_r32x32x3.h5", 64)

    logits = food_Model.model(images)

    predictions=tf.equal(tf.argmax(tf.cast(logits,'float32')),tf.argmax(tf.cast(labels,'float32')))
    #predictions=tf.nn.in_top_k(logits,tf.argmax(labels),1)

    variable_avg=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore=variable_avg.variables_to_restore()
    saver=tf.train.Saver(variable_to_restore)


    while True:
        eval_model(saver,predictions)
        break


if __name__=='__main__':
    evaluate()






