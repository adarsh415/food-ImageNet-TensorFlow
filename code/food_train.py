import tensorflow as tf

import food_Model
import create_batch
import food_input
import numpy as np
from datetime import datetime
import time

BATCH_SIZE=64
max_steps=10000
ckpt_dir='D:\\Food_check'
log_frequency=10
def train():

    with tf.Graph().as_default():
        global_steps=tf.train.get_or_create_global_step()


        images,labels=food_input.distored_inputs("C:\\Users\\padarsh\\Documents\\Food_images\\Train\\_food_c101_n10099_r32x32x3.h5",BATCH_SIZE)

        logits = food_Model.model(images)

        loss=food_Model.losses(logits,labels)

        train_op=food_Model.train(loss,global_steps)

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step=-1
                self._start_time=time.time()
            def before_run(self, run_context):
                self._step+=1
                return tf.train.SessionRunArgs(loss)

            def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
                if self._step%log_frequency==0:
                    current_time=time.time()
                    duration=current_time-self._start_time
                    self._start_time=current_time

                    loss_value=run_values.results
                    examples_per_sec=log_frequency*BATCH_SIZE/duration
                    sec_per_batch=float(duration/log_frequency)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))







        with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                                                      tf.train.NanTensorHook(loss),
                                                      _LoggerHook()]
                                               ) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run([train_op])




def main():
    train()



if __name__=='__main__':
    main()




#init_op = tf.group(tf.global_variables_initializer(),
#                           tf.local_variables_initializer())
#sess=tf.Session()
        #sess.run(init_op)
        #EPOCH = 10
        #coord=tf.train.Coordinator()
        #threads=tf.train.start_queue_runners(coord=coord,sess=sess)
        #while not coord.should_stop():
            #while EPOCH != 0:

                    # print(pos)
                    #return images[pos:pos + batch_Size], labels[pos:pos + batch_Size]
                    #sess.run([logits],feed_dict={images:images[pos:pos + BATCH_SIZE]})
                    #sess.run([loss],feed_dict={logits:logits,labels:labels[pos:pos + BATCH_SIZE]})

                #print (images.shape)
            #sess.run([train_op])
            #EPOCH = EPOCH - 1
            #print('loss after  epoch is {} '.format( sess.run(loss)))



        #coord.request_stop()
        #coord.join(threads)