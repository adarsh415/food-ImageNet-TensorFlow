import tensorflow as tf
import create_batch
import food_input

NO_CLASSES=101
NO_EPOCHS_PER_DECAY=50
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
MOVING_AVERAGE_DECAY=0.9999

#size_dict=food_input.imageShape()
NO_DATA_PER_EPOCH=10099


def _activation_summary(x):

    tf.summary.histogram(x.op.name+ '/activation',x)
    tf.summary.scalar(x.op.name+ '/sparsity',tf.nn.zero_fraction(x))





def _create_variable(name,shape,initializer):

    var=tf.get_variable(name,shape,initializer=initializer,dtype=tf.float32)
    return var


def _variable_with_weight_decay(name,shape,stddev,wd):

    var=_create_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32))
    if wd is not None:
        weight_decay=tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var





def model(images):


    with tf.variable_scope('conv1') as scope:
        kernel=_variable_with_weight_decay('weights',[5,5,3,64],stddev=5e-2,wd=None)
        conv=tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases=_create_variable('biases',[64],initializer=tf.constant_initializer(0.0))
        pre_activate=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(pre_activate,name=scope.name)
        _activation_summary(conv1)

    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')


    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _create_variable( 'biases',[64], initializer=tf.constant_initializer(0.1))
        pre_activate = tf.nn.bias_add(conv, biases)
        conv2=tf.nn.relu(pre_activate,name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    pool2=tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    with tf.variable_scope('local2') as scope:
        reshape=tf.reshape(pool2,[64,-1])
        dim=reshape.get_shape()[1].value
        weights=_variable_with_weight_decay('weights',shape=[dim,384],stddev=0.04,wd=0.004)
        biases=_create_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local2=tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(local2)


    with tf.variable_scope('local3') as scope:
        weights=_variable_with_weight_decay('weights',shape=[384,192],stddev=0.04,wd=0.004)
        biases=_create_variable('biases',[192],initializer=tf.constant_initializer(0.01))
        local3=tf.nn.relu(tf.matmul(local2,weights)+biases,name=scope.name)
        _activation_summary(local3)



    with tf.variable_scope('softmax_layer') as scope:
        weights=_variable_with_weight_decay('weights',shape=[192,NO_CLASSES],stddev=1/192,wd=None)
        biases=_create_variable('biases',[NO_CLASSES],initializer=tf.constant_initializer(0.0))
        softmax_linear=tf.add(tf.matmul(local3,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear



def losses(logits,labels):

    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')

    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'),name='total_loss')


def _add_loss_summary(total_loss):

    moving_avg=tf.train.ExponentialMovingAverage(0.9,'avg')
    losses=tf.get_collection('losses')
    moving_avg_op=moving_avg.apply(losses+[total_loss])


    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name+ '(raw)',l)
        tf.summary.scalar(l.op.name,moving_avg.average(l))
    return moving_avg_op

def train(total_loss,global_steps):
    NO_BATCH_PER_EPOCH=NO_DATA_PER_EPOCH/create_batch.BATCH_SIZE
    decay_step=int(NO_BATCH_PER_EPOCH*NO_EPOCHS_PER_DECAY)

    decayed_learning_rate=tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                     global_steps,
                                                     decay_step,
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
    tf.summary.scalar('decayed learning rate',decayed_learning_rate)

    loss_average_op=_add_loss_summary(total_loss)


    with tf.control_dependencies([total_loss]):
        opt=tf.train.GradientDescentOptimizer(decayed_learning_rate)
        grads=opt.compute_gradients(total_loss)

    apply_gradient_op=opt.apply_gradients(grads,global_steps)

    #Add histogram for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)

    #Add histogram for gradients
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(grad.op.name + '/gradient',grad)

    #Track the moving average of trainable varibales
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_steps)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
        train_op=tf.no_op(name='train')


    return train_op


