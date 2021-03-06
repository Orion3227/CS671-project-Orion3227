# An example on MNIST to introduce how to train and test under CPL

from nets import mnist_net
import functions as func
import numpy as np
import tensorflow as tf
import argparse
import time
import os
import _pickle as pickle
import pdb
#import cPickle as pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None

# compute accuracy on the test dataset
def do_eval(sess, eval_correct, images, labels, test_x, test_y):
    true_count = 0.0
    test_num = test_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1

    for i in range(batch_num):
        batch_x = test_x[i*batch_size:(i+1)*batch_size]
        batch_y = test_y[i*batch_size:(i+1)*batch_size]
        batch_x_reshp = np.reshape(batch_x, (batch_size, 1, 28, 28))
        true_count += sess.run(eval_correct, feed_dict={images:batch_x_reshp, labels:batch_y})
        #true_count += sess.run(eval_correct, feed_dict={images:batch_x, labels:batch_y})
    
    return true_count / test_num

# initialize the prototype with the mean vector (on the train dataset) of the corresponding class
def compute_centers(sess, add_op, count_op, average_op, images_placeholder, labels_placeholder, train_x, train_y):
    train_num = train_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = train_num // batch_size if train_num % batch_size == 0 else train_num // batch_size + 1

    for i in range(batch_num):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        batch_x_reshp = np.reshape(batch_x, (batch_size, 1, 28, 28))
        sess.run([add_op, count_op], feed_dict={images_placeholder:batch_x_reshp, labels_placeholder:batch_y})

    sess.run(average_op)

def run_training(log_file):

    # load the data
    print(150*'*')
    with open("./mnist.pkl", "rb") as fid:
        dataset = pickle.load(fid, encoding='latin1')
    train_x, train_y = dataset[0]
    test_x, test_y = dataset[1]
    train_num = train_x.shape[0]
    test_num = test_x.shape[0]


    # label_index: list of list 
    # ex) label_index[3] conatins every indicies of 3 data
    class_num = 10
    label_index = [[] for i in range(class_num)]
    for i in range(len(train_y)):
        label_index[train_y[i]].append(i)
        

    # construct the computation graph
    images = tf.placeholder(tf.float32, shape=[None,1,28,28])
    labels = tf.placeholder(tf.int32, shape=[None])
    lr= tf.placeholder(tf.float32)

    features, _ = mnist_net(images)
    centers = func.construct_center(features, FLAGS.num_classes)
    loss = func.dce_loss(features, labels, centers, FLAGS.temp)
    eval_correct = func.evaluation(features, labels, centers)
    find_wrong = func.find_wrong(features, labels, centers)
    train_op = func.training(loss, lr)
    
    # counts = tf.get_variable('counts', [FLAGS.num_classes], dtype=tf.int32,
        # initializer=tf.constant_initializer(0), trainable=False)
    # add_op, count_op, average_op = net.init_centers(features, labels, centers, counts)

    init = tf.global_variables_initializer()

    # init the variables
    sess = tf.Session()
    sess.run(init)
    #compute_centers(sess, add_op, count_op, average_op, images, labels, train_x, train_y)

    # run the computation graph (training and test)
    epoch = 1
    loss_before = np.inf
    score_before = 0.0
    stopping = 0
    index = list(range(train_num))
    np.random.shuffle(index)
    batch_size = FLAGS.batch_size
    batch_num = train_num//batch_size if train_num % batch_size==0 else train_num//batch_size+1
    #saver = tf.train.Saver(max_to_keep=1)

    iter = 0
    step = 0
    log_loss = 0
    log_acc = 0
    # pdb.set_trace()
    # train the framework with the training data
    while stopping < FLAGS.stop:
        time1 = time.time()
        loss_now = 0.0
        score_now = 0.0
        iter += 1
        if iter % 100 == 0:
            print("iter : {}".format(iter))

        for i in range(batch_num):
            batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
            batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
            # JG
            batch_x_reshp = np.reshape(batch_x, (batch_size, 1, 28, 28))
            # batch_x.shape : (50, 784)
            # images : <tf.Tensor 'Placeholder:0' shape=(?, 1, 28, 28) dtype=float32>
            ##result = sess.run([train_op, loss, eval_correct], feed_dict={images:batch_x,
            ##    labels:batch_y, lr:FLAGS.learning_rate})
            result = sess.run([train_op, loss, eval_correct, find_wrong], feed_dict={images:batch_x_reshp,
   		labels:batch_y, lr:FLAGS.learning_rate})
            loss_now += result[1]
            score_now += result[2]
            log_loss += result[1]
            log_acc += result[2]
            if (step % FLAGS.log_period == 0) and (step != 0):
                log_file.write('{}, loss, {:.3f}, acc, {:.3f}\n'.format(
                    step, log_loss/FLAGS.log_period, log_acc/batch_size/FLAGS.log_period*100))
                log_loss = 0
                log_acc = 0
            step += 1
        score_now /= train_num

        print ('epoch {}: training: loss --> {:.3f}, acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))
        log_file.write('{}, loss, {:.3f}, acc, {:.3f}\n'.format(epoch, loss_now, score_now*100))
        if loss_now > loss_before or score_now < score_before:
            stopping += 1
            FLAGS.learning_rate *= FLAGS.decay
            print ("\033[1;31;40mdecay learning rate {}th time!\033[0m".format(stopping))

        loss_before = loss_now
        score_before = score_now

        #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=epoch)

        epoch += 1
        np.random.shuffle(index)

        time2 = time.time()
        print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))
    
    # pdb.set_trace()

    # test the framework with the test data
    test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)
    print ('accuracy on the test dataset: {:.3f}%'.format(test_score*100))

    # test the framework with the test data
    test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for training and test')
    #parser.add_argument('--log_dir', type=str, default='data/', help='directory to save the data')
    parser.add_argument('--stop', type=int, default=3, help='stopping number')
    parser.add_argument('--decay', type=float, default=0.3, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=0.5, help='the temperature used for calculating the loss')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id for use')
    parser.add_argument('--num_classes', type=int, default=10, help='the number of the classes')
    parser.add_argument('--log_dir', type = str, default = os.getcwd()+'/log', help = 'Directory where the log file is stored')
    parser.add_argument('--log_period', type=int, default=20, help='log writing period')

    FLAGS, unparsed = parser.parse_known_args()
    print (150*'*')
    print ('Configuration of the training:')
    print ('learning rate:', FLAGS.learning_rate)
    print ('batch size:', FLAGS.batch_size)
    print ('stopping:', FLAGS.stop)
    print ('learning rate decay:', FLAGS.decay)
    print ('value of the temperature:', FLAGS.temp)
    print ('number of classes:', FLAGS.num_classes)
    print ('GPU id:', FLAGS.gpu)
    #print 'path to save the model:', FLAGS.log_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    log_file = open(os.path.join(FLAGS.log_dir, 'log.csv'),'w')
    run_training(log_file)
    log_file.close()

