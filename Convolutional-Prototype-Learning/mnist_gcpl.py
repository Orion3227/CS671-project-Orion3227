# An example on MNIST to introduce how to train and test under GCPL

from nets import mnist_net
import functions as func
import numpy as np
import tensorflow as tf
import argparse
import time
import os
import _pickle as pickle
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distribution_distances
#import cPickle as pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None



# compute accuracy on the test dataset
def do_eval(sess, eval_correct, images, labels, test_x, test_y, features):
    true_count = 0.0
    test_num = test_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1 
    eval_points = []

    for i in range(batch_num):
        batch_x = test_x[i*batch_size:(i+1)*batch_size]
        batch_y = test_y[i*batch_size:(i+1)*batch_size]
        batch_x_reshp = np.reshape(batch_x, (batch_size, 1, 28, 28))
        true_count += sess.run(eval_correct, feed_dict={images:batch_x_reshp, labels:batch_y})
        features_eval = sess.run(features, feed_dict={images:batch_x_reshp, labels:batch_y})
        #fe_y = [[fe[0], fe[1], batch_y[i]] for i, fe in enumerate(features_eval)]
        fe_y = []
        for i, fe in enumerate(features_eval):
            fe_y.append([fe[0], fe[1], batch_y[i]])
        eval_points += fe_y
        #eval_points += list(features_eval)
    
    num_categories = 10
    #colors = cm.rainbow(np.linspace(0, 1, num_categories))
    if os.path.isfile('color.pickle'):
        with open('color.pickle', 'rb') as handle:
            colors = pickle.load(handle)
    else:
        with open('color.pickle', 'wb') as handle:
            colors = cm.rainbow(np.linspace(0, 1, num_categories))
            pickle.dump(colors, handle)

    per_same_cats = []
    for i in range(num_categories):
        same_cats_x = []
        same_cats_y = []
        for e in eval_points:
            if e[2] == i:
                same_cats_x.append(e[0])
                same_cats_y.append(e[1])
        same_cats = list(zip(same_cats_x, same_cats_y))
        per_same_cats.append(same_cats)
        plt.scatter(same_cats_x, same_cats_y, color=colors[i]) 
        
    plt.legend(["Class: {}".format(i) for i in range(num_categories)])
    plt.savefig('final_distribution_plot')
    return [true_count / test_num, per_same_cats]

def compute_overlap(eval_dots_perclass):
    num_classes = len(eval_dots_perclass)
    if num_classes < 2:
        return np.array([0])

    distributions = [distribution_distances.GaussianDistribution(data=eval_dots_perclass[i])
                     for i in range(num_classes)]
    
    overlap_matrix = np.zeros((num_classes, num_classes))
    for idx_1, dist_1 in enumerate(distributions):
        for idx_2, dist_2 in enumerate(distributions):
            if dist_1 is dist_2:
                continue

            overlap_matrix[idx_1, idx_2] = distribution_distances.JensenShannonDivergenceMultiVarianteGaussians(
                dist_1, dist_2)

    print(overlap_matrix)
    return overlap_matrix

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

def run_training():

    # load the data
    print (150*'*')
    #with open("mnist.data", "rb") as fid:
    with open("/home/ubuntu/datasets/mnist.pkl", "rb") as fid:
        dataset = pickle.load(fid, encoding='latin1')
    train_x, train_y = dataset[0]
    test_x, test_y = dataset[1]
    train_num = train_x.shape[0]
    test_num = test_x.shape[0]

    # construct the computation graph
    images = tf.placeholder(tf.float32, shape=[None,1,28,28])
    labels = tf.placeholder(tf.int32, shape=[None])
    lr= tf.placeholder(tf.float32)

    features, _ = mnist_net(images)
    centers = func.construct_center(features, FLAGS.num_classes)
    loss1 = func.dce_loss(features, labels, centers, FLAGS.temp)
    loss2 = func.pl_loss(features, labels, centers)
    loss = loss1 + FLAGS.weight_pl * loss2
    eval_correct = func.evaluation(features, labels, centers)
    train_op = func.training(loss, lr)
    
    #counts = tf.get_variable('counts', [FLAGS.num_classes], dtype=tf.int32,
    #    initializer=tf.constant_initializer(0), trainable=False)
    #add_op, count_op, average_op = net.init_centers(features, labels, centers, counts)
  
    sess = tf.Session()
    load_saver = tf.train.Saver()
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    file_list = os.listdir(FLAGS.log_dir)
    keep_last_int = 0
    last_load_file_name = ''
    for name in file_list:
        if len(name.split('.')) < 2:
            continue
        if keep_last_int < int(name.split('.')[1].split('-')[1]):
            keep_last_int = int(name.split('.')[1].split('-')[1])
            last_load_file_name = '.'.join(name.split('.')[:2])
    load_file = os.path.join(FLAGS.log_dir, last_load_file_name)
    if os.path.isfile(load_file+".meta"):
        load_saver.restore(sess, load_file)
    else:
        init = tf.global_variables_initializer()

        # initialize the variables
        sess = tf.Session()
        sess.run(init)
        #compute_centers(sess, add_op, count_op, average_op, images, labels, train_x, train_y)

        # run the computation graph (train and test process)
        epoch = 1
        loss_before = np.inf
        score_before = 0.0
        stopping = 0
        index = list(range(train_num))
        np.random.shuffle(index)
        batch_size = FLAGS.batch_size
        batch_num = train_num//batch_size if train_num % batch_size==0 else train_num//batch_size+1
        saver = tf.train.Saver(max_to_keep=1)

        # train the framework with the training data
        while stopping<FLAGS.stop:
            time1 = time.time()
            loss_now = 0.0
            score_now = 0.0
        
            for i in range(batch_num):
                batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
                batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
                batch_x_reshp = np.reshape(batch_x, (batch_size, 1, 28, 28))
                result = sess.run([train_op, loss, eval_correct], feed_dict={images:batch_x_reshp,
                    labels:batch_y, lr:FLAGS.learning_rate})
                # features_eval = sess.run([features], feed_dict={images:batch_x_reshp, 
                #    labels:batch_y, lr:FLAGS.learning_rate})
                # features_eval.shape (1, 50, 2)
                loss_now += result[1]
                score_now += result[2]
            score_now /= train_num

            print ('epoch {}: training: loss --> {:.3f}, acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))
            print (sess.run(centers))
        
            if loss_now > loss_before or score_now < score_before:
                stopping += 1
                FLAGS.learning_rate *= FLAGS.decay
                print ("\033[1;31;40mdecay learning rate {}th time!\033[0m".format(stopping))
                
            loss_before = loss_now
            score_before = score_now

            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)

            epoch += 1
            np.random.shuffle(index)

            time2 = time.time()
            print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))

            break # For testing only the first episode
        
    #pdb.set_trace() 
    # test the framework with the test data
    test_score, eval_dots_perclass = do_eval(sess, eval_correct, images, labels, test_x, test_y, features) 
    compute_overlap(eval_dots_perclass)
    # eval_dots_perclass
    # len(eval_dots_perclass) : 10 [num_categories=10]
    # eval_dots_perclass[0] : dots of category 0
    # eval_dots_perclass[0][0] : (x, y) of index 0 of category 0
    print ('accuracy on the test dataset: {:.3f}%'.format(test_score*100))
    # pdb.set_trace()

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
    parser.add_argument('--log_dir', type=str, default='data/', help='directory to save the data')
    parser.add_argument('--stop', type=int, default=3, help='stopping number')
    parser.add_argument('--decay', type=float, default=0.3, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=1.0, help='the temperature used for calculating the loss')
    parser.add_argument('--weight_pl', type=float, default=0.001, help='the weight for the prototype loss (PL)')
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id for use')
    parser.add_argument('--num_classes', type=int, default=10, help='the number of the classes')

    FLAGS, unparsed = parser.parse_known_args()
    print (150*'*')
    print ('Configuration of the training:')
    print ('learning rate:', FLAGS.learning_rate)
    print ('batch size:', FLAGS.batch_size)
    print ('stopping:', FLAGS.stop)
    print ('learning rate decay:', FLAGS.decay)
    print ('value of the temperature:', FLAGS.temp)
    print ('prototype loss weight:', FLAGS.weight_pl)
    print ('number of classes:', FLAGS.num_classes)
    print ('GPU id:', FLAGS.gpu)
    #print 'path to save the model:', FLAGS.log_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    run_training()

