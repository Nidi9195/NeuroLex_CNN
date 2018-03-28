import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm

batch_size    = 100
learning_rate = 0.003
n_epoch       = 50
n_total       = 1000
n_samples     = 600                              # change to 1000 for entire dataset
cv_split      = 0.8                             
#train_size    = int(n_samples * cv_split)
train_size  = 57000
test_size     = n_samples - train_size

op_set = ['Another time', 'nan', "I don't take Parkinson medications", 'Just after Parkinson medication (at your best)', 'Immediately before Parkinson medication']

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

def get_labels(p, json_files, op, batch=1, labels_size = 100, num_classes = 5):
    #num_labels = 100
    js_i = []
    jss = json_files[labels_size*(batch-1):(labels_size*batch)]
    for var in jss:
        path = p + var
        with open(path) as json_file:
            json_text = json.load(json_file)
            l1 = [0 for _ in range(num_classes)]
            index = op.index(json_text['medtimepoint'])
            l1[index] = 1
            js_i.append(l1)

    js_np = np.asarray(js_i)
    return js_np


def get_ms_indexed(p, json_files, batch=1, labels_size=100, num_classes = 5):
    #js_i = np.empty([13,396]) #size
    js_prenp = []
    jss = json_files[labels_size*(batch-1):(labels_size*batch)]
    for var in jss:
        path = p + var
        with open(path) as json_file:
            json_text = json.load(json_file)
            js_i = []
            for v1 in json_text['features']['audio']['MFCC']:
                if len(v1)<396:
                    print("Length of MFCC features",len(v1))
                js_i.append(v1[len(v1)-396:])
            #js1 = np.asarray(js_i) #size: 13x396
        js_prenp.append(js_i)
    js_np = np.asarray(js_prenp)
    js_np1 = js_np.reshape(labels_size, 13, 396, 1)
    #print(js_np1.shape)
    #sys.exit()
        
    return js_np1

def get_ip_op(p, json_files, op, batch=1, labels_size = 1000, num_classes = 5, d_type="train"):
    if d_type == "train":
        indices = random.sample(range(0,57000), labels_size+10)
    else:
        indices = random.sample(range(57000,len(json_files)), labels_size+10)
        
    json_f1 = list(json_files[x] for x in indices)
    
    js_i = []
    js_prenp = []
    count = 0
    #jss = json_files[labels_size*(batch-1):(labels_size*batch)]
    for var in json_f1:
        if (count==labels_size):
            break
        path = p + var
        with open(path) as json_file:
            js_i1 = []
            json_text = json.load(json_file)
            if len(json_text['features']['audio']['MFCC'][0])<396:
                print("Skipping")
                continue
            else:
                for v1 in json_text['features']['audio']['MFCC']:
                    js_i1.append(v1[len(v1)-396:])
                l1 = [0 for _ in range(num_classes)]
                try:
                    mp = json_text['medtimepoint']
                except:
                    mp = 'nan'
                index = op.index(mp)
                l1[index] = 1
                js_i.append(l1)
            
                js_prenp.append(js_i1)
                count += 1

    js_np = np.asarray(js_prenp)
    js_np1 = js_np.reshape(labels_size, 13, 396, 1)
    
    js_np_op = np.asarray(js_i)
    return js_np_op, js_np1

if __name__ == '__main__':
    print("Beginning")
    
    #path_to_json = "C:/Users/Nidhi1/Downloads/PK_trunc/"
    path_to_json = "C:/Users/Nidhi1/Downloads/Parkinsons_mPower_Voice_Features/Parkinsons_mPower_Voice_Features/"

    #jjj = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    #json_files = jjj[:train_size]

    #json_test_files = jjj[train_size:]

    #for var in json_test_files:
    #    path = path_to_json + var
    #    with open(path) as json_file:
    #        json_text = json.load(json_file)
    #        print(len(json_text['features']['audio']['MFCC'][0]))

    #sys.exit()      
    
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print("Total number of samples",len(json_files))
    #path = path_to_json + file
    #xx = get_labels(p = path_to_json, json_files = json_files, op = op_set) #size: 100x5
    #print(xx)

    xx, yy = get_ip_op(p = path_to_json, json_files = json_files, op = op_set)
    
    print("labels length",xx.shape)

    #yy = get_ms_indexed(p = path_to_json, json_files = json_files) #size: 100x13x396x1
    print("Input to cnn shape",yy.shape)

    weights = {
        'wconv1':init_weights([3, 3, 1, 16]),
        'wconv2':init_weights([3, 3, 16, 64]),
        'wconv3':init_weights([3, 3, 64, 64]),
        
        'bconv1':init_biases([16]),
        'bconv2':init_biases([64]),
        'bconv3':init_biases([64]),

        'wd2':init_weights([1*49*64,1000]),
        'b2':init_biases([1000]),

        'wd1':init_weights([1000,5]),
        'b1':init_biases([5]),

        'woutput':init_weights([64, 5]),
        'boutput':init_biases([5])}

    X = tf.placeholder("float", [None, 13, 396, 1])
    y = tf.placeholder("float", [None, 5])

    x = tf.reshape(X,[-1,13,396,1])
    conv2_1_1 = tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_1 = tf.nn.relu(conv2_1_1 + weights['bconv1'])
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.2) #100,6,199,16

    conv2_2_1 = tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_2 = tf.nn.relu(conv2_2_1 + weights['bconv2'])
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.2) #100,3,99,64

    conv2_3_1 = tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.relu(conv2_3_1 + weights['bconv3'])
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.2) #100,1,49,64

    #conv2_4_1 = tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME')
    #conv2_4 = tf.nn.relu(conv2_4_1 + weights['bconv4'])
    #mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    #dropout_4 = tf.nn.dropout(mpool_4, 0.2) #Shape:[4, 1, 14, 128]

    #conv2_5_1 = tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME')
    #conv2_5 = tf.nn.relu(conv2_5_1 + weights['bconv5'])
    #mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    #dropout_5 = tf.nn.dropout(mpool_5, 0.2)

    tf_flattened = tf.reshape(dropout_3, [-1, 1*49*64])
    t1 = tf.nn.relu(tf.add(tf.matmul(tf_flattened,weights['wd2']),weights['b2']))

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(t1,weights['wd1']),weights['b1']))    
    
    #flat = tf.reshape(dropout_3, [-1, weights['woutput'].get_shape().as_list()[0]])
    #y_ = tf.nn.sigmoid(tf.add(tf.matmul(flat,weights['woutput']),weights['boutput']))
        
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    #train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    predict_op = y_

    with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        for i in range(1,n_epoch):
            print("Epoch",i)
            training_batch_op, training_batch_ip = get_ip_op(p = path_to_json, json_files = json_files, op = op_set, batch=i, labels_size=1000)
            #training_batch_ip = get_ms_indexed(p = path_to_json, json_files = json_files, batch=i, labels_size=100)
            #training_batch_op = get_labels(p = path_to_json, json_files = json_files, op = op_set, batch=i, labels_size=100)

            train_input_dict = {X:training_batch_ip,
                                y: training_batch_op }
            #sess.run(train_op, feed_dict=train_input_dict)
            #print("SHAPE")
            #x1 = sess.run(tf.shape(dropout_3), feed_dict={X:training_batch_ip})
            #print(x1)
            #sys.exit()
            
            _, mm = sess.run([train_op, cost], feed_dict=train_input_dict)
            print("Cost:",mm)

            if (i%5 == 0):
                print("TEST BATCH")
                training_batch_op, training_batch_ip = get_ip_op(d_type="test", p = path_to_json, json_files = json_files, op = op_set, batch=1, labels_size=1000)
                train_input_dict = {X:training_batch_ip,
                                y: training_batch_op }
                mm1 = sess.run([cost], feed_dict=train_input_dict)
                print("Test Cost:",mm1)
                predictions = sess.run(predict_op, feed_dict=train_input_dict)
                print('Epoch : ', i,  'AUC : ', sm.roc_auc_score(training_batch_op, predictions, average='samples'))
                



sys.exit()

with open(path) as json_file:
    json_text = json.load(json_file)
    for var in json_text['features']['audio']['MFCC']:
        print(len(var))
