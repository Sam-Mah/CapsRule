from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
# from Rule_Extraction_And import extract_rules_boundary
from Rule_Extraction_and_or import extract_rules_boundary, mean, std
from Rule_Evaluation import evaluate_rules_boundary
from Rule_Validation import validate_rules
from Squash_s import squash, safe_norm
from sklearn.preprocessing import Normalizer, minmax_scale
import tensorflow as tf
import matplotlib.pyplot as plt
from Performance_measures import confusion_metrics_tf, Micro_calculate_measures, Macro_calculate_measures_basic, Macro_calculate_measures_tf

'''
Global Scope
'''
batch_indx = 0
init_sigma = 1.0
'''
CONSTANTS
'''
# # @@@@@@@@@@@@@@@@@@FFCN Constants@@@@@@@@@@@@@@@@@@@@@@@@
# &&&&&&&&&&&&&&&&vertebral&&&&&&&&&&&&&&&&&&&
# N_CLASSSES = 2
# input = 6
# caps1_n_caps = 4
# caps2_n_caps = N_CLASSSES
# batch_size = 5
# caps1_dim = 8
# caps2_dim = 6
# Routing_Iter = 2
# LRATE = 0.001
# n_epochs = 50
# &&&&&&&&&&&&IRIS&&&&&&&&&&&&&&
# N_CLASSSES = 2
# input = 4
# caps1_n_caps = 3
# caps2_n_caps = N_CLASSSES
# batch_size = 5
# caps1_dim = 8
# caps2_dim = 6
# Routing_Iter = 2
# LRATE = 0.001
# n_epochs = 50
# @@@@@@@@@@@@@@@@@@Glass@@@@@@@@@@@@@@@@@@@@@@@@
# N_CLASSSES = 7
# input = 9
# caps1_n_caps = 8
# caps2_n_caps = N_CLASSSES
# batch_size = 10
# caps1_dim = 20
# caps2_dim = 30
# Routing_Iter = 2
# LRATE = 0.01
# n_epochs = 50
# &&&&&&&&&&&&&&&&ecoli2&&&&&&&&&&&&&&&&&&&
# N_CLASSSES = 5
# input = 7
# caps1_n_caps = 8
# caps2_n_caps = N_CLASSSES
# batch_size = 5
# caps1_dim = 8
# caps2_dim = 6
# Routing_Iter = 2
# LRATE = 0.001
# n_epochs = 50
# &&&&&&&&&&&&&&&&CICDOS2019 Full Features&&&&&&&&&&&&&&&&&&&
# N_CLASSSES = 2
# input = 82
# caps1_n_caps = 40
# caps2_n_caps = N_CLASSSES
# batch_size = 20
# caps1_dim = 30
# caps2_dim = 20
# Routing_Iter = 2
# LRATE = 0.05
# n_epochs = 10
# &&&&&&&&&&&&&&&&CICDOS2019-12 Features&&&&&&&&&&&&&&&&&&&
batch_size = 50
N_CLASSSES = 4
num_layers = 5
size_layers = [25, 20, 15, 8, N_CLASSSES] #output_dimesion for each layer
caps_dim = [30, 25, 20, 15, 10]
# num_layers = 4
# size_layers = [20, 15, 8, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [25, 20, 15, 10]
# num_layers = 4
# size_layers = [24, 18, 10, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [28, 24, 18, 10]
# num_layers = 5
# size_layers = [24, 20, 14, 5, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [29, 25, 19, 10, 8]
# num_layers = 6
# size_layers = [26, 21, 16, 11, 6, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [32, 27, 23, 18, 11, 7]
# num_layers = 7
# size_layers = [26, 22, 18, 14, 10, 6, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [35, 30, 25, 20, 15, 10, 8]
# num_layers = 3
# size_layers = [20, 10, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [25, 15, 10]
# size_layers = [15, 12, N_CLASSSES] #output_dimesion for each layer
# caps_dim = [20, 18, 16]
caps_dim_input = 1
input = 30
# input = 20
# caps1_n_caps = 6
# caps2_n_caps = N_CLASSSES
# batch_size = 5
# caps1_dim = 10
# caps2_dim = 8
Routing_Iter = 2
# LRATE = 1e-5
# start_lr = 0.1
LRATE = 0.1
n_epochs = 150
# Small epsilon value for the BN transform
# epsilon = 1e-3
loss_thresh = 0.01
restore_checkpoint = True
# $$$$$$$$$$$$$$$$$Decoder Constants$$$$$$$$$$$$$$$$$$$$$$$
# n_hidden1 = caps_dim[-1]
# n_hidden2 = math.floor((caps_dim[-1]+input)/2)
# n_hidden1 = 100
n_hidden1 = 5
n_hidden2 = 15
n_hidden3 = 20
n_output = input
# n_hidden1 = 5
# n_hidden2 = 10
# n_hidden3 = 15
# n_output = input
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def condition(d, caps_predicted, routing_weights, raw_weights, caps1_n,caps2_n, counter):
    return tf.less(counter, Routing_Iter)

def loop_body(caps_output, caps_predicted, routing_weights, raw_weights, caps1_n, caps2_n, counter):
    '''to do elementwise matrix multiplication'''
    '''uj|i * caps2_predicted shape=(caps1_n_caps, caps2_n_caps, caps_dim, caps_dim)'''
    weighted_predictions = tf.multiply(routing_weights, caps_predicted, name="weighted_predictions")
    '''s'''
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")

    '''Batch Normalization'''
    # Calculate batch mean and variance
    # batch_mean, batch_var = tf.nn.moments(weighted_sum,[-3])
    # batch_mean_expanded = tf.expand_dims(batch_mean,-3)
    # batch_var_expanded = tf.expand_dims(batch_var, -3)
    # batch_mean_tiled = tf.tile(batch_mean_expanded,[1,1,caps2_n,1,1])
    # batch_var_tiled = tf.tile(batch_var_expanded, [1, 1, caps2_n, 1, 1])
    # # Apply the initial batch normalizing transform
    # z_hat = tf.divide(tf.subtract(weighted_sum, batch_mean_tiled), tf.sqrt(tf.add(batch_var_tiled, epsilon)))
    #
    # # Scale and shift to obtain the final output of the batch normalization
    # # this value is fed into the activation function (here a sigmoid)
    # weighted_sum = tf.add(tf.matmul(z_hat,scale),beta)
    # # BN = tf.layers.batch_normalization(weighted_sum,momentum=0.9)
    # '''Squash the output 𝐯𝑗=squash(𝐬𝑗) '''
    caps_output = squash(weighted_sum, axis=-2, name="caps_output")
    # caps_output = tf.nn.relu(BN, name="caps_output")
    '''let's measure how close each predicted vector  𝐮̂ 𝑗|𝑖  is to the actual output vector
      𝐯𝑗  by computing their scalar product  𝐮̂ 𝑗|𝑖⋅𝐯𝑗'''
    '''We need to make the dimensions match'''
    caps_output_tiled = tf.tile(caps_output, [1, caps1_n, 1, 1, 1], name="caps_output_tiled")
    agreement = tf.matmul(caps_predicted, caps_output_tiled, transpose_a=True, name="agreement")

    '''We can now update the raw routing weights  𝑏𝑖,𝑗  by simply adding the scalar product  
    𝐮̂ 𝑗|𝑖⋅𝐯𝑗  we just computed:  𝑏𝑖,𝑗←𝑏𝑖,𝑗+𝐮̂ 𝑗|𝑖⋅𝐯𝑗'''
    raw_weights = tf.add(raw_weights, agreement, name="raw_weights")

    '''repeated exactly as round 1'''
    '''Ci: coupling coefficients'''
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

    return caps_output, caps_predicted, routing_weights, raw_weights, caps1_n, caps2_n, tf.add(counter, 1)
def caps_layer(input_size, output_size, caps_dim_input,caps_dim_output):

    '''
    Computation Graph
    '''

    hidden_layer = {
                      'weights': tf.Variable(tf.random_normal(shape=(1, input_size, output_size, caps_dim_output, caps_dim_input)
                                                              , stddev=init_sigma, mean=0.0, dtype=tf.float32, name='weights')
                                             , name='weights'),
                      # 'weights': tf.get_variable("weights", shape=[1, input_size, output_size, caps_dim_output, caps_dim_input],
                      #                            initializer=xavier_init(input_size,output_size)),
                      # 'biases': tf.Variable(tf.random_normal([caps1_n_caps])),
                      'raw_weights': tf.Variable(tf.zeros([batch_size, input_size, output_size, 1, 1],
                                                          dtype=np.float32, name="raw_weights")),
                      # Create two new parameters, scale (gamma) and beta (shift) for batch normalization
                      # 'scale': tf.Variable(tf.ones([batch_size,1,output_size,1,1])),
                      # 'beta': tf.Variable(tf.zeros([batch_size,1,output_size,1,1]))
                      }

    return hidden_layer

def neural_network_model(data):

    input_size=input
    caps_dim_input=1

    '''Expand the input capsule (u) three times to have the shape of u=(input, caps_layer_1, input_dim=1, 1)'''
    input_expanded_1 = tf.expand_dims(data, -1, name="input_expanded_1")
    input_expanded_2 = tf.expand_dims(input_expanded_1, 2, name="input_expanded_2")
    input_expanded = tf.expand_dims(input_expanded_2, 3, name="input_expanded")

    counter = tf.constant(1)
    h = list()
    a_lst = list()
    b_lst = list()
    res_lst = list()

    for i in range(num_layers):
        h.append(caps_layer(input_size, size_layers[i], caps_dim_input,caps_dim[i]))
        with tf.name_scope("routing_by_agreement"):
            W_tiled = tf.tile(h[i]['weights'], [batch_size, 1, 1, 1, 1], name="W_tiled")
            input_expanded_tiled = tf.tile(input_expanded, [1, 1, size_layers[i], 1, 1], name="input_expanded_tiled")
            caps_predicted = tf.matmul(W_tiled, input_expanded_tiled, name="caps_predicted")
            ''' Add routing weights ci = softmax(bi)'''
            routing_weights = tf.nn.softmax(h[i]['raw_weights'], dim=2, name="routing_weights")
            result = tf.Variable(tf.zeros([batch_size, 1, size_layers[i], caps_dim[i], 1], dtype=np.float32, name="result"))
            result, a, b, c, d, e, f = tf.while_loop(condition, loop_body, [result, caps_predicted, routing_weights,
                                                                          h[i]['raw_weights'], input_size,size_layers[i],counter])
            pred_vector = tf.squeeze(a, axis=[4])
            a_lst.append(pred_vector)  # Caps-predicted list
            cpl_coeff = tf.squeeze(b, axis=[3, 4], name="cpl_coeff")
            b_lst.append(cpl_coeff)  # Routing weights list
            caps_out = tf.squeeze(result, axis=[1, -1])
            res_lst.append(caps_out)  # Caps-out list
            result_sq = tf.squeeze(result, axis=[1], name="result_sq")
            input_expanded = tf.expand_dims(result_sq, 2, name="input_expanded")
        input_size = size_layers[i]
        caps_dim_input = caps_dim[i]

    y_proba = safe_norm(result, axis=-2, name="y_proba")
    y_pred = tf.squeeze(y_proba, axis=[1, 3], name="y_pred")

    return data, y_pred, b_lst,a_lst, res_lst, result, W_tiled

def reconstruction(y, y_pred, caps_out, mask_with_labels):
    '''We need a placeholder to tell TensorFlow whether we want to mask the output vectors based on the labels (True)
    or on the predictions (False, the default):'''

    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred,  # if False
                                     name="reconstruction_targets")
    ''' Reconstruction_target is equal to 1.0 for the target class, and 0.0 for the other classes,'''

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_targets, [-1, 1, N_CLASSSES, 1, 1],
        name="reconstruction_mask_reshaped")

    caps_output_masked = tf.multiply(
        caps_out, reconstruction_mask_reshaped,
        name="caps_output_masked")

    decoder_input = tf.reshape(caps_output_masked,
                               [-1, size_layers[-1] * caps_dim[-1]],
                               name="decoder_input")

    return decoder_input

def decoder(decoder_input):
    with tf.name_scope("decoder"):
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3,
                                  activation=tf.nn.relu,
                                  name="hidden3")
        # hidden4 = tf.layers.dense(hidden3, n_hidden4,
        #                           activation=tf.nn.relu,
        #                           name="hidden4")
        decoder_output = tf.layers.dense(hidden3, n_output,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")

    return decoder_output

def next_batch(data, batch_indx):
    list_data = np.array(data.iloc[:, 0:col_num - 1])
    list_labels = np.zeros((len(list_data), N_CLASSSES), dtype=int)
    list_labels = np.array(data.iloc[:, col_num - 1], dtype=int)
    # 1-d array to one-hot conversion
    onehot_labels = np.zeros((len(list_labels), N_CLASSSES))
    onehot_labels[np.arange(list_labels.size), list_labels] = 1
    list_labels = onehot_labels

    try:
        lst_data = list_data[batch_indx:batch_indx + batch_size]
        lst_lbl = list_labels[batch_indx:batch_indx + batch_size]
    # The number of samples in the last batch is less than batch_size
    except Exception as E:
        print(E)

    return lst_data, lst_lbl

def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(n_hidden1) # 0, 1, ..., 15
    tweaks = np.zeros([n_hidden1, n_steps, 1, 1, 1, n_hidden1, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded

'''                 
Reading the data
'''
def normalize(data):

    # normalized_data = Normalizer(norm='l2').fit_transform(data)
    normalized_data = minmax_scale(data)

    return normalized_data

# data = np.genfromtxt("Datasets/20_Features_CICDOS_Testing_balanced_80k.csv", delimiter=",")
# data = np.genfromtxt("Datasets/Exploitation_ch3_15k.csv", delimiter=",")
# data = np.genfromtxt("Datasets/Exploitation_ch3_15k.csv", delimiter=",")

with open("Datasets/Reflection_ch1_20k.csv") as csv_file:
# with open("Datasets/Exploitation_ch3_15k.csv") as csv_file:
    csv_reader = pd.read_csv(csv_file, delimiter=',')
np_cols = csv_reader.columns.to_numpy()
features_test = np.array([x.strip() for x in np_cols])
np.savetxt("Experiments/Features.csv", features_test, delimiter=",", fmt='%s')
data = csv_reader.to_numpy()
# data = np.genfromtxt("Datasets/30_Features_CICDOS_Testing_balanced_LDAPMSSQLNetBios_15k.csv", delimiter=",")

# print("mean input data:",np.mean(data),"std input data:",np.std(data))
# data = np.genfromtxt("Datasets/glass.csv", delimiter=",")
# data = np.genfromtxt("Datasets/iris.csv", delimiter=",")

col_num = data.shape[1]
row_num = data.shape[0]

labels = np.array(data[:, col_num - 1])
labels = labels.astype(int)
print(np.unique(labels))
# dropping the labels' columns
data = data[:, :col_num - 1]
# data = normalize(data)
''' 1-d array to one-hot conversion
# Labels 1 >>> 1 0 >>> Positive (Malicious)
# Labels 2 >>> 0 1 >>> Negative (Benign)
# In the rules, output 0 means (1 0)
# In the rules, output 1 means (0 1)'''

onehot_labels = np.zeros((row_num, N_CLASSSES))
onehot_labels[np.arange(labels.size), labels - 1] = 1

'''
Computation Graph
'''
X = tf.placeholder(shape=[None, input], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None, None], dtype=tf.float32, name="y")

'''
ypred=prediction=The output of the capsnet for class prediction
b_lst=cpl_lst=Routing weights list
a_lst=pred_lst=Caps-predicted list
out_vect=caps_out list
'''

dt, prediction, cpl_lst, pred_lst, out_vect, caps_out, W_tiled = neural_network_model(X)

'''Reconstruction Loss'''
mask_with_labels = tf.placeholder_with_default(True, shape=(), name="mask_with_labels")
decoder_inp = reconstruction(y, prediction, caps_out, mask_with_labels)
decoder_out = decoder(decoder_inp)

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_out,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

# cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.cast(prediction, tf.float32)))
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
caps_out_norm = safe_norm(caps_out, axis=-2, keep_dims=True,
                              name="caps_out_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - caps_out_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, size_layers[-1]),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps_out_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1,  size_layers[-1]),
                          name="absent_error")
L = tf.add(y * present_error, lambda_ * (1.0 - y) * absent_error,
           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
'''Final Loss'''
eta = 5.0e-15
# eta = 5.0e-5
# cost = tf.add(margin_loss, eta * reconstruction_loss, name="loss")
cost = tf.add(margin_loss, eta * reconstruction_loss, name="loss")

correct = tf.equal(tf.argmax(y, -1), tf.argmax(prediction, -1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
# --------------------------------------------------------------------------
# Optimize
# --------------------------------------------------------------------------
# decayed_learning_rate = learning_rate *
#                         decay_rate ^ (global_step / decay_steps)
# global_step = tf.Variable(0, trainable=False, dtype= tf.int32)
# learning_rate = tf.compat.v1.train.exponential_decay(
#     start_lr, global_step=global_step,
#     decay_steps=100, decay_rate=0.96)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=LRATE).minimize(cost)
# --------------------------------------------------------------------------

init = tf.global_variables_initializer()
saver = tf.train.Saver()

'''
Pre-processing
'''

train_data, test_data, train_labels, test_labels = train_test_split(
    data, onehot_labels, test_size=0.3, stratify=onehot_labels)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.3, stratify=train_labels)

# convert the one-hot labels back to float
tclass_labels = np.argmax(train_labels, axis=1)
vclass_labels = np.argmax(val_labels, axis=1)
teclass_labels = np.argmax(test_labels, axis=1)

tmp_train = np.hstack((train_data, tclass_labels.reshape((len(train_labels), 1))))
df_train = pd.DataFrame(tmp_train)
# df_train.sort_values(df_train.columns.values[-1], inplace=True)

tmp_val = np.hstack((val_data, vclass_labels.reshape((len(val_labels), 1))))
df_val = pd.DataFrame(tmp_val)
# df_val.sort_values(df_val.columns.values[-1], inplace=True)

tmp_test = np.hstack((test_data, teclass_labels.reshape((len(test_labels), 1))))
df_test = pd.DataFrame(tmp_test)
# df_test.sort_values(df_test.columns.values[-1], inplace=True)

n_iterations_per_epoch = len(train_data) // batch_size
n_iterations_validation = len(val_data) // batch_size
best_loss_val = np.infty
checkpoint_path = "./Capsnet_FFNN"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    # summary_writer = tf.summary.FileWriter("./Capsnet_FFNN", sess.graph)
    epoch = 1
    loss_epochs = [2, 1]
    #Train until the number of maximum epochs or validation loss of two consecutive loops is less that loss threshold
    while (epoch < n_epochs or loss_epochs[epoch]-loss_epochs[epoch-1]> loss_thresh):
        '''Training'''
        rule_set = list()
        # iteration_list = np.zeros(1)
        # neural_network_cost_list = np.zeros(1)
        # cond_order_set = list()
        btch_indx = 0
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = next_batch(df_train, btch_indx)
            # print(type(X_batch))
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [optimizer, cost],
                feed_dict={X: X_batch,
                           # To have the batch samples as the elements of the activation vectors of each capsule
                           y: y_batch})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="")

            pred_vect_lst = [pred.eval({X: X_batch, y: y_batch}) for pred in pred_lst]
            cpl_coeff_lst = [cpl.eval({X: X_batch, y: y_batch}) for cpl in cpl_lst]
            out_vect_lst = [out.eval({X: X_batch, y: y_batch}) for out in out_vect]
            pred = prediction.eval({X: X_batch, y: y_batch})

            if(epoch == (n_epochs-1)):
                rule_set.append(extract_rules_boundary(dt.eval({X: X_batch, y: y_batch}), cpl_coeff_lst, pred_vect_lst,
                                                       out_vect_lst, pred))

            btch_indx += batch_size

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        '''Validation'''
        loss_vals = []
        acc_vals = []
        if (epoch == (n_epochs - 1)):
            conf_vals = np.zeros((N_CLASSSES,N_CLASSSES))
            tp_v = []
            tn_v = []
            fp_v = []
            fn_v = []
            pr_v = []
            rc_v = []
            f1_v = []
            y_true_val = list()
            y_pre_val = list()
        btch_indx = 0
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = next_batch(df_val, btch_indx)

            # Run the training operation and measure the loss:
            loss_val, acc_val = sess.run(
                [cost, accuracy],
                feed_dict={X: X_batch,
                           y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation),
                end=" " * 10)
            if (epoch == (n_epochs - 1)):
                conf, tp, tn, fp, fn, act, prob = confusion_metrics_tf(y, prediction, sess, {X: X_batch, y: y_batch})
                conf_vals = np.add(conf_vals,conf)
                tp_v.append(tp)
                tn_v.append(tn)
                fp_v.append(fp)
                fn_v.append(fn)
                y_true_val.append(act)
                y_pre_val.append(prob)
                pr, rc, f1 = Macro_calculate_measures_tf(y, prediction, sess, {X: X_batch, y: y_batch})
                pr_v.append(pr)
                rc_v.append(rc)
                f1_v.append(f1)
            btch_indx += batch_size

        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)

        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val

        if (epoch == (n_epochs - 1)):
            #######################Confusion Matrix & Loss##########################
            # confusion matrix: column: prediction labels and rows real labels
            conf_val = conf_vals / n_iterations_validation
            tp_val = np.mean(tp_v)
            tn_val = np.mean(tn_v)
            fp_val = np.mean(fp_v)
            fn_val = np.mean(fn_v)
            pr_val = np.mean(pr_v)
            rc_val = np.mean(rc_v)
            f1_val = np.mean(f1_v)
            sum_conf_val = np.sum(conf_val, axis=1)
            lst_val = []
            for i in range(len(sum_conf_val)):
                lst_val.append(np.round((conf_val[i, :] / sum_conf_val[i]), 2))
            arr_val = np.array(lst_val)
            val_measures = Micro_calculate_measures(tp_val, tn_val, fp_val, fn_val, 0)
            output = np.vstack((conf_val, arr_val))
            np.savetxt("Experiments/Micro_Validation_Conf_FFCN.csv", output, delimiter=',', fmt='%s')
            np.savetxt("Experiments/Micro_Validation_Measures_FFCN.csv",val_measures.to_numpy(), delimiter=',', fmt='%s')
            f = open("Experiments/Macro_Validation_Results_FFCN.txt", 'w')
            f_str = "PR:"+str(pr_val)+" RC:"+str(rc_val)+" F1:"+str(f1_val)
            f.write(f_str)
            f.close()
            np.savetxt("Experiments/y_true_FFCN_val.csv", np.array(y_true_val).flatten(), delimiter=',')
            np.savetxt("Experiments/y_pre_FFCN_val.csv", np.array(y_pre_val).flatten(), delimiter=',')
            #########################################################################
            rules = validate_rules(rule_set, val_data, val_labels)

        loss_epochs.append(loss_val)
        epoch +=  1


global mean, std
np.savetxt("Parameters/mean_couple.csv", mean, delimiter=",", fmt='%s')
np.savetxt("Parameters/std_couple.csv", std, delimiter=",", fmt='%s')
'''
Evaluation
'''
n_iterations_test = len(test_data) // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    conf_tests = np.zeros((N_CLASSSES, N_CLASSSES))
    tp_t = []
    tn_t = []
    fp_t = []
    fn_t = []
    pr_t = []
    rc_t = []
    f1_t = []
    y_true_t = list()
    y_pre_t = list()
    btch_indx = 0
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = next_batch(df_test, btch_indx)
        loss_test, acc_test = sess.run(
            [cost, accuracy],
            feed_dict={X: X_batch,
                       y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_test,
           iteration * 100 / n_iterations_test),
            end=" " * 10)
        conf, tp, tn, fp, fn, act, prob = confusion_metrics_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        conf_tests = np.add(conf_tests, conf)
        tp_t.append(tp)
        tn_t.append(tn)
        fp_t.append(fp)
        fn_t.append(fn)
        y_true_t.append(act)
        y_pre_t.append(prob)
        pr, rc, f1 = Macro_calculate_measures_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        pr_t.append(pr)
        rc_t.append(rc)
        f1_t.append(f1)
        btch_indx += batch_size
        # np.savetxt("input.csv", dt.eval({X: np.transpose(X_batch), y: y_batch}), delimiter=",")
        # np.savetxt("output.csv", prediction.eval({X: np.transpose(X_batch), y: y_batch}), delimiter=",")
        # np.savetxt("real_output.csv", y.eval({X: np.transpose(X_batch), y: y_batch}), delimiter=",")

    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)

    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))
    #######################Confusion Matrix & Loss##########################
    # confusion matrix: column: prediction labels and rows real labels
    conf_test = conf_tests / n_iterations_test
    tp_test = np.mean(tp_t)
    tn_test = np.mean(tn_t)
    fp_test = np.mean(fp_t)
    fn_test = np.mean(fn_t)
    pr_test = np.mean(pr_t)
    rc_test = np.mean(rc_t)
    f1_test = np.mean(f1_t)
    sum_conf_test = np.sum(conf_test, axis=1)
    lst_test = []
    for i in range(len(sum_conf_test)):
        lst_test.append(np.round((conf_test[i, :] / sum_conf_test[i]), 2))
    arr_test = np.array(lst_test)
    test_measures = Micro_calculate_measures(tp_test, tn_test, fp_test, fn_test, 0)
    output = np.vstack((conf_test, arr_test))
    np.savetxt("Experiments/Micro_Test_Conf_FFCN.csv", output, delimiter=',', fmt='%s')
    np.savetxt("Experiments/Micro_Test_Measures_FFCN.csv", test_measures.to_numpy(), delimiter=',', fmt='%s')
    f = open("Experiments/Macro_Test_Results_FFCN.txt", 'w')
    f_str = "PR:" + str(pr_test) + " RC:" + str(rc_test) + " F1:" + str(f1_test)
    f.write(f_str)
    f.close()
    np.savetxt("Experiments/y_true_FFCN_test.csv", np.array(y_true_t).flatten(), delimiter=',')
    np.savetxt("Experiments/y_pre_FFCN_test.csv", np.array(y_pre_t).flatten(), delimiter=',')

     #########################################################################
     # Rule Evaluation
    evaluate_rules_boundary(rules, test_data, test_labels)

