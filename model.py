"""
    The ``model`` module
    ======================
 
    Contains the class Model which implements the core model for CG detection, 
    training, testing and visualization functions.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import random
import image_loader as il
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import csv
import configparser

import numpy as np

from PIL import Image

GPU = '/gpu:0'
config = 'server'

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score as acc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

import pickle

# seed initialisation

random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility


# tool functions


def image_summaries(var, name):
  tf.summary.image(name + '_1', var[:,:,:,0:1], max_outputs = 1)
  tf.summary.image(name + '_2', var[:,:,:,1:2], max_outputs = 1)
  tf.summary.image(name + '_3', var[:,:,:,2:3], max_outputs = 1)
  # tf.summary.image(name + '_4', var[:,:,:,3:4], max_outputs = 1)
  # tf.summary.image(name + '_5', var[:,:,:,4:5], max_outputs = 1)
  # tf.summary.image(name + '_6', var[:,:,:,5:6], max_outputs = 1)
  # tf.summary.image(name + '_7', var[:,:,:,6:7], max_outputs = 1)
  # tf.summary.image(name + '_8', var[:,:,:,7:8], max_outputs = 1)

def filter_summary(filters, name):
  tf.summary.image(name + '_1', tf.stack([filters[:,:,0,0:1]]), max_outputs = 1)
  tf.summary.image(name + '_2', tf.stack([filters[:,:,0,1:2]]), max_outputs = 1)
  tf.summary.image(name + '_3', tf.stack([filters[:,:,0,2:3]]), max_outputs = 1)
  tf.summary.image(name + '_4', tf.stack([filters[:,:,0,3:4]]), max_outputs = 1)
  tf.summary.image(name + '_5', tf.stack([filters[:,:,0,4:5]]), max_outputs = 1)
  tf.summary.image(name + '_6', tf.stack([filters[:,:,0,5:6]]), max_outputs = 1)
  # tf.summary.image(name + '_7', tf.stack([filters[:,:,0,6:7]]), max_outputs = 1)
  # tf.summary.image(name + '_8', tf.stack([filters[:,:,0,7:8]]), max_outputs = 1)

def weight_variable(shape, nb_input, seed = None):
  """Creates and initializes (truncated normal distribution) a variable weight Tensor with a defined shape"""
  sigma = np.sqrt(2/nb_input)
  # print(sigma)
  initial = tf.truncated_normal(shape, stddev=sigma, seed = random_seed)
  return tf.Variable(initial)

def bias_variable(shape):
  """Creates and initializes (truncated normal distribution with 0.5 mean) a variable bias Tensor with a defined shape"""
  initial = tf.truncated_normal(shape, mean = 0.5, stddev=0.1, seed = random_seed)
  return tf.Variable(initial)
  
def conv2d(x, W):
  """Returns the 2D convolution between input x and the kernel W"""  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def gaussian_func(mu, x, n, sigma):
  """Returns the average of x composed with a gaussian function

    :param mu: The mean of the gaussian function
    :param x: Input values 
    :param n: Number of input values
    :param sigma: Variance of the gaussian function
    :type mu: float
    :type x: Tensor
    :type n: int 
    :type sigma: float
  """ 
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  # return(tf.reduce_sum(gauss.pdf(xmax - tf.nn.relu(xmax - x))/n))
  return(tf.reduce_sum(gauss.pdf(x)/n))

def gaussian_kernel(x, nbins = 8, values_range = [0, 1], sigma = 0.1,image_size = 100):
  """Returns the values of x's nbins gaussian histogram 

    :param x: Input values (supposed to be images)
    :param nbins: Number of bins (different gaussian kernels)
    :param values_range: The range of the x values
    :param sigma: Variance of the gaussian functions
    :param image_size: The size of the images x (for normalization)
    :type x: Tensor
    :type nbins: int 
    :type values_range: table
    :type sigma: float
    :type image_size: int
  """ 
  mu_list = np.float32(np.linspace(values_range[0], values_range[1], nbins + 1))
  n = np.float32(image_size**2)
  function_to_map = lambda m : gaussian_func(m, x, n, sigma)
  return(tf.map_fn(function_to_map, mu_list))

def classic_histogram_gaussian(x, k, nbins = 8, values_range = [0, 1], sigma = 0.6):
  """Computes gaussian histogram values for k input images"""
  function_to_map = lambda y: tf.stack([gaussian_kernel(y[:,:,i], nbins, values_range, sigma) for i in range(k)])
  res = tf.map_fn(function_to_map, x)
  return(res)

def compute_stat(x, k):
  """Computes statistical features for k images"""
  # function_to_map = lambda y: tf.stack([stat(y[:,:,i]) for i in range(k)])
  # res = tf.map_fn(function_to_map, x)
  res = tf.transpose(tf.stack([tf.reduce_mean(x, axis=[1,2]), tf.reduce_min(x, axis=[1,2]), tf.reduce_max(x, axis=[1,2]), tf.reduce_mean((x - tf.reduce_mean(x, axis=[1,2], keep_dims = True))**2, axis=[1,2])]), [1,2,0])
  return(res)

def extract_subimages(image, subimage_size, only_green = True):
    
    nb_channels = 3          # return only the green channel of the images
    
    if(only_green == True) :
      nb_channels = 1
    
    subimages = []
    width = image.size[0]
    height = image.size[1]

    current_height = 0
    
    while current_height + subimage_size <= height: 
        current_width = 0
        while current_width + subimage_size <= width: 
            box = (current_width, current_height, 
                    current_width + subimage_size, 
                    current_height + subimage_size)
            sub = np.asarray(image.crop(box))
            if len(sub.shape) > 2: 
                if nb_channels == 1:
                    subimages.append(sub[:,:,1].astype(np.float32)/255)
                else:
                    subimages.append(sub.astype(np.float32)/255)
            else: 
                subimages.append(sub.astype(np.float32)/255)
            current_width += subimage_size

        current_height += subimage_size

    nb_subimages = len(subimages)

    return((np.reshape(np.array(subimages), (nb_subimages, subimage_size, subimage_size, nb_channels)), width, height))


class Model:

  """
    Class Model
    ======================
 
    Defines a model for single-image CG detection and numerous methods to : 
    - Create the TensorFlow graph of the model
    - Train the model on a specific database
    - Reload past weights 
    - Test the model (simple classification, full-size images with boosting and splicing)
    - Visualize some images and probability maps
"""

  def __init__(self, database_path='', image_size=100, config = 'Personal', filters = [32, 64],
              feature_extractor = 'Stats', remove_context = False, 
              nbins = 10, remove_filter_size = 3, batch_size = 50, 
              using_GPU = False, only_green = True):
    """Defines a model for single-image classification

    :param database_path: Absolute path to the default patch database (training, validation and testings are performed on this database)
    :param image_size: Size of the patches supposed squared
    :param config: Name of the section to use in the config.ini file for configuring directory paths (weights, training summaries and visualization dumping)
    :param filters: Table with the number of output filters of each layer
    :param feature_extractor: Two choices 'Stats' or 'Hist' for the feature extractor
    :param nbins: Number of bins on the histograms. Used only if the feature_extractor parameter is 'Hist'
    :param batch_size: The size of the batch for training
    :param using_GPU: Whether to use GPU for computation or not 
    
    :type database_path: str
    :type image_size: int
    :type config: str
    :type filters: table
    :type feature_extractor: str
    :type nbins: int
    :type batch_size: int
    :type using_GPU: bool
  """ 

    
    # read the configuration file
    conf = configparser.ConfigParser()
    conf.read('config.ini')

    if config not in conf:
      raise ValueError(config + ' is not in the config.ini file... Please create the corresponding section')
    
    self.dir_ckpt = conf[config]['dir_ckpt']
    self.dir_summaries = conf[config]['dir_summaries']
    self.dir_visualization = conf[config]['dir_visualization']


    # setting the parameters of the model
    self.nf = filters
    self.nl = len(self.nf)
    self.filter_size = 3

    self.feature_extractor = 'Stats'

    if self.feature_extractor != 'Stats' and self.feature_extractor != 'Hist':
      raise ValueError('''Feature extractor must be 'Stats' or 'Hist' ''')

    self.database_path = database_path
    self.image_size = image_size
    self.batch_size = batch_size
    self.nbins = nbins
    self.using_GPU = using_GPU
    self.remove_context = remove_context
    self.remove_filter_size = remove_filter_size
    self.only_green = only_green

    # getting the database
    #self.import_database()

    self.nb_channels = 1
    self.nb_class = 2
    self.create_graph(nb_class = self.nb_class, 
                        feature_extractor = self.feature_extractor,
                        nl = self.nl, nf = self.nf, filter_size = self.filter_size)

  def create_graph(self, nb_class, nl = 2, nf = [32, 64], filter_size = 3,
                   feature_extractor = 'Stats'): 
    """Creates the TensorFlow graph"""


    # input layer. One entry is a float size x size, 3-channels image. 
    # None means that the number of such vector can be of any lenght.


    graph = tf.Graph()

    with graph.as_default():

      with tf.name_scope('Input_Data'):
        x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.nb_channels])
        self.x = x
        # reshape the input data:
        x_image = tf.reshape(x, [-1,self.image_size, self.image_size, self.nb_channels])
        with tf.name_scope('Image_Visualization'):
          tf.summary.image('Input_Data', x_image)
        

      # first conv net layer
      with tf.name_scope('Conv1'):

        with tf.name_scope('Weights'):
          if self.remove_context:
            W_conv1 = weight_variable([self.remove_filter_size, self.remove_filter_size, self.nb_channels, nf[0]], 
                                      nb_input = self.remove_filter_size*self.remove_filter_size*self.nb_channels,
                                      seed = random_seed)
          else:
            W_conv1 = weight_variable([self.filter_size, self.filter_size, self.nb_channels, nf[0]], 
                                      nb_input = self.filter_size*self.filter_size*self.nb_channels,
                                      seed = random_seed)
          self.W_conv1 = W_conv1
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nf[0]])


        # relu on the conv layer
        if self.remove_context: 
          h_conv1 = conv2d(x_image, W_conv1)
        else:         
          h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, 
                               name = 'Activated_1')
        self.h_conv1 = h_conv1

      self.W_convs = [W_conv1]
      self.b_convs = [b_conv1]
      self.h_convs = [h_conv1]

      image_summaries(self.h_convs[0], 'hconv1')
      filter_summary(self.W_convs[0], 'Wconv1')

      for i in range(1, nl):
        # other conv 
        with tf.name_scope('Conv' + str(i+1)):
          with tf.name_scope('Weights'):
            W_conv2 = weight_variable([self.filter_size, self.filter_size, nf[i-1], nf[i]],
                                      self.filter_size*self.filter_size*nf[i-1])
            self.W_convs.append(W_conv2)
          with tf.name_scope('Bias'):
            b_conv2 = bias_variable([nf[i]])
            self.b_convs.append(b_conv2)

          h_conv2 = tf.nn.relu(conv2d(self.h_convs[i-1], W_conv2) + b_conv2, 
                               name = 'Activated_2')

          self.h_convs.append(h_conv2)    



      nb_filters = nf[nl-1]
      if self.feature_extractor == 'Hist':
        # Histograms
        nbins = self.nbins
        size_flat = (nbins + 1)*nb_filters

        range_hist = [0,1]
        sigma = 0.07

        # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = sigma)

        with tf.name_scope('Gaussian_Histogram'): 
          hist = classic_histogram_gaussian(self.h_convs[nl-1], k = nb_filters, 
                                            nbins = nbins, 
                                            values_range = range_hist, 
                                            sigma = sigma)
          self.hist = hist

        flatten = tf.reshape(hist, [-1, size_flat], name = "Flatten_Hist")
        self.flatten = flatten

      else: 
        nb_stats = 4
        size_flat = nb_filters*nb_stats
        with tf.name_scope('Simple_statistics'): 
          s = compute_stat(self.h_convs[nl-1], nb_filters)
          self.stat = s
          
        flatten = tf.reshape(s, [-1, size_flat], name = "Flattened_Stat")
        self.flatten = flatten



      # Densely Connected Layer
      # we add a fully-connected layer with 1024 neurons 
      with tf.variable_scope('Dense1'):
        with tf.name_scope('Weights'):
          W_fc1 = weight_variable([size_flat, 1024],
                                  nb_input = size_flat)
        with tf.name_scope('Bias'):
          b_fc1 = bias_variable([1024])
        # put a relu
        h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1, 
                           name = 'activated')

      # dropout
      with tf.name_scope('Dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      self.h_fc1 = h_fc1

      # readout layer
      with tf.variable_scope('Readout'):
        with tf.name_scope('Weights'):
          W_fc3 = weight_variable([1024, nb_class],
                                  nb_input = 1024)
        with tf.name_scope('Bias'):
          b_fc3 = bias_variable([nb_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

      self.y_conv = y_conv

      # support for the learning label
      y_ = tf.placeholder(tf.float32, [None, nb_class])
      self.y_ = y_



      # Define loss (cost) function and optimizer

      # softmax to have normalized class probabilities + cross-entropy
      with tf.name_scope('cross_entropy'):

        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
        with tf.name_scope('total'):
          cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

      tf.summary.scalar('cross_entropy', cross_entropy_mean)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

      # with tf.name_scope('enforce_constraints'):
      if self.remove_context:
        # self.zero_op = tf.assign(ref = self.W_convs[0][1,1,0,:], value = tf.zeros([nf[0]]))
        center = int(self.remove_filter_size/2)
        self.zero_op = tf.scatter_nd_update(ref = self.W_convs[0], indices = tf.constant([[center,center,0,i] for i in range(nf[0])]), updates = tf.zeros(nf[0]))
        self.norm_op = tf.assign(ref = self.W_convs[0], value = tf.divide(self.W_convs[0],tf.reduce_sum(self.W_convs[0], axis = 3, keep_dims = True)))
        self.minus_one_op = tf.scatter_nd_update(ref = self.W_convs[0], indices = tf.constant([[center,center,0,i] for i in range(nf[0])]), updates = tf.constant([-1.0 for i in range(nf[0])]))
        self.norm = tf.reduce_sum(self.W_convs[0], axis = 3, keep_dims = True)

      self.train_step = train_step
      # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

      # 'accuracy' is a function: cast the boolean prediction to float and average them
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

      self.accuracy = accuracy

    self.graph = graph

  def classify_one_image(self, image, decision_rule='majority_vote'):

    minibatch_size = 25
    valid_decision_rule = ['majority_vote', 'weighted_vote']
    
    if decision_rule not in valid_decision_rule:
      raise NameError(decision_rule + ' is not a valid decision rule.')
    
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = 'final_weights.ckpt'
      saver.restore(sess, self.dir_ckpt + file_to_restore)
      
      batch, width, height = extract_subimages(image, 100)

      batch_size = batch.shape[0]
      
      j = 0
      prediction = 0
      diff = []
     
      while j < batch_size:
        feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
        pred = self.y_conv.eval(feed_dict)
        
        label_image = np.argmax(pred, 1)
        d =  np.max(pred, 1) - np.min(pred, 1)
        for k in range(d.shape[0]):
          diff.append(np.round(d[k], 1))

        if decision_rule == 'majority_vote':
          prediction += np.sum(label_image)
        if decision_rule == 'weighted_vote':
          prediction += np.sum(2*d*(label_image - 0.5))


        j+=minibatch_size
      
      if decision_rule == 'majority_vote':
        prediction = int(np.round(prediction/batch_size))
      if decision_rule == 'weighted_vote':
        prediction = int(max(prediction,0)/abs(prediction))
      

    return prediction
  
  def classify_list_of_images(self, imageList, decision_rule='majority_vote'):
    list_of_predictions = []
    minibatch_size = 25
    valid_decision_rule = ['majority_vote', 'weighted_vote']
    
    if decision_rule not in valid_decision_rule:
      raise NameError(decision_rule + ' is not a valid decision rule.')
    
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = 'final_weights.ckpt'
      saver.restore(sess, self.dir_ckpt + file_to_restore)
      
      for image in imageList:
        batch, width, height = extract_subimages(image, 100)
        batch_size = batch.shape[0]
        j = 0
        prediction = 0
        while j < batch_size:
          feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
          pred = self.y_conv.eval(feed_dict)
        
          label_image = np.argmax(pred, 1)
          d =  np.max(pred, 1) - np.min(pred, 1)

          if decision_rule == 'majority_vote':
            prediction += np.sum(label_image)
          if decision_rule == 'weighted_vote':
            prediction += np.sum(2*d*(label_image - 0.5))

          j+=minibatch_size
      
        if decision_rule == 'majority_vote':
          prediction = int(np.round(prediction/batch_size))
        if decision_rule == 'weighted_vote':
          prediction = int(max(prediction,0)/abs(prediction))

        list_of_predictions.append(prediction)

    return list_of_predictions





