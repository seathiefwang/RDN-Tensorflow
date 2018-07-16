import tensorflow.contrib.slim as slim
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os

class RDN(object):

    def __init__(self,img_size=32,global_layers=16,local_layers=8,feature_size=64,scale=2,output_channels=3):
        print("Building RDN...")
        self.img_size = img_size
        self.scale = scale
        self.output_channels = output_channels

        #Placeholder for image inputs
        self.input = x = tf.placeholder(tf.float32,[None,None,None,output_channels])
        #Placeholder for upscaled image ground-truth
        self.target = y = tf.placeholder(tf.float32,[None,None,None,output_channels])

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        """
        Preprocessing as mentioned in the paper, by subtracting the mean
        However, the subtract the mean of the entire dataset they use. As of
        now, I am subtracting the mean of each batch
        """
        mean_x = 127
        image_input =x- mean_x
        mean_y = 127
        image_target =y- mean_y

        scaling_factor = 0.1

        x1 = slim.conv2d(image_input,feature_size,[3,3])

        x = slim.conv2d(x1,feature_size,[3,3])

        outputs = []
        for i in range(global_layers):
            x = utils.resDenseBlock(x, feature_size, layers=local_layers, scale=scaling_factor)
            outputs.append(x)

        x = tf.concat(outputs, 3)
        x = slim.conv2d(x, feature_size, [1,1], activation_fn=None)

        x = slim.conv2d(x,feature_size,[3,3])

        x = x + x1

        x = utils.upsample(x,scale,feature_size)

        #output = slim.conv2d(x,output_channels,[3,3])
        output = tf.layers.conv2d(x, output_channels, (3, 3), padding='same', use_bias=False)

        #l1 loss
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,output))

        self.out = tf.clip_by_value(output+mean_x,0.0,255.0)

        #Calculating Peak Signal-to-noise-ratio
        #Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        mse = tf.reduce_mean(tf.squared_difference(image_target,output))
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        self.PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)

        #Scalar to keep track for loss
        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("PSNR",self.PSNR)
        #Image summaries for input, target, and output
        tf.summary.image("input_image",tf.cast(self.input,tf.uint8))
        tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
        tf.summary.image("output_image",tf.cast(self.out,tf.uint8))

        #Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building!")

    """
    Save the current state of the network to file
    """
    def save(self,savedir='saved_models'):
        print("Saving...")
        self.saver.save(self.sess,savedir+"/model")
        print("Saved!")

    """
    Resume network from previously saved weights
    """
    def resume(self,savedir='saved_models'):
        print("Restoring...")
        self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
        print("Restored!")

    def predict(self,x):
        print('Predicting...')
        return self.sess.run(self.out, feed_dict={self.input:[x],self.is_training:False})

    """
    Function to setup your input data pipeline
    """
    def set_data_fn(self,fn,args,test_set_fn=None,test_set_args=None):
        self.data = fn
        self.args = args
        self.test_data = test_set_fn
        self.test_args = test_set_args

    """
    Train the neural network
    """
    def train(self,iterations=1000,save_dir="saved_models",use_pre=False):
        #Removing previous save directory if there is one
        if not use_pre and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            #Make new save directory
            os.mkdir(save_dir)
        #Just a tf thing, to merge all summaries into one
        merged = tf.summary.merge_all()
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(2e-4)
        #This is the train operation for our objective
        train_op = optimizer.minimize(self.loss)
        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if use_pre:
                self.resume(save_dir)
            test_exists = self.test_data
            #create summary writer for train
            train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            if test_exists:
                test_writer = tf.summary.FileWriter(save_dir+"/test",sess.graph)
                test_x,test_y = self.test_data(*self.test_args)
                #test_feed = {self.input:test_x,self.target:test_y,self.is_training:False}
                

            #This is our training loop
            for i in tqdm(range(iterations)):
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data(*self.args)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y,
                    self.is_training:True
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([merged,train_op],feed)
                #If we're testing, don't train on test set. But do calculate summary
                if test_exists and i%1000==0:
                    #t_summary = sess.run(merged,test_feed)
                    #test_writer.add_summary(t_summary,i)
                    tloss, tpsnr = 0, 0
                    length = len(test_x)
                    for idx in range(length):
                        test_feed = {self.input:[test_x[idx]],self.target:[test_y[idx]],self.is_training:False}
                        loss, psnr = sess.run([self.loss, self.PSNR], test_feed)
                        tloss, tpsnr = tloss+loss, tpsnr+psnr
                    summary_loss, summary_psnr = tf.Summary(), tf.Summary()
                    summary_loss.value.add(tag="loss", simple_value=tloss/length)
                    summary_psnr.value.add(tag="PSNR", simple_value=tpsnr/length)
                    test_writer.add_summary(summary_loss, i)
                    test_writer.add_summary(summary_psnr, i)
                if i % 100 == 0:
                    #Write train summary for this step
                    train_writer.add_summary(summary,i)

                if (i+1) % (iterations//5) == 0:
                    self.save()
            #Save our trained model
            self.save()
