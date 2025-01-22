from bertmaster import run_classifier
import os
import tensorflow as tf
import json
import numpy as np
import lib as lb
#import bertmaster as lm

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
tokenization = run_classifier.tokenization
#base_path = "/Data/divya/exp/experiments/Reinforce/en-hi/flipkard_10aug/8oct/nacl_classifieri/optimized_model" #modify accordingly
#base_path = '/Data/divya/exp/experiments/Reinforce/texar/examples/transformer/temp/generic_it_fr/classifier_eng/model_nopretrain/optimized_model'
base_path='/Data/divya/exp/experiments/Reinforce/emnlp/itallian_task/optimized_mono_model'
init_checkpoint = os.path.join(base_path, 'model.ckpt')
bert_config_file = os.path.join(base_path, 'bert_config.json')
vocab_file = os.path.join(base_path, 'vocab.txt')
processor = run_classifier.ColaProcessor()
label_list = processor.get_labels()   

#we need to feed such data during initialization, can be anything as it is needed for run configuration
BATCH_SIZE = 1024
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 500
OUTPUT_DIR = "/home/rupesh/flipkard/Reiforce/bertmaster/bert_output_NOBPE"

#variables that needed to be modified
#labels = ["positive", "neutral", "negative"] #modify based on the labels that you have
labels = ["positive", "negative"] 
#print(labels)
MAX_SEQ_LENGTH = 128  #modify based on the seq length
is_lower_case = True #modify based on uncased or cased

#variables for configuration
tokenization.validate_case_matches_checkpoint(is_lower_case, init_checkpoint)
bert_config = run_classifier.modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=is_lower_case)
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2





########---Setting Parameters---#########


# Call this function to use this utility
def classifications_english():
  

     run_config = tf.contrib.tpu.RunConfig(
       model_dir=OUTPUT_DIR,
       cluster=None,
       master=None,
       save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
       tpu_config=tf.contrib.tpu.TPUConfig(
       iterations_per_loop=1000,
       num_shards=8,
       per_host_input_for_training=is_per_host))

     #model
     model_fn = run_classifier.model_fn_builder(
       bert_config=bert_config,
       num_labels=len(label_list),
       init_checkpoint=init_checkpoint,
       learning_rate=5e-5,
       num_train_steps=None,
       num_warmup_steps=None,
       use_tpu=False,
       use_one_hot_embeddings=False)

     #estimator
     estimator = tf.contrib.tpu.TPUEstimator(
       use_tpu=False,
       model_fn=model_fn,
       config=run_config,
       train_batch_size=BATCH_SIZE,
       eval_batch_size=BATCH_SIZE,
       predict_batch_size=BATCH_SIZE)
 
     return estimator

def _esentiment(model, in_sentences):
 batch_rewards, batch_samples = [],[]
 #label_list = processor.get_labels()
 MAX_SEQ_LENGTH = 128
 #1
 input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = "positive") for x in in_sentences] # here, "" is just a dummy label
    
 #2
 input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    
 #3
 predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    
 #4
 predictions = model.predict(input_fn=predict_input_fn)
 predictions = [p["probabilities"] for p in list(predictions)]
 predictions = np.asarray(predictions).tolist()
 samples =     [[x] for x in in_sentences] 
 assert len(in_sentences) == len(predictions) == len(samples) 
 print("Showing Results")
 #print(samples)
 #print(np.shape(predictions))
 #print(predictions)
 return predictions , samples



def _esentiment_max_n(model, in_sentences):

 batch_size = len(in_sentences)
 MAX_SEQ_LENGTH = 128    
    
 #2
 input_examples = flat_sent_list_repr(in_sentences, batch_size)
 
 #3
 input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    
 #3
 predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    
 #4
 predictions = model.predict(input_fn=predict_input_fn)
 predictions = [p["probabilities"] for p in list(predictions)]
 predictions = np.asarray(predictions).tolist()
 samples =     [[x.text_a] for x in input_examples] 
 #assert len(predictions) == len(samples) 
 print("Showing Results")
 print(np.shape(predictions))
 print(samples)
 print(np.shape(predictions))
 print(predictions)
 return predictions , samples

def flat_sent_list_repr(in_sentences, batch_size, sample_size=5):
   print("Flat sent representation")
   input_examples=[]
   for src_sent_id in range(batch_size):
      for sample_id in range(sample_size):
         x = in_sentences[src_sent_id][sample_id].value
         print(x)
         input_examples += [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = "positive")]
   #print(input_examples)
   #print(np.shape(input_examples))  
   print("==End==")    
   return input_examples        
         
         
            
  
def sentiment_model(fun_name, data):
         if fun_name == "classifier":
            classifier = classifications_english() # classifications_english() instiantiate and create object of this class and .model() call model fun in that class
            #classification = _esentiment(classifier,data)
            return _esentiment(classifier,data)
     
    #def __call__(self, data):     # Function that can be called by name
    #     return classification(self.classifier, data)
              
def sentiment_model_max(fun_name, data):
         if fun_name == "max_classifier":
            classifier = classifications_english() # classifications_english() instiantiate and create object of this class and .model() call model fun in that class
            #classification = _esentiment(classifier,data)
            return _esentiment_max_n(classifier,data)

        
