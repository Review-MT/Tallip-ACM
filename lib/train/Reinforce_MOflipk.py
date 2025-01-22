import datetime
import math
import os
import time
import numpy as np
from torch.autograd import Variable
import torch
import pandas as pd
import lib as lb
import re

##==============================
from collections import namedtuple
from typing import List, Tuple, Dict, Set, Union
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
##===============================
class Reinforce_MOflipk(object):

    def __init__(self, actor, train_data, eval_data, metrics, dicts, optim, opt):
        self.actor = actor
        
        self.sent_reward_func = metrics["sent_reward"]   
        self.corpus_reward_func = metrics["corp_reward"]
        self.max_classifier = metrics["classifier_max_reward"]
            
          
        self.train_data = train_data
       
       
        self.eval_data = eval_data
        self.evaluator = lb.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["nmt_loss"]
        self.dicts = dicts

        self.optim = optim
        self.max_length = opt.max_predict_length
    
        self.opt = opt
        self.sbleu_reward = False #Turnon this flag if want other reward in evaluation 
        print("==")
        print(actor)
        print("==")
       

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss  = None
        self.optim.set_lr(self.opt.reinforce_lr)
 

        for epoch in range(start_epoch, end_epoch + 1):
            print("")
            
            print("* REINFORCE epoch with %s*" % self.reward)
            print("Actor optim lr: %g" %
                (self.optim.lr))
        
            no_update = self.opt.no_update  and \
                                   (epoch == start_epoch)
           
        

            if no_update: print("No update...")
            
            
            train_reward, actor_loss = self.train_epoch(epoch, no_update)
            
            print("Train sentence reward: %.2f" % (train_reward *100))
            print("Actor loss: %g" % actor_loss)
 
            
            self.sbleu_reward = False # set this to true, for using only BLEU as reward in evaluation
            valid_loss, valid_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data, epoch, self.sbleu_reward)
            valid_ppl = math.exp(min(valid_loss, 100))
            print("")
            print("Reported Result for %s data set " % self.eval_data.data)
            print("Validation perplexity: %.2f" % valid_ppl)
            print("Loss: %.6f" % valid_loss)
            print("Train Reward: %.2f" % (valid_reward * 100))
            print("Corpus reward: %.2f" % (valid_corpus_reward * 100))


            if no_update: break



            self.optim.updateLearningRate(-valid_reward, epoch)

            checkpoint = {
                "model": self.actor,
                "dicts": self.dicts,
                "opt": self.opt,
                "epoch": epoch,
                "optim": self.optim,
     
            }
            model_name = os.path.join(self.opt.save_dir, "model_%d" % epoch)
          
            model_name += "_reinforce"
            model_name += ".pt"
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)
            


           
    def fix_sample(self, sample): #fix the sample to by padding to length of prediction
          length = len(sample)
          print(length)
          print(sample)
          if lb.Constants.EOS in sample:
             sample = sample[:sample.index(lb.Constants.EOS) + 1] 
             len_pred = len(sample) 
             if len_pred == 0:
                sample = [lb.Constants.PAD] * length
             else:
                while len(sample) < length:
                  sample.append(lb.Constants.PAD) 
             #if len(sample) > 0 and (sample[-1] == lb.Constants.EOS or sample[-1] == lb.Constants.PAD): 
             #   sample = sample[:-1]        
          return sample   

    def train_epoch(self, epoch, no_update):
        self.actor.train()

        total_reward, report_reward = 0, 0, 
        total_actor_loss, report_actor_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
       
        for i in range(len(self.train_data)):
            batch   = self.train_data[i]
            sources = batch[0]
            targets = batch[1]
            batch_size = targets.size(1)
            gold = sources[2]
            self.actor.zero_grad()
           
            #=================================================
            #Sample translations
            attention_mask = sources[0].data.eq(lb.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)
            #=================================================
            candidate_set = {}
            
            sample_size=5
            
            completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
            completed_outputs = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
            ksamples , koutputs = [], []
            
            for k in range(sample_size): 
               samples, outputs = self.actor.sample(batch, self.max_length) #sample : sl x bs
               print(samples.shape)
               print(outputs.shape)
               for i, s in enumerate(samples.t()):
                  print("*", s.shape)
                  print("^", outputs.transpose(0,1)[i,:,:].shape)
                  completed_samples[i][k] = s.tolist()
                  completed_outputs[i][k] = outputs.transpose(0,1)[i,:,:]
            
            
            #print(np.shape(completed_samples))
            print(completed_samples)
            #print(np.shape(completed_outputs))
          
            #Calculate rewards  
            reinfore_rewards , select_sample_id = self.max_reward_sample(completed_samples, targets , gold, self.dicts, batch_size)
            #reinfore_rewards, bert_reward = self.reward(samples.t().tolist(), targets.data.t().tolist(), sources)
            #selected_idx = [idx * batch_size + i for i, idx in enumerate(select_sample_id)]
            
            print(completed_samples)
            
            samples = [samples[idx] for idx, samples in zip(select_sample_id , completed_samples)] #bsxsl
            outputs = [outputs[idx] for idx, outputs in zip(select_sample_id , completed_outputs)]
            
            print(type(outputs))
            #print(outputs.shape)
            outputs = torch.stack(outputs)
            print(type(outputs))
            #print(samples)
            print(outputs.shape)

            print(samples)
            print(outputs) 
            decoded = samples
           
            #check samples
            print("Maximum rewarding Target Sample")
            sent = [self.dicts["tgt"].convertToLabels(s, stop=3)[:-1] for s in samples]
            print(sent) 
            print("Original Source Sentence")
            sent = [self.dicts["src"].convertToLabels(s.tolist(), stop =0)[:-1] for s in sources[0].transpose(0,1)]
            print(sent)
          
            reinfore_reward = sum(reinfore_rewards)
            #print(np.shape(samples))
            #print(np.shape(outputs))
            #fix sample
            samples = list(map(self.fix_sample, samples)) #bsxsl
            print(np.shape(samples)) 
          
        
            
            samples = Variable(torch.LongTensor(samples).t().contiguous()) #slxbs
            outputs = outputs.transpose(0,1)
            print(samples.shape) 
            print(outputs.shape)   
            reinfore_rewards = Variable(torch.FloatTensor([reinfore_rewards] * samples.size(0)).contiguous()) #samples.size(0) =SL
            print(reinfore_rewards.shape)
            
            if self.opt.cuda:
                #print("*Rbleu*",type(reinfore_rewards))
                samples = samples.to('cuda')
                #print("*Sample*",type(samples))
                reinfore_rewards = reinfore_rewards.to('cuda')

            # Update   
            actor_weights = samples.ne(lb.Constants.PAD).float()
            num_words = actor_weights.data.sum()
            
            # Update actor
            if not no_update:
                # Subtract baseline from reward
                norm_rewards = Variable((reinfore_rewards).data)
                actor_weights = norm_rewards * actor_weights
                
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                print(outputs.shape, samples.shape)
                
                actor_loss = self.actor.backward(outputs, samples, actor_weights, num_words, self.actor_loss_func)
                self.optim.step()
            else :
                actor_loss =0
            # Gather stats
 
            report_reward += reinfore_reward
            total_reward += reinfore_reward

            total_sents += batch_size
            report_sents += batch_size

            total_actor_loss += actor_loss
            report_actor_loss += actor_loss

            total_words += num_words
            report_words += num_words
          
            print("Epoch %d,%d Reinforce Training using % s  reward" %(epoch, i, self.opt.reward_model))
            if i % self.opt.log_interval == 0 and i > 0:  #TODO Add printf for model and harmonic reward
                     print("""Epoch %3d, %6d/%d batches;                                 
                        actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                        (epoch, i, len(self.train_data),
                        (report_reward / report_sents) * 100,
                        report_actor_loss / report_words,
                        report_words / (time.time() - last_time),
                        str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                     report_reward = report_sents = report_actor_loss = report_words = 0
                     last_time = time.time()
      
        return total_reward / total_sents, total_actor_loss / total_words
#=======================================================================

    def flat_list_representation(self, _input_list, batch_size, sample_size=5  ):
      flat_list=[]
      for bid in range(batch_size):
         for sid in range(sample_size):
            flat_list += [_input_list[bid][sid]]
       
      return flat_list   
    
    
    def max_reward_sample(self, raw_samples, targets, gold, dicts, batch_size,  \
           sample_size=5):
        '''
        return Max reward and corresponding Sample from each batch
        '''
        
        if self.opt.reward_model == 'sent_reward':
           b_sent_rewards =[]
           for pair in zip(raw_sample, targets): #one batch of sample
              top_5_sample, target = pair 
              s_sent_rewards, _  = list(map(lambda sample: self.sent_reward_func(list(sample), list(target)) , top_5_sample)) #reward wrt each sample in batch
              b_sent_rewards += sent_rewards
              #sent_rewards, _ = self.sent_reward_func(preds, targets)
              
           return sent_rewards, None
        elif self.opt.reward_model == 'bert_reward':
           #convert ids to token for input to classifier
           sent_samples = self.convert_index2Label_topfive(raw_samples, dicts, batch_size)
           print("===Input to Classifier===")
           print(sent_samples)
           bert_pred, _ = self.max_classifier("max_classifier", sent_samples)
           print(np.shape(bert_pred))
           #
           
           print("== Broadcast each Gold Reward in a batch to sampe size ==")
           print(gold)
           gold = [[gold[g]] * sample_size  for g in range(batch_size)]
           print(gold)
           print(np.shape(gold))
           print("***"*3)
          
           #
           print("Flat to match shape of classifier prediction")
           gold = self.flat_list_representation(gold, batch_size)
           print(gold)
           print("One Hot Representaion") 
           gold = list(map(lb.metric.reward_conversion, gold)) 
           print(gold)
           print(np.shape(gold), np.shape(bert_pred))
           
           #
           print("== Classifier Reward for gold class ==")  
           reward = lb.metric.msent_reward_func(bert_pred, gold, mode_convert = True) # as gold reward is already onehot
           print(reward)
           print("Group flat reward in n samples")
           group_rwd = [reward[i:i + sample_size] for i in range(0, batch_size*sample_size , sample_size)] #group as per 5 sample in batch 
           print(group_rwd)
           
           #
           print("Max rewarding sample index")
           max_rwd_smp_idx = [rewards.index(max(rewards)) for rewards in group_rwd]
           bert_rewards  = [ rwd[idx] for rwd, idx in zip(group_rwd,max_rwd_smp_idx)]
           print(bert_rewards)
           print(max_rwd_smp_idx)
          
           return bert_rewards, max_rwd_smp_idx

        elif self.opt.reward_model == 'harmonic_reward':
           
           sent_rewards, _ = self.sent_reward_func(preds, targets)
           psents, tsents, ssents = self.evaluator.convert_id2token(preds, targets, sources[0].data.t().tolist() )
           bert_pred, _ = self.max_classifier("classifier", psents)
           bert_rewards = lb.metric.msent_reward_func(bert_pred, sources[2], mode_convert = False)     
           harmonic_rewards = lb.metric.harmonic(bert_rewards, sent_rewards) 
           return harmonic_rewards, bert_pred
        else:
           raise ValueError('This is not a suitable reward type')    
     

    def convert_index2Label_topfive(self, _completed_samples, dicts, batch_size, sample_size=5):
       completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
       print("Maping to sentences.")
       for src_sent_id in range(batch_size): #row is batch
          for sample_id in range(sample_size): #column
             print("====") 
             print(_completed_samples[src_sent_id][sample_id])
             if lb.Constants.EOS in _completed_samples[src_sent_id][sample_id]:             
               index = _completed_samples[src_sent_id][sample_id].index(3)
               value= dicts["tgt"].convertToLabelss(_completed_samples[src_sent_id][sample_id])[:index]
             else:
               value= dicts["tgt"].convertToLabelss(_completed_samples[src_sent_id][sample_id])  
             value= re.sub(r'(@@ )|(@@ ?$)',""," ".join(value))
             
             #print(value)
             hyp = Hypothesis(value= value,
                                 score=0)
             print(hyp)                  
             completed_samples[src_sent_id][sample_id] = hyp
       return completed_samples       
      
       
    def reward(self, preds, targets, sources): # sl x bs sources is a 3 tuple here (sentences, length,indics)
       
             
        if self.opt.reward_model == 'sent_reward':
           sent_rewards, _ = self.sent_reward_func(preds, targets)
           return sent_rewards, None
        elif self.opt.reward_model == 'bert_reward':

           psents, tsents, ssents = self.evaluator.convert_id2token(preds, targets, sources[0].data.t().tolist() ) 
           bert_pred, _ = self.classifier("classifier", psents)
         
           print(sources)
           print(sources[0])
           print(sources[2])
           
           print(np.shape(sources[0]))
           print(np.shape(sources[1]))
           print(np.shape(psents),np.shape(tsents),np.shape(ssents))
           #print(psents,tsents,ssents)    
           
           bert_rewards = lb.metric.msent_reward_func(bert_pred, sources[2], mode_convert = False)
           
           return bert_rewards, bert_pred

        elif self.opt.reward_model == 'harmonic_reward':
           
           sent_rewards, _ = self.sent_reward_func(preds, targets)
           psents, tsents, ssents = self.evaluator.convert_id2token(preds, targets, sources[0].data.t().tolist() )
           bert_pred, _ = self.classifier("classifier", psents)
           bert_rewards = lb.metric.msent_reward_func(bert_pred, sources[2], mode_convert = False)     
           harmonic_rewards = lb.metric.harmonic(bert_rewards, sent_rewards) 
           return harmonic_rewards, bert_pred
        else:
           raise ValueError('This is not a suitable reward type')

