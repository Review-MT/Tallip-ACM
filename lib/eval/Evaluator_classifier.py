from __future__ import division
import lib as lb
import re
import torch
import numpy as np
import csv
import pandas as pd
from torch.autograd import Variable
from itertools import chain 
import os
class Evaluator(object):
    def __init__(self, model, metrics, dicts, opt, mode = None):
        self.model = model
        self.loss_func = metrics["nmt_loss"]
        self.sent_reward_func = metrics["sent_reward"]   
        self.corpus_reward_func = metrics["corp_reward"]
        self.classifier =metrics["classifier_reward"]
        self.dicts = dicts
        self.max_length = opt.max_predict_length
        self.report_gold_reward =[]
        self.opt = opt
        self.reward = opt.reward_model
    def eval(self, data, epoch =0, sbleu_reward=None):
        self.model.eval()
        self.current_epoch = epoch
        print("Evaluating Model")
        if sbleu_reward:self.reward = 'sent_reward' # Set so that SBLEU reward used in eval, as other time consuming
        total_loss =  0
        total_words = 0
        total_sents = 0
        total_reinforce_reward = 0
        
        all_preds = []
        all_targets = []
        all_sources = []
        all_rewards = []
        
        all_gold_rewards = []
        all_bert_reward = [] #used for calculating classifier accuracy
        accumulate_reward =[]
        #
        for i in range(len(data)):
            batch = data[i]
            
            targets = batch[1]
            sources  = batch[0][0]
            if self.reward == 'harmonic_reward' or self.reward == 'bert_reward':
               gold = batch[0][2]   # TODO GOLD REWARD for Eval Data set
               #print("%%")
               print(gold)
               print(np.shape(gold))
               
               
            else:
               gold = '4'  #TODO 4 is not valid reward, temp fix
            all_gold_rewards.extend(gold)
            attention_mask = batch[0][0].data.eq(lb.Constants.PAD).t()
            self.model.decoder.attn.applyMask(attention_mask)
            outputs = self.model(batch, True)
            

            weights = targets.ne(lb.Constants.PAD).float()
            num_words = weights.data.sum()
            _, loss = self.model.predict(outputs, targets, weights, self.loss_func)

            preds = self.model.translate(batch, self.max_length)
            
            preds = preds.t().tolist()
            targets = targets.data.t().tolist()
            sources = sources.data.t().tolist() # BS * SL
            #reinfore_rewards, bert_reward = self.reward_module(preds, targets, sources, gold)

                        
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_sources.extend(sources)
            

            total_loss += loss
            total_words += num_words
            
     
            total_sents += batch[1].size(1)
        #Called here to speed up and avoid classifier call for each batch
        reinfore_rewards, bert_reward = self.reward_module(all_preds, all_targets, # bert_reward 1X3
                                                    all_sources, all_gold_rewards)
        all_rewards.extend(reinfore_rewards)
        #print(reinfore_rewards)
        
        #print("++"*10)
        #print(all_rewards) 
        #exit(0)
        #print(all_rewards[0])
        all_bert_reward.extend(bert_reward)
        total_reinforce_reward += sum(all_rewards)
        print(total_reinforce_reward)
       
        loss = total_loss / total_words
        reinforce_reward = total_reinforce_reward / total_sents
        print("<<<"*3)
        print(reinforce_reward)
        
        corpus_reward = self.corpus_reward_func(all_preds, all_targets)

        print(np.shape(all_preds))
        psents, tsents, ssents = self.convert_id2token(all_preds, all_targets, all_sources)
        print(np.shape(psents))
     
        if self.reward == 'harmonic_reward' or self.reward == 'bert_reward':
            lb.utils.classifier_statistic(all_gold_rewards, all_bert_reward) 
        lb.utils.write_statistic_to_file(ssents, psents, tsents, all_gold_rewards, all_rewards, str(self.current_epoch))
        print("Done Evalutaion")
      
        return loss, reinforce_reward, corpus_reward  


    def convert_id2token(self, preds, targets, sources):
        ssents, psents, tsents=[], [], []

        assert len(sources) == len(targets) == len(preds)
        for psent,tsent, ssent in zip(preds, targets, sources):
            psent = lb.Reward.clean_up_sentence(psent, remove_unk=False, remove=True)
            tsent = lb.Reward.clean_up_sentence(tsent, remove_unk=False, remove=True)
            ssent = lb.Reward.clean_up_sentence(ssent, remove_unk=False, remove=True)
            psent = [self.dicts["tgt"].getLabel(w) for w in psent]
            tsent = [self.dicts["tgt"].getLabel(w) for w in tsent]
            ssent = [self.dicts["src"].getLabel(w) for w in ssent]
            
            psent=re.sub(r'(@@ )|(@@ ?$)',""," ".join(psent))
            tsent=re.sub(r'(@@ )|(@@ ?$)',""," ".join(tsent))
            ssent=re.sub(r'(@@ )|(@@ ?$)',""," ".join(ssent))
            psents.append(psent)
            tsents.append(tsent)
            ssents.append(ssent)
        print(ssents)

        return psents, tsents, ssents        

    def reward_module(self,preds, targets, sources,  all_gold_rewards):
      if self.reward == 'sent_reward':
         sent_rewards, _ = self.sent_reward_func(preds, targets)
        
         
         print("**"*20)
         
         return sent_rewards, '4'  # 4 is dummy value, temp fix
      elif self.reward == 'bert_reward':
         psents, tsents, ssents = self.convert_id2token(preds, targets, sources)
         print(np.shape(targets), np.shape(all_gold_rewards))
         print(targets[:10],all_gold_rewards[:10])
         
         
         bert_pred, _ = self.classifier("classifier", psents)
         bert_rewards = lb.metric.msent_reward_func(bert_pred,  all_gold_rewards, mode_convert = False)
         print("=="*20)
        
         return bert_rewards, bert_pred

      elif self.reward == 'harmonic_reward':
         sent_rewards, _ = self.sent_reward_func(preds, targets)
         psents, tsents, ssents = self.convert_id2token(preds, targets, sources)
         #print(tsents[:10],all_gold_rewards[:10])
         #exit(0)
         bert_pred, _ = self.classifier("classifier", psents)
         bert_rewards = lb.metric.msent_reward_func(bert_pred,  all_gold_rewards, mode_convert = False)     
         harmonic_rewards = lb.metric.harmonic(bert_rewards, sent_rewards) 
         print("--"*20)
         return harmonic_rewards, bert_pred
      else:
         raise ValueError('This is not a suitable reward type')



          
      
        
      
