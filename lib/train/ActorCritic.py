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
class ActorCritic(object):

    def __init__(self, actor, critic, train_data, eval_data, metrics, dicts, optim, critic_optim, opt): #++++
        #+++
        self.actor = actor
        self.critic = critic
        self.sent_reward_func = metrics["sent_reward"]   
        self.corpus_reward_func = metrics["corp_reward"]
        self.classifier = metrics["classifier_reward"] 

            
          
        self.train_data = train_data
       
       
        self.eval_data = eval_data
        self.evaluator = lb.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["nmt_loss"]
        self.critic_loss_func = metrics["critic_loss"]  #++++
        
        self.dicts = dicts
      
        self.critic_optim = critic_optim #++++
        self.optim = optim
        self.max_length = opt.max_predict_length
    
        self.opt = opt
        self.sbleu_reward = False #Turnon this flag if want other reward in evaluation 
        print("==")
        print(actor)
        print("==")
        print(critic)

    def train(self, start_epoch, end_epoch, pretrain_critic, start_time=None):  #++++
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss  = None
        self.optim.set_lr(self.opt.reinforce_lr)
        
        hist_valid_scores = []
        num_trial = 0
        #  Use large learning rate for critic during pre-training.
        if pretrain_critic:
            self.critic_optim.set_lr(1e-3)  #++++
        else:
            self.critic_optim.set_lr(self.opt.reinforce_lr)  #++++
        
        for epoch in range(start_epoch, end_epoch + 1):
            print("")
            
            print("* REINFORCE epoch with %s*" % self.reward)
            print("Actor optim lr: %g; Critic optim lr: %g" %
                (self.optim.lr, self.critic_optim.lr))
                
            if pretrain_critic:  #++++
                print("Pretrain critic...")
            no_update = self.opt.no_update and (not pretrain_critic) and \
                        (epoch == start_epoch)

           
        

            if no_update: print("No update...")
            
            
            
            train_reward, critic_loss = self.train_epoch(epoch, pretrain_critic, no_update)
            print("Train sentence reward: %.2f" % (train_reward * 100))
            print("Critic loss: %g" % critic_loss)
 
 
            
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
            
            #valid_metric = valid_ppl 
            #is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)


            self.optim.updateLearningRate(-valid_reward, epoch)
            # Actor and critic use the same lr when jointly trained.
            # TODO: using small lr for critic is better?
            if not pretrain_critic:                         #++++
                self.critic_optim.set_lr(self.optim.lr)     #++++
                
            checkpoint = {                                  #++++     
                "model": self.actor,
                "critic": self.critic,
                "dicts": self.dicts,
                "opt": self.opt,
                "epoch": epoch,
                "optim": self.optim,
                "critic_optim": self.critic_optim
            }
            model_name = os.path.join(self.opt.save_dir, "model_%d" % epoch)


            if pretrain_critic:
                model_name += "_pretrain"
            else:
                model_name += "_reinforce"
            model_name += ".pt"
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)
 

           
      

    def train_epoch(self, epoch, pretrain_critic, no_update): #++++
        self.actor.train()

        total_reward, report_reward = 0, 0
        total_critic_loss, report_critic_loss = 0, 0 #+++
        total_actor_loss, report_actor_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
       
        for i in range(len(self.train_data)):
            batch   = self.train_data[i]
            sources = batch[0]
            targets = batch[1]
            batch_size = targets.size(1)
           
            self.actor.zero_grad()
            self.critic.zero_grad() #++++
            

            # Sample translations
            attention_mask = sources[0].data.eq(lb.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)
            samples, outputs = self.actor.sample(batch, self.max_length)
            
           
            #Calculate rewards  

            reinfore_rewards, bert_reward = self.reward(samples.t().tolist(), targets.data.t().tolist(), sources) #sample : bsxsl
            reinfore_reward = sum(reinfore_rewards)
                       
            #fix sample
            samples = list(map(self.fix_sample, samples.t().tolist()))
            

           
            print("show sample")
            print("Rewarding Target Sample")
            sent = [self.dicts["tgt"].convertToLabels(s, stop=3)[:-1] for s in samples]
            print(sent) 
            print("Original Source Sentence")
            sent = [self.dicts["src"].convertToLabels(s.tolist(), stop =0)[:-1] for s in sources[0].transpose(0,1)]
            print(sent)
            
          
            samples = Variable(torch.LongTensor(samples).t().contiguous()) #slXbs            
            reinfore_rewards = Variable(torch.FloatTensor([reinfore_rewards] * samples.size(0)).contiguous()) #samples.size(0) =SL  
            
             
            #print(sent_rewards.shape)
            if self.opt.cuda:
                #print("*Rbleu*",type(reinfore_rewards))
                samples = samples.to('cuda')
                #print("*Sample*",type(samples))
                reinfore_rewards = reinfore_rewards.to('cuda')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                
                
            # Update  Critic
            critic_weights = samples.ne(lb.Constants.PAD).float()
            num_words = critic_weights.data.sum()
            if not no_update:
                baselines = self.critic((sources, samples), eval=False, regression=True)   #Critic predicted baseline
                if not pretrain_critic:
                    if self.opt.reward_model == 'bert_reward':
                      print("CRITIC UPDATE for epoch %d with Posttrain Loss using %s" %(epoch, self.opt.reward_model))
                      critic_loss = self.critic.backward(
                          baselines, reinfore_rewards, critic_weights, num_words, self.critic_loss_func, regression=True) # Minimize the predicted baseline
                    if self.opt.reward_model == 'harmonic_reward':
                     print("CRITIC UPDATE for epoch %d with Posttrain Loss using %s" %(epoch, self.opt.reward_model))
                     critic_loss = self.critic.backward(
                          baselines, reinfore_rewards, critic_weights, num_words, self.critic_loss_func, regression=True) # Minimize the predicted baseline
                    if self.opt.reward_model == 'sent_reward':
                     print("CRITIC UPDATE for epoch %d with Posttrain Loss using %s" %(epoch, self.opt.reward_model))
                     critic_loss = self.critic.backward(
                          baselines, reinfore_rewards, critic_weights, num_words, self.critic_loss_func, regression=True) # Minimize the predicted baseline                          
                          
                else :
                    print("CRITIC  UPDATE for epoch %d with Pretrain Loss  %s " %(epoch, self.opt.reward_model))                    
                    critic_loss = self.critic.backward(
                       baselines, reinfore_rewards, critic_weights, num_words, self.critic_loss_func, regression=True)
                self.critic_optim.step()
            else:
                critic_loss = 0
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

            
            # Update actor
            if not pretrain_critic and not no_update: #++++
                # Subtract baseline from reward
                norm_rewards = Variable((reinfore_rewards - baselines).data)
                actor_weights = norm_rewards * critic_weights
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                actor_loss = self.actor.backward(outputs, samples, actor_weights, 1, self.actor_loss_func)  #++++
                self.optim.step()
                
  
                #mle training
                #mle_outputs = self.actor(batch, eval=False)
                #actor_weights = targets.ne(lb.Constants.PAD).float()
                #num_words = actor_weights.data.sum()
                #actor_loss = self.actor.backward(mle_outputs, targets, actor_weights, num_words, self.actor_loss_func)
                #self.optim.step()
                
                
            else :
                actor_loss =0
            # Gather stats
 
            report_reward += reinfore_reward
            total_reward += reinfore_reward

            total_sents += batch_size
            report_sents += batch_size

            total_actor_loss += actor_loss
            report_actor_loss += actor_loss

            total_critic_loss += critic_loss #++++
            report_critic_loss += critic_loss
            
            total_words += num_words
            report_words += num_words
          
            print("Epoch %d,%d Reinforce Training using % s  reward" %(epoch, i, self.opt.reward_model))
            
            if i % self.opt.log_interval == 0 and i > 0:
                print("""Epoch %3d, %6d/%d batches;
                      actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data),
                      (report_reward / report_sents) * 100,
                      report_critic_loss / report_words,                            #++++
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_reward = report_sents = report_critic_loss = report_words = 0   #++++

                last_time = time.time()
      
        return total_reward / total_sents, total_critic_loss / total_words #++++
   
   
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

