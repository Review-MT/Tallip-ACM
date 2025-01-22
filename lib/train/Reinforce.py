import datetime
import math
import os
import time
import numpy as np
from torch.autograd import Variable
import torch
import pandas as pd
import lib as lib
import re
class Reinforce(object):

    def __init__(self, actor, train_data, eval_data, metrics, dicts, optim, opt):
        self.actor = actor
        self.reward = opt.reward_model
        rewards = ['bert_reward', 'harmonic_reward']
        if opt.reward_model in rewards:
           print("Reward is %s " % opt.reward_model)
           
           self.reward_func= metrics["classifier_reward"] # #  # instiantiate object of classifier
        elif: 
           
           self.reward = 'sent_reward'
           self.reward_func = metrics["sent_reward"] 
        else :
           #print("Reward %s not valid," % )
                raise RuntimeError("Invalid Reward method: " + opt.reward_model)
            
          
        self.train_data = train_data
       
       
        self.eval_data = eval_data
        self.evaluator = lib.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["nmt_loss"]
        #self.critic_loss_func = metrics["critic_loss"]
        self.sent_reward_func = metrics["sent_reward"]

        self.dicts = dicts

        self.optim = optim
        self.max_length = opt.max_predict_length
    
        self.opt = opt
      
        print("")
        print(actor)
        print("")
       

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
            print(self.optim.lr)
        

            if no_update: print("No update...")
            
            train_reward, actor_loss = self.train_epoch(epoch, no_update)
            #sent_bleu_rw, bert_rw = train_reward
            print("Train sentence reward: %.2f" % (train_reward *100))
            print("Actor loss: %g" % actor_loss)
 
            if no_update: break
           
            valid_loss, valid_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print("Reported Result for %s data set " % data.data)
            print("")
            
            print("Validation perplexity: %.2f" % valid_ppl)
            print("Loss: %.6f" % loss)
            print("Train Reward: %.2f" % valid_reward * 100))
            print("Corpus reward: %.2f" % (corpus_reward * 100))

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


           
      

    def train_epoch(self, epoch, no_update):
        self.actor.train()

        stotal_reward, sreport_reward, mtotal_reward, mreport_reward, htotal_reward, hreport_reward = 0, 0, 0, 0, 0, 0
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
           

            # Sample translations
            attention_mask = sources[0].data.eq(lib.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)
            samples, outputs = self.actor.sample(batch, self.max_length)
        
           
            #Calculate rewards for pre training and optimize BLEU


            sent_rewards, samples = self.reward_func(samples.t().tolist(), targets.data.t().tolist())
            sent_reward = sum(sent_rewards)
            
            samples = Variable(torch.LongTensor(samples).t().contiguous())
           
            # Perturb rewards for pre-training : Reward is from Classifier.
            # TODO : Logic to handel pretraining and post training seperately : Pretraing Rewars only Sent_BLEU : Post Training Reward Classifier Accuracy
                  
            sent_rewards = Variable(torch.FloatTensor([sent_rewards] * samples.size(0)).contiguous())
            #print(sent_rewards.shape)
            if self.opt.cuda:
                print("*Sbleu*",type(sent_rewards))
                samples = samples.to('cuda')
                print("*Sample*",type(samples))
                sent_rewards = sent_rewards.to('cuda')

            # Update  Critic
            critic_weights = samples.ne(lib.Constants.PAD).float()
            num_words = critic_weights.data.sum()
            
            # Update actor
            if not no_update:
                # Subtract baseline from reward
                if self.reward == 'sent_reward':
                   norm_rewards = Variable((sent_rewards).data)
                actor_weights = norm_rewards * critic_weights
                
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                actor_loss = self.actor.backward(outputs, samples, actor_weights, num_words, self.actor_loss_func)
                self.optim.step()
            else :
                actor_loss =0
            # Gather stats
 
            sreport_reward += sent_reward
            stotal_reward += sent_reward

            total_sents += batch_size
            report_sents += batch_size
            total_actor_loss += actor_loss
            report_actor_loss += actor_loss
            total_words += num_words
            report_words += num_words
            if self.reward =='sent_reward':
                  print("Epoch %d,%d Reinforce Post Training using % s  reward" %(epoch, i, self.reward))
                  if i % self.opt.log_interval == 0 and i > 0:  #TODO Add printf for model and harmonic reward
                     print("""Epoch %3d, %6d/%d batches;                                 
                       actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                       (epoch, i, len(self.train_data),
                       (sreport_reward / report_sents) * 100,
                       report_actor_loss / report_words,
                       report_words / (time.time() - last_time),
                       str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                  hreport_reward = mreport_reward = sreport_reward = report_sents = report_actor_loss = report_words = 0
                  last_time = time.time()
      
        return stotal_reward / total_sents, total_actor_loss / total_words
       


