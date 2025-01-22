import datetime
import math
import os
import time
import numpy as np
from torch.autograd import Variable
import torch

import lib as lib
import re
class ReinforceTrainer(object):

    def __init__(self, actor, critic, train_data, eval_data, metrics, dicts, optim, critic_optim, opt, s_classifier = None):
        self.actor = actor
        self.critic = critic
        print("Inside REINFORCE")
        if s_classifier:
           print("Classifier will give Reward")
           self.s_classifier= s_classifier("eclassifier") # instiantiate object of classifier
        else: self.s_classifier = None
        self.train_data = train_data
       
       
        self.eval_data = eval_data
        self.evaluator = lib.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["nmt_loss"]
        self.critic_loss_func = metrics["critic_loss"]
        self.sent_reward_func = metrics["sent_reward"]

        self.dicts = dicts

        self.optim = optim
        self.critic_optim = critic_optim

        self.max_length = opt.max_predict_length
        self.pert_func = opt.pert_func
        self.opt = opt
      
        print("")
        print(actor)
        print("")
        print(critic)

    def train(self, start_epoch, end_epoch, pretrain_critic, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss = self.critic_optim.last_loss = None
        self.optim.set_lr(self.opt.reinforce_lr)

        #  Use large learning rate for critic during pre-training.
        if pretrain_critic:
            self.critic_optim.set_lr(1e-3)
        else:
            self.critic_optim.set_lr(self.opt.reinforce_lr)

        for epoch in range(start_epoch, end_epoch + 1):
            print("")

            print("* REINFORCE epoch *")
            print("Actor optim lr: %g; Critic optim lr: %g" %
                (self.optim.lr, self.critic_optim.lr))
            if pretrain_critic:
                print("Pretrain critic...")
            no_update = self.opt.no_update and (not pretrain_critic) and \
                        (epoch == start_epoch)

            if no_update: print("No update...")

            train_reward, critic_loss = self.train_epoch(epoch, pretrain_critic, no_update)
            print("Train sentence reward: %.2f" % (train_reward * 100))
            print("Critic loss: %g" % critic_loss)
            
            if self.s_classifier and not pretrain_critic : #TODO Validation reward ans Training teward repoting same
                pred_file='sampling_RLepoch_'+str(epoch)+'.csv'
                valid_loss, valid_sent_reward, valid_corpus_reward, classifier_reward = self.evaluator.eval(self.eval_data, pred_file, self.s_classifier)
                valid_ppl = math.exp(min(valid_loss, 100))
                print("Validation perplexity: %.2f" % valid_ppl)
                print("Validation sentence reward: %.2f" % (valid_sent_reward * 100))
                print("Validation corpus reward: %.2f" %(valid_corpus_reward * 100))
                print("Classifier Accuracy reward",classifier_reward[0].tolist() )
           
            else:
                valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)  # TODO
            
                valid_ppl = math.exp(min(valid_loss, 100))
                print("Validation perplexity: %.2f" % valid_ppl)
                print("Validation sentence reward: %.2f" % (valid_sent_reward * 100))
                print("Validation corpus reward: %.2f" %(valid_corpus_reward * 100))
 
            if no_update: break

            #self.optim.updateLearningRate(-valid_sent_reward, epoch)
            if epoch >= 16 :
               self.optim.updateLearningRate(-classifier_reward[0].tolist(), epoch) # TODO Dont update learning rate for first 5 epoch of posttraining
            #else:
            #   self.optim.updateLearningRate(-valid_sent_reward, epoch)
            # Actor and critic use the same lr when jointly trained.
            # TODO: using small lr for critic is better?
            if not pretrain_critic and epoch >=16:
                self.critic_optim.set_lr(self.optim.lr)
            print(self.optim)
            
            checkpoint = {
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


    def _convert_and_report(self, data, pred_file, preds, original, gold_reward, epoch):   # Shape pred(SL * BS), original(SL * BS), Reward :BS
       # preds = data.restore_pos(preds)
        with open(pred_file+ str(epoch) +'.csv', "a") as f :
           
            combine_sen_rw = list(map(list,zip(*(original.t().tolist(), gold_reward)))) # original (BS * SL)
            
            #transpose = np.array(combine_sen_rw).T.tolist()
            #print(transpose)
            #print(preds.t().tolist())
            sents , reals, scores=  [],[],[]     
           
            for sent, orig in zip(preds.t().tolist(), combine_sen_rw):
               
                len_pred=len(sent) 
                sent = lib.Reward.clean_up_sentence(sent, remove_unk=False, remove=True)
                real = lib.Reward.clean_up_sentence(orig[0], remove_unk=False, remove=True)
              
                
                sent = [self.dicts["tgt"].getLabel(w) for w in sent]
                real = [self.dicts["src"].getLabel(w) for w in real]
               
               
                real=" ".join(real)
                
                
                sent=" ".join(sent)
              
             
                #real=re.sub(r"@@ ",""," ".join(real))
                #sent=re.sub(r"@@ ",""," ".join(sent))
                print(sent + "\t" +real + "\t" + str(orig[1]) , file=f)
              
                sents.append(sent)
                reals.append(real)
                scores.append(orig[1])
            
            return sents, (reals, scores)
           
      

    def train_epoch(self, epoch, pretrain_critic, no_update):
        self.actor.train()

        stotal_reward, sreport_reward, mtotal_reward, mreport_reward, htotal_reward, hreport_reward = 0, 0, 0, 0, 0, 0
        total_critic_loss, report_critic_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
       
        for i in range(len(self.train_data)):
            batch   = self.train_data[i]
            sources = batch[0]
            targets = batch[1]
            batch_size = targets.size(1)
           
            self.actor.zero_grad()
            self.critic.zero_grad()

            # Sample translations
            attention_mask = sources[0].data.eq(lib.Constants.PAD).t()
            self.actor.decoder.attn.applyMask(attention_mask)
            samples, outputs = self.actor.sample(batch, self.max_length)
        
           
            #Calculate rewards for pre training and optimize BLEU
            sent_rewards, samples = self.sent_reward_func(samples.t().tolist(), targets.data.t().tolist())
            sent_reward = sum(sent_rewards)
            
            samples = Variable(torch.LongTensor(samples).t().contiguous())
           
            # Perturb rewards for bandid training : Reward is from Classifier.
            # TODO : Logic to handel pretraining and post training seperately : Pretraing Rewars only Sent_BLEU : Post Training Reward Classifier Accuracy
            if self.s_classifier and not pretrain_critic : 
                pred_file='RL_posttrain_sampling'
                pred, real = self._convert_and_report(self.train_data, pred_file, samples ,sources[0],sources[2],epoch)
            
                model_rewards,osample  = self.s_classifier(pred)  # function call
        
                print("Classifier Prediction passed")
                print(model_rewards)
                print("Gold Reward")
                print(real[1])
                model_rewards = lib.metric.msent_reward_func(model_rewards, real[1], batch_size)
                print(len(model_rewards))
                
        
                print("**Final Classifier Reward ",model_rewards)
                print("**Final Sentence Bleu Reward",sent_rewards)
                
                print(np.array(model_rewards),len(model_rewards))
                print(np.array(sent_rewards),len(sent_rewards)) 
                   
                #harmonic_rewards = (2/(1.0/(1e-6 + np.array(model_rewards))+1.0/(1e-6+ np.array(sent_rewards))))
               
                #print("$$$$$")
                #print(harmonic_rewards,len(harmonic_rewards))
             
                model_reward = sum(model_rewards)
                model_rewards=tuple(model_rewards) 
                #harmonic_reward = sum(harmonic_rewards)
                #harmonic_rewards = Variable(torch.FloatTensor([harmonic_rewards]* samples.size(0)).contiguous())
                model_rewards = Variable(torch.FloatTensor([model_rewards] * samples.size(0)).contiguous())
                
            sent_rewards = Variable(torch.FloatTensor([sent_rewards] * samples.size(0)).contiguous())
            #print(sent_rewards.shape)
            if self.opt.cuda:
                samples = samples.cuda()
                #sent_rewards = sent_rewards.cuda()
                if not pretrain_critic:
                   model_rewards = model_rewards.cuda()
                   #harmonic_rewards = harmonic_rewards.cuda()
                  # print("Reward Shape in POST PRETRAIN", model_rewards.shape, sent_rewards.shape)


            # Update  Critic
            critic_weights = samples.ne(lib.Constants.PAD).float()
            num_words = critic_weights.data.sum()
            if not no_update:
                baselines = self.critic((sources, samples), eval=False, regression=True)   #Critic predicted baseline
                if not pretrain_critic:
                    print("CRITIC UPDATE for epoch %d with Posttrain Loss" % epoch)
                    critic_loss = self.critic.backward(
                       baselines, model_rewards, critic_weights, num_words, self.critic_loss_func, regression=True) # Minimize the predicted baseline
                else :
                    print("CRITIC UPDATE for epoch %d with Pretrain Loss" % epoch)                    
                    critic_loss = self.critic.backward(
                       baselines, sent_rewards, critic_weights, num_words, self.critic_loss_func, regression=True)
                self.critic_optim.step()
            else:
                critic_loss = 0

            # Update actor
            if not pretrain_critic and not no_update:
                # Subtract baseline from reward
                norm_rewards = Variable((model_rewards - baselines).data)
                actor_weights = norm_rewards * critic_weights
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                actor_loss = self.actor.backward(outputs, samples, actor_weights, 1, self.actor_loss_func)
                self.optim.step()
                torch.cuda.empty_cache()

            # Gather stats
            
            if not pretrain_critic :
               mtotal_reward += model_reward
               mreport_reward += model_reward
               #htotal_reward += harmonic_reward
               #hreport_reward += harmonic_reward
            sreport_reward += sent_reward
            stotal_reward += sent_reward

            total_sents += batch_size
            report_sents += batch_size
            total_critic_loss += critic_loss
            report_critic_loss += critic_loss
            total_words += num_words
            report_words += num_words
            if i % self.opt.log_interval == 0 and i > 0:  #TODO Add printf for model and harmonic reward
                print("""Epoch %3d, %6d/%d batches;                                 
                      actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data),
                      (mreport_reward / report_sents) * 100,
                      report_critic_loss / report_words,
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                hreport_reward = mreport_reward = sreport_reward = report_sents = report_critic_loss = report_words = 0
                last_time = time.time()
        if not pretrain_critic :
            print("Epoch %d Reinforce Post Training using Only Classifier reward" % epoch)
            return mtotal_reward / total_sents, total_critic_loss / total_words
        
        print("EPOCH : %d is Reinforce Pre Training using Sentence BLUE rewards" % epoch)
        return stotal_reward / total_sents, total_critic_loss / total_words




