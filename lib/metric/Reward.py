import lib as lb
import numpy as np
import torch
from torch.autograd import Variable 
from sklearn.metrics import f1_score
def reward_conversion(reward):
    if reward == 1: return [1.,0.] #tpos
    if reward == 2: return [0.,1.] #tneu
    
def multiply_two(pair):
     source, machine = pair
     return np.sum(np.multiply(source,machine)), source

def _harmonic_func(pair):
      sent,model = pair
      
      
      harmonic = 2/(1.0/(1e-6 + model)+1.0/(1e-6+ sent))
      print("(*)"*10)
      print(harmonic)
      return harmonic, sent

def harmonic(sent,model):
    pair = map(_harmonic_func, zip(sent,model))
    harmonic , source = zip(*pair)
    return harmonic 
def msent_reward_func(machine_rewards, source_reward, mode_convert = False): #TO_DO batch_size is dummy can remove it
      print("Using Classifier as Reward")
      print(np.shape(machine_rewards),np.shape( source_reward))
      if not mode_convert :
        print("Converting Reward to vector as Reward is in raw form 1,2,3")
        source_reward=list(map(reward_conversion, source_reward))
      print("Machine",machine_rewards)
      print("Source",source_reward)
      print(np.shape(source_reward))
     
      #source_reward   = Variable(torch.FloatTensor(source_reward))
      #machine_rewards = Variable(torch.FloatTensor(machine_rewards))
      print("Machine",np.shape(machine_rewards))
      print("Source",np.shape(source_reward))
      #print(len(source_reward))
      #exit(0)
      #source_reward    = source_reward.view(batchsize,-1,3)
      reward = map(multiply_two, zip(source_reward, machine_rewards))
      #reward = [np.sum(np.multiply(x,y)) for x,y in zip(source_reward, machine_rewards)]
   
      #result           = source_reward * machine_rewards 
      #print(source_reward)
      #print(machine_rewards)
     
      reward , source = zip(*reward)
      #reward= torch.sum(result, dim=-1)
      #print(reward)
      #(r   for r in reward:
      return reward
      




def clean_up_sentence(sent, remove_unk=False, remove=False):

    if lb.Constants.EOS in sent:
        sent = sent[:sent.index(lb.Constants.EOS) + 1]
    if lb.Constants.PAD in sent:
        sent = sent[:sent.index(lb.Constants.PAD) + 1]
    
    if remove_unk:
        sent = filter(lambda x: x != lb.Constants.UNK, sent)
    if remove:
        if len(sent) > 0 and (sent[-1] == lb.Constants.EOS or sent[-1] == lb.Constants.PAD) : 
            sent = sent[:-1]
    return sent

def single_sentence_bleu(pair):
    length =  len(pair[0]) 
    pred, gold = pair
    pred = clean_up_sentence(pred, remove_unk=False, remove=False)
    gold = clean_up_sentence(gold, remove_unk=False, remove=False)
    len_pred =  len(pred)
    if len_pred == 0:
        score = 0.
        pred = [lb.Constants.PAD] * length
    else:
        score = lb.Bleu.score_sentence(pred, gold, 4, smooth=1)[-1]
        while len(pred) < length:
            pred.append(lb.Constants.PAD)

          

    return score, pred

def sentence_bleu(preds, golds):
    #print("Using BLEU as Reward")
    results = map(single_sentence_bleu, zip(preds, golds))
    scores, preds = zip(*results)
    return scores, preds

def corpus_bleu(preds, golds):
    assert len(preds) == len(golds)
    clean_preds = []
    clean_golds = []
    for pred, gold in zip(preds, golds):
        pred = clean_up_sentence(pred, remove_unk=False, remove=True)
        gold = clean_up_sentence(gold, remove_unk=False, remove=True)
        clean_preds.append(pred)
        clean_golds.append(gold)
    return lb.Bleu.score_corpus(clean_preds, clean_golds, 4)

from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix




def classification_accuracy(true, pred):
    print("Pass to McNemar's Test")
    print(np.array(true))
    print(np.array(pred))
    assert len(true) == len(pred)
    score = f1_score(true, pred, average='weighted')
    print(confusion_matrix(true, pred, labels=[0,1]))
    print("Classification Weighted f1 score on Evaluation")
    #cm=ConfusionMatrix(y_true, y_pred)
    #cm.print_stats()
    #print(score)

    return(score)


    
    
