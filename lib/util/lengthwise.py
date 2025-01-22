import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Read file and  divide the based on token length 1-4, 4-10, above 10
def length_split_token(input_file_review, input_file_label):
    with open(input_file_review) as text, open(input_file_label) as lb:
       line = text.readlines()
             


def classification_accuracy(true, pred):
    print("Pass to McNemar's Test")
    print(np.array(true))
    print(np.array(pred))
    assert len(true) == len(pred)
    score = f1_score(true, pred, average='weighted')
    print(confusion_matrix(true, pred, labels=[0,1,2]))
    print("Classification Weighted f1 score on Evaluation")
    #cm=ConfusionMatrix(y_true, y_pred)
    #cm.print_stats()
    #print(score)
    return score


def classifier_statistic():
   with open('/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/Reinforce/HAR_balanced_bs4/re/km/lengthwise/hr.4gen','r') as f1, open('/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/Reinforce/HAR_balanced_bs4/re/km/lengthwise/hr.4pen', 'r') as  f2:
     gold_rewards = f1.readlines()
     all_bert_reward = f2.readlines()
     gold_rewards = [int(gold.strip())   for gold in gold_rewards] 
     all_bert_reward = [int(bert.strip())   for bert in all_bert_reward]
     # STEP FOR CLASSIFIER ACCURACY COMPUTATION
     #gold_rewards = list(map(lb.Reward.reward_conversion, gold_rewards))
     print("Classification Statisctic")
     print(gold_rewards)
     #print(all_bert_reward)
   
     t_gold_rewards = np.array(gold_rewards)
    # t_gold_rewards = t_gold_rewards.argmax(axis=1)
     print("==="*10)
     print(t_gold_rewards)
     print("***"*10)
     print(all_bert_reward)
     #exit(0)
     #mrw_pred, osample= self.classifier("classifier",psents)
     t_mrw_pred = np.array(all_bert_reward)
    # t_mrw_pred = t_mrw_pred.argmax(axis=1)
    # with open('perplx/class/pred_labels.txt', 'w') as f, open('perplx/class/gold_labels.txt', 'w') as ff :
    #  for item1, item2 in zip(t_mrw_pred, t_gold_rewards):
    #     f.write("%s\n" % item1)
    #     ff.write("%s\n" % item2)
                 #f.write(json.dumps(t_mrw_pred.tolist()))


     print(t_mrw_pred)
     
     score = classification_accuracy(t_mrw_pred, t_gold_rewards)

     target_names = ['positive 0', 'neutral 1', 'negative 2']
     print("Classification Weighted F1 score", score)
     #print("Avg Classifier reward : %f" (sum(all_bert_reward)/len(all_bert_reward)))  
     
     print(classification_report( t_gold_rewards,  t_mrw_pred, target_names=target_names))          


classifier_statistic()


def makeDataReinforce(srcFile, tgtFile, glabelFile, plabelFile, gold, pred_file='mle',UNEQUAL=False):
    src, tgt ,labels= [], [], []
    sizes = []
    count, ignored = 0, 0
    count_idx=[]

    print("Processing %s & %s ..." % (srcFile, glabelFile))
    srcF  = open(srcFile)
    tgtF  = open(tgtFile)
    glabel = open(glabelFile)
    plabel = open(plabelFile)
    gold = open(gold)
    with open(str(pred_file)+'.4gold',"w") as f1,open(str(pred_file)+'.4pred',"w") as f2,open(str(pred_file)+'.4gen',"w") as f3,open(str(pred_file)+'.4ref',"w") as gf3,  \
                 open(str(pred_file)+'.5-9gold', "w") as f4, open(str(pred_file)+'.5-9pred', "w") as f5, open(str(pred_file)+'.5-9gen', "w") as f6, open(str(pred_file)+'.5-9ref', "w") as gf6, \
                    open(str(pred_file)+'.10gold', "w") as f7, open(str(pred_file)+'.10pred', "w") as f8, open(str(pred_file)+'.10gen', "w") as f9, open(str(pred_file)+'.10ref', "w") as gf9,\
                       open(str(pred_file)+'.4pen', "w") as pf3, open(str(pred_file)+'.5-9pen', "w") as pf6, open(str(pred_file)+'.10pen', "w") as pf9 :
       while True:
           srcWords = srcF.readline().split()
           tgtWords = tgtF.readline().split()
           goldWords = gold.readline().split() 
           glb    = glabel.readline().strip()
           plb    = plabel.readline().strip()
           print(srcWords,tgtWords, glb, goldWords)   
           if not srcWords or not tgtWords or not glb:
               if srcWords and not tgtWords or not srcWords and tgtWords or not glb and srcWords and tgtWords :
                   print("WARNING: source and target and label do not have the same number")
                   print(len(srcWords),len(tgtWords),len(goldWords))
                   UNEQUAL = True
                   continue   
               break
                   
           if len(goldWords) <=4 :
           
               print(' '.join(srcWords), file=f1)
               print(' '.join(tgtWords), file=f2)
               print(glb , file=f3)
               print(plb , file=pf3)
               print(' '.join(goldWords), file=gf3)
           elif 4 <  len(goldWords)  <=10 :
               print(' '.join(srcWords), file=f4)
               print(' '.join(tgtWords), file=f5)
               print(glb , file=f6)
               print(plb , file=pf6)
               print(' '.join(goldWords), file=gf6)
           elif len(goldWords) > 10:
               print(' '.join(srcWords), file=f7)
               print(' '.join(tgtWords), file=f8)
               print(glb , file=f9)
               print(plb , file=pf9)
               print(' '.join(goldWords), file=gf9)



               src += [' '.join(srcWords)]
               tgt += [' '.join(tgtWords)]
               labels +=[glb]
               count_idx.append(count)
               
               sizes += [len(srcWords)]
           
            
           else:
               ignored += 1
       
           count += 1
           #if count % opt.report_every == 0:
           #    print("... %d sentences prepared" % count)
    

   # labels_is=list(map(label_is.__getitem__, count_idx))
    srcF.close()
    tgtF.close()
    glabel.close()
    plabel.close()
    gold.close()
    return UNEQUAL


#10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test and flip_km
#srcFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test/src'
#tgtFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test/pred'
#tgtGFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test/ref'
#glabelFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test/gold_labels.txt'
#plabelFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_MO_split2/result/flip_test/pred_labels.txt'

#10_aug/tebbefekr/IITB/GNMT_HAR_split2
#srcFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_HAR_split2/result/flip_test/src'
#tgtFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_HAR_split2/result/flip_test/pred'
#tgtGFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_HAR_split2/result/flip_test/ref'
#glabelFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_HAR_split2/result/flip_test/gold_labels.txt'
#plabelFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/IITB/GNMT_HAR_split2/result/flip_test/pred_labels.txt'

#10_aug/tebbefekr/IITB/GNMT_HAR_split2/
#srcFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/all_flip/HARMONIC_10246/flip_km_eph5/src'
#tgtFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/all_flip/HARMONIC_10246/flip_km_eph5/pred'
#tgtGFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/all_flip/HARMONIC_10246/flip_km_eph5/ref'
#glabelFile ='/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/all_flip/HARMONIC_10246/flip_km_eph5/gold_labels.txt'
#plabelFile = '/Data/divya/exp/experiments/Reinforce/10_aug/tebbefekr/all_flip/HARMONIC_10246/flip_km_eph5/pred_labels.txt'

srcFile ='/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/re/flip/src'
tgtFile ='/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/re/flip/pred'
tgtGFile = '/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/re/flip/ref'
glabelFile ='/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/re/flip/gold_labels.txt'
plabelFile = '/Data/divya/exp/experiments/Reinforce/flip_5oct/generic_long_do0/re/flip/pred_labels.txt'
#UNEQUAL = makeDataReinforce(srcFile, tgtFile, glabelFile, plabelFile, tgtGFile)
#print(UNEQUAL)
#if UNEQUAL:
# print("WARNING: source and target and label do not have the same number")

