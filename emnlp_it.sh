export CUDA_VISIBLE_DEVICES=1
#python actor_critic_generic.py  -data itallian_task/itallian-mo-train.pt -save_dir 3000/it-en   -end_epoch 50    -batch_size 64  -reward_model sent_reward  -start_decay_at 8  -brnn  -dropout 0.2  -max_predict_length 50 

#python actor_critic_generic.py  -data itallian_task/itallian-mo-train.pt -save_dir 3000/it-en  -load_from 3000/it-en/model_5.pt  -end_epoch 50    -batch_size 64  -reward_model sent_reward  -start_decay_at 8  -brnn  -dropout 0.2  -max_predict_length 50

#python  Reinforce_generic.py -max_predict_length 50  -seed 3000 -data data/emnlp-200-50-train.pt  -save_dir 3000/mle+rl/100/RHAR  -load_from 3000/model_67.pt  -start_reinforce -1  -end_epoch 88  -batch_size 4  -reward_model harmonic_reward
#python  Reinforce_generic.py -max_predict_length 50  -seed 3000 -data data/emnlp-200-50-train.pt -reinforce_lr 1e-5   -save_dir 3000/lr05/rl/100/RHAR  -load_from 3000/model_67.pt  -start_reinforce -1  -end_epoch 88  -batch_size 4  -reward_model harmonic_reward

#python  Reinforce_generic.py -max_predict_length 50  -seed 3000 -data data/emnlp-200-50-train.pt  -save_dir 3000/mle+rl/100/RBLEU  -load_from 3000/model_67.pt  -start_reinforce -1  -end_epoch 88  -batch_size 4  -reward_model sent_reward
#python  Reinforce_generic.py -max_predict_length 50  -seed 3000 -data data/emnlp-200-50-train.pt -reinforce_lr 1e-5  -save_dir 3000/lr05/rl/100/RBLEU  -load_from 3000/model_67.pt  -start_reinforce -1  -end_epoch 88  -batch_size 4  -reward_model sent_reward

#python  Reinforce_generic.py -max_predict_length 50  -seed 3000 -data data/emnlp-200-50-train.pt  -save_dir 3000/mle+rl/100/RBERT  -load_from 3000/model_67.pt  -start_reinforce -1  -end_epoch 88  -batch_size 4  -reward_model bert_reward
#python  Reinforce_generic.py -max_predict_length 20  -seed 3000 -data itallian-mo-train.pt   -save_dir 3000/it-en/lr04/rl/100/RBERT  -load_from ../3000/it-en/model_35.pt  -start_reinforce -1  -end_epoch 65  -batch_size 3  -reward_model bert_reward


#pretarin
#python  actor_critic_generic.py -data itallian-mo-train.pt -max_predict_length 30 -seed 3000  -load_from ../3000/it-en/model_35.pt  -save_dir 3000/it-en/pretrain -start_reinforce -1  -end_epoch 36 -critic_pretrain_epochs 1 -batch_size 64  -reward_model sent_reward  -brnn  -dropout 0.2

#AC Harmonic Model MLE+RL and RL 
#python actor_critic_generic.py -data  data/emnlp-200-50-train.pt  -max_predict_length 50  -seed 3000 -reinforce_lr 1e-5 -load_from 3000/pretrain/model_68_pretrain.pt  -save_dir  3000/lr05/rl/100/ACHAR -start_reinforce -1  -end_epoch 89  -batch_size 4  -reward_model harmonic_reward -critic_pretrain_epochs 0 -dropout 0.2
#python actor_critic_generic.py -data  data/emnlp-200-50-train.pt  -max_predict_length 50  -seed 3000 -reinforce_lr 1e-5 -load_from 3000/pretrain/model_68_pretrain.pt  -save_dir  3000/lr05/mle+rl/100/ACHAR1 -start_reinforce -1  -end_epoch 89  -batch_size 4  -reward_model harmonic_reward -critic_pretrain_epochs 0 -dropout 0.2


# AC BLEU Model MLE+RL and RL
#python actor_critic_generic.py -data  data/emnlp-200-50-train.pt  -max_predict_length 50  -seed 3000 -reinforce_lr 1e-5 -load_from 3000/pretrain/model_68_pretrain.pt  -save_dir  3000/lr05/rl/100/ACBLEU -start_reinforce -1  -end_epoch 89  -batch_size 4  -reward_model sent_reward -critic_pretrain_epochs 0 -dropout 0.2
#python actor_critic_generic.py -data  data/emnlp-200-50-train.pt  -max_predict_length 50  -seed 3000 -reinforce_lr 1e-5 -load_from 3000/pretrain/model_68_pretrain.pt  -save_dir  3000/lr05/mle+rl/100/ACBLEU -start_reinforce -1  -end_epoch 89  -batch_size 4  -reward_model sent_reward -critic_pretrain_epochs 0 -dropout 0.2


# AC BLEU Model MLE+RL and RL
#python actor_critic_generic.py -data  data/emnlp-200-50-train.pt  -max_predict_length 50 -reinforce_lr 1e-5 -seed 3000 -load_from 3000/pretrain/model_68_pretrain.pt  -save_dir  3000/lr05/rl/100/ACBERT -start_reinforce -1  -end_epoch 89  -batch_size 4  -reward_model bert_reward -critic_pretrain_epochs 0 -dropout 0.2
python actor_critic_generic.py -data  itallian-mo-train.pt  -max_predict_length 20  -seed 3000 -load_from  3000/it-en/pretrain/model_36_pretrain.pt  -save_dir  3000/it-en/lr04/100/ACBERT -start_reinforce -1  -end_epoch 56  -batch_size 3  -reward_model bert_reward -critic_pretrain_epochs 0 -dropout 0.2


#python MO_bs.py -data  itallian-mo-train.pt  -max_predict_length 30 -load_from ../3000/it-en/model_35.pt  -save_dir ../3000/it-en/lr04/rl/100/MO  -start_reinforce -1  -end_epoch 65  -batch_size 3  -reward_model bert_reward -seed 3000 -dropout 0.2 

