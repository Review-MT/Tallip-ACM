import argparse
import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

import lib as lb
#import bertmaster 

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '7' #use GPU with ID=[0..9]# for gpu alloc


parser = argparse.ArgumentParser(description="Reinforce_generic.py")

## Data options
parser.add_argument("-data", required=True,
                    help="Path to the *-train.pt file from preprocess.py")
parser.add_argument("-save_dir", required=True,
                    help="Directory to save models")
parser.add_argument("-load_from", help="Path to load a pretrained model.")

## Model options

parser.add_argument("-layers", type=int, default=1,
                    help="Number of layers in the LSTM encoder/decoder")
parser.add_argument("-rnn_size", type=int, default=512,
                    help="Size of LSTM hidden states")
parser.add_argument("-word_vec_size", type=int, default=256,
                    help="Size of word embeddings")
parser.add_argument("-input_feed", type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument("-brnn", action="store_true",
                    help="Use a bidirectional encoder")
parser.add_argument("-brnn_merge", default="concat",
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument("-batch_size", type=int, default=32,
                    help="Maximum batch size")
parser.add_argument("-max_generator_batches", type=int, default=32,
                    help="""Split softmax input into small batches for memory efficiency.
                    Higher is faster, but uses more memory.""")
parser.add_argument("-end_epoch", type=int, default=50,
                    help="Epoch to stop training.")
parser.add_argument("-start_epoch", type=int, default=1,
                    help="Epoch to start training.")
parser.add_argument("-param_init", type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument("-optim", default="adam",
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument("-lr", type=float, default=1e-3,
                    help="Initial learning rate")
parser.add_argument("-max_grad_norm", type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("-dropout", type=float, default=0,
                    help="Dropout probability; applied between LSTM stacks.")
parser.add_argument("-learning_rate_decay", type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument("-start_decay_at", type=int, default=5,
                    help="Start decay after this epoch")

# GPU
parser.add_argument("-gpus", default=[0], nargs="+", type=int,
                    help="Use CUDA")
parser.add_argument("-log_interval", type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument("-seed", type=int, default=3321,
                     help="Seed for random initialization")

# Critic
parser.add_argument("-start_reinforce", type=int, default=None,
                    help="""Epoch to start reinforcement training.
                    Use -1 to start immediately.""")
parser.add_argument("-critic_pretrain_epochs", type=int, default=0,
                    help="Number of epochs to pretrain critic (actor fixed).")
parser.add_argument("-reinforce_lr", type=float, default=1e-4,
                    help="""Learning rate for reinforcement training.""")

# Evaluation
parser.add_argument("-eval", action="store_true", help="Evaluate model only")
parser.add_argument("-eval_sample", action="store_true", default=False,
        help="Eval by sampling")
parser.add_argument("-max_predict_length", type=int, default=90,
                    help="Maximum length of predictions.")


# Reward shaping
parser.add_argument("-pert_func", type=str, default=None,
        help="Reward-shaping function.")
parser.add_argument("-pert_param", type=float, default=None,
        help="Reward-shaping parameter.")


#Reward Type
parser.add_argument("-reward_model",  required=True,
        help="specify on reward type : [sent_reward, harmonic_reward, bert_reward]")

# Others
parser.add_argument("-no_update", action="store_true", default=False,
        help="No update round. Use to evaluate model samples.")
parser.add_argument("-sup_train_on_reinfore", action="store_true", default=False,
        help="Supervised learning update round.")

parser.add_argument("-sample",  default=5,
        help="set for multinomial sampling")

opt = parser.parse_args()
print(opt)

# Set seed
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

opt.cuda = len(opt.gpus)

if opt.save_dir and not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 1")

if opt.cuda:
    cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

def init(model):
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

def create_optim(model):
    optim = lb.Optim(
        model.parameters(), opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay, start_decay_at=opt.start_decay_at
    )
    return optim

def create_model(model_class, dicts, gen_out_size):
    encoder = lb.Encoder(opt, dicts["src"])
    decoder = lb.Decoder(opt, dicts["tgt"])
    # Use memory efficient generator when output size is large and
    # max_generator_batches is smaller than batch_size.
    if opt.max_generator_batches < opt.batch_size and gen_out_size > 1:
        generator = lb.MemEfficientGenerator(nn.Linear(opt.rnn_size, gen_out_size), opt)
    else:
        generator = lb.BaseGenerator(nn.Linear(opt.rnn_size, gen_out_size), opt)
    model = model_class(encoder, decoder, generator, opt)
    init(model)
    optim = create_optim(model)
    return model, optim





def main():

    print('Loading data from "%s"' % opt.data)

    dataset = torch.load(opt.data)

    supervised_data = lb.Dataset(dataset["train_iitb"], opt.batch_size, opt.cuda, eval=False)
    #valid_data = lb.Dataset(dataset["valid_pre"], opt.batch_size, opt.cuda, name ="valid_pre", eval=True)
    valid_data = lb.Dataset(dataset["valid_post"], opt.batch_size, opt.cuda, name ="valid_post", gold= True, eval=True)
    reinfore_data = lb.Dataset(dataset["train_pg"], opt.batch_size, opt.cuda, name ="train_pg", gold = True, eval=False)
    #test_data  = lb.Dataset(dataset["test"], opt.batch_size, opt.cuda, name="test", gold= True, eval=True)

    dicts = dataset["dicts"]
    print(" * vocabulary size. source = %d; target = %d" %
          (dicts["src"].size(), dicts["tgt"].size()))
    print(" * number of XENT training sentences. %d" %
          len(dataset["train_iitb"]["src"]))
    print(" * number of PG training sentences. %d" %
          len(dataset["train_pg"]["src"]))
    print(" * maximum batch size. %d" % opt.batch_size)
    print("Building model...")
    print("@@",opt.load_from,opt.data)
    use_reinforce = opt.start_reinforce is not None

    if opt.load_from is None:
        model, optim = create_model(lb.NMTModel, dicts, dicts["tgt"].size())
        checkpoint = None
    else:
        print("Loading from checkpoint at %s" % opt.load_from)
        checkpoint = torch.load(opt.load_from)
        model = checkpoint["model"]
        optim = checkpoint["optim"]
        opt.start_epoch = checkpoint["epoch"] + 1

    # GPU.
    if opt.cuda:
        device = torch.device("cuda")     
        #model.cuda(opt.gpus[0])
        model = model.to(device)

    # Start reinforce training immediately.
    if opt.start_reinforce == -1:
        opt.start_decay_at = opt.start_epoch
        opt.start_reinforce = opt.start_epoch


    nParams = sum([p.nelement() for p in model.parameters()])
    print("* number of parameters: %d" % nParams)

    # Metrics.
    metrics = {}
    metrics["nmt_loss"] = lb.Loss.weighted_xent_loss
    metrics["critic_loss"] = lb.Loss.weighted_mse
    metrics["sent_reward"] = lb.Reward.sentence_bleu
    metrics["corp_reward"] = lb.Reward.corpus_bleu
    metrics["classifier_reward"] = lb.classifier.sentiment_model
    metrics["classifier_max_reward"]  = lb.classifier.sentiment_model_max 

    
    if opt.pert_func is not None:
        opt.pert_func = lb.PertFunction(opt.pert_func, opt.pert_param)

   
    
    # Evaluate model on heldout dataset.
    if opt.eval:
        evaluator = lb.Evaluator(model, metrics, dicts, opt)
        # On validation set.
        pred_file = opt.load_from.replace(".pt", ".valid")
        evaluator.eval(valid_data, pred_file)
        # On test set.
        #pred_file = opt.load_from.replace(".pt", ".test")
        #evaluator.eval(test_data, pred_file)
    elif opt.eval_sample:
        opt.no_update = True
        
        
        reinforce_trainer = lb.Reinforce_flip(model, reinfore_data, test_data,
            metrics, dicts, optim , opt)
        reinforce_trainer.train(opt.start_epoch, opt.start_epoch, False)
    elif opt.sup_train_on_reinfore:
        optim.set_lr(opt.reinforce_lr)
        #xent_trainer = lb.Trainer(model, reinfore_data, test_data, metrics, dicts, optim, opt)
        #xent_trainer.train(opt.start_epoch, opt.start_epoch)
        xent_trainer = lb.Trainer(model, reinfore_data, valid_data, metrics, dicts, optim, opt)
        xent_trainer.train(opt.start_epoch, opt.end_epoch)
    else:
        xent_trainer = lb.Trainer(model, supervised_data, valid_data, metrics, dicts, optim, opt)
        if use_reinforce:
           
            start_time = time.time()
            # Supervised training.
            xent_trainer.train(opt.start_epoch, opt.start_reinforce - 1, start_time)
     
            # Reinforce training
            reinforce_trainer = lb.Reinforce_flip(model, reinfore_data, valid_data,
                    metrics, dicts, optim, opt)
           
            reinforce_trainer.train(opt.start_reinforce, opt.end_epoch, start_time)
        # Supervised training only.
        else:
            xent_trainer.train(opt.start_epoch, opt.end_epoch)


if __name__ == "__main__":
    main()
