import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np
import lib as lb
import re
#extra
##==============================
from collections import namedtuple
from typing import List, Tuple, Set, Union
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
##===============================
class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lb.Constants.PAD)
        self.rnn = nn.LSTM(opt.word_vec_size, self.hidden_size, 
            num_layers=opt.layers, dropout=opt.dropout, bidirectional=opt.brnn)

    def forward(self, inputs, hidden=None):
        """
        print(inputs[0].shape)
        print(inputs[1])
        print(hidden_t[0].shape,hidden_t[1].shape, outputs.shape)
        
        torch.Size([38, 32])
        
        (38, 25, 24, 20, 19, 16, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 8, 7, 6, 6, 6, 5, 5, 5, 4, 3, 3, 3, 3, 2, 2)
        torch.Size([1, 32, 256]) torch.Size([1, 32, 256]) torch.Size([38, 32, 256])
        
        
        """

        #print("**"*10)

     
        emb = pack(self.word_lut(inputs[0]), inputs[1])
        outputs, hidden_t = self.rnn(emb, hidden)
        outputs = unpack(outputs)[0]
        
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, inputs, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inputs, (h_0[i], c_0[i]))
            inputs = h_1_i
            if i != self.num_layers:
                inputs = self.dropout(inputs)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return inputs, (h_1, c_1)


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lb.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = lb.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size
        self.dict = dicts

    def step(self, emb, output, hidden, context):#context #batch x sourceL x dim

        """
        print("STEP SHAPE")
        print(emb.shape, output.shape, hidden[0].shape, hidden[1].shape,context.shape)
        print(output.shape) 
        torch.Size([32, 128]) torch.Size([32, 256]) torch.Size([1, 32, 256]) torch.Size([1, 32, 256]) torch.Size([32, 38, 256])
        torch.Size([32, 256])
        """
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output, hidden = self.rnn(emb, hidden)
        output, attn = self.attn(output, context)
        output = self.dropout(output)

        return output, hidden
    #NEw fun experiment    
    def steps(self, emb, output, hidden, context):#context #batch x sourceL x dim
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output, (h_t, cell_t)  = self.rnn(emb, hidden)
        output, attn = self.attn(output, context)
        output = self.dropout(output)
        return (h_t, cell_t ), output 
 
    def stepp(self, emb, output, hidden, context):#context #batch x sourceL x dim

        """
        print("STEP SHAPE")
        print(emb.shape, output.shape, hidden[0].shape, hidden[1].shape,context.shape)
        print(output.shape) 
        torch.Size([32, 128]) torch.Size([32, 256]) torch.Size([1, 32, 256]) torch.Size([1, 32, 256]) torch.Size([32, 38, 256])
        torch.Size([32, 256])
        """
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output, (h_t, cell_t) = self.rnn(emb, hidden)
        print("%"*5)
        
        output, attn = self.attn(output, context)
        print(output.shape, context.shape)
        output = self.dropout(output)

        return  (h_t, cell_t), output     

    def forward(self, inputs, init_states):
        emb, output, hidden, context = init_states  #context #BS*SL #inputs to decoder is target sentence
        embs = self.word_lut(inputs)

        outputs = []
        for i in range(inputs.size(0)): #unless taget sentence do not end : shape is SL*BS*dim
            output, hidden = self.step(emb, output, hidden, context) #emb of first <y> as decoder input, which is <bos>, output is 
                                                                     #first decoder output, initially set to zero for t0 time step as no decoder output is avalibale initially 
            outputs.append(output)
            emb = embs[i]

        outputs = torch.stack(outputs)
        return outputs


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator, opt):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def initialize(self, inputs, eval):
        src = inputs[0]
        tgt = inputs[1]
        #print("^^"*10)
        #print(src)
        #print(tgt)
       
        enc_hidden, context = self.encoder(src)
        #print(enc_hidden, context)
        #exit(0)
        init_output = self.make_init_decoder_output(context)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        init_token = Variable(torch.LongTensor(
            [lb.Constants.BOS] * init_output.size(0)), volatile=eval)
        if self.opt.cuda:
            #print("*")
            init_token = init_token.to('cuda')
            #print(init_token.is_cuda)
        emb = self.decoder.word_lut(init_token)
        return tgt, (emb, init_output, enc_hidden, context.transpose(0, 1)) #BS*SL

    def forward(self, inputs, eval, regression=False):
        targets, init_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, init_states) #it will return all n=SL time steps out

        if regression:
            logits = self.generator(outputs)
            return logits.view_as(targets)
        return outputs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        grad_output, loss = self.generator.backward(outputs, targets, weights, normalizer, criterion, regression)
        outputs.backward(grad_output)
        return loss

    def predict(self, outputs, targets, weights, criterion): 
        return self.generator.predict(outputs, targets, weights, criterion)

    def translate(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=True)
        emb, output, hidden, context = init_states
        
        preds = [] 
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_() #targets SL*BS

        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            logit = self.generator(output)
            pred = logit.max(1)[1].view(-1).data
            preds.append(pred)

            # Stop if all sentences reach EOS.
            num_eos |= (pred == lb.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(pred))

        preds = torch.stack(preds)
        return preds

    def sample(self, inputs, max_length):
        max_length = 20
        targets, init_states = self.initialize(inputs, eval=False)
        emb, output, hidden, context = init_states

        outputs = []
        samples = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()
        print("Original sample")
        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            outputs.append(output)
            dist = F.softmax(self.generator(output))
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Stop if all sentences reach EOS.
            num_eos |= (sample == lb.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(sample))

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        print(samples.shape)
        print(max_length)
        return samples, outputs

##Extra func
    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to('cuda')
    def samples(self, inputs, max_length, dicts, sample_size=5):
        targets, init_states = self.initialize(inputs, eval=False)
        print(targets.shape)
        emb, output, hidden, context = init_states
        print(emb.shape,hidden[0].shape,hidden[1].shape,context.shape, output.shape)
        
        context = context.repeat(sample_size, 1, 1)
        emb = emb.repeat(sample_size, 1)
        h0, h1 = hidden
        h0 = h0.repeat(1,sample_size, 1)
        h1 = h1.repeat(1,sample_size, 1)
        hidden = (h0,h1)
        output= output.repeat(sample_size, 1)
        
        print(emb.shape, hidden[0].shape,hidden[1].shape, context.shape, output.shape)
        
        outputs = []
        samples = []
        batch_size = targets.size(1)
        total_sample_size = sample_size * batch_size
        
        attention_mask = inputs[0][0].data.eq(lb.Constants.PAD).t()
        
        print(attention_mask.shape)
        lenght = list(inputs[0][1]) * sample_size
        print(lenght)
        l = [len(sent) for _ in range(sample_size) for sent in context ]
        src_sent_masks = self.get_attention_mask(context, lenght)
        src_sent_masks= src_sent_masks.byte()
        print(l)
        print(inputs[0][1])
        
        attention_mask= attention_mask.repeat(sample_size, 1)
        #print(src_sent_masks.shape)
        #print(attention_mask.shape)
       
        self.decoder.attn.applyMask(src_sent_masks)
        num_eos = targets[0].data.byte().new(total_sample_size).zero_()
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device='cuda')
        sample_scores = torch.zeros(total_sample_size, device='cuda')

        #print(num_eos.shape)
        
        
        
        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            outputs.append(output)
            p_t = F.softmax(self.generator(output))
            log_p_t = torch.log(p_t)
            print(output.shape)
            y_t = p_t.multinomial(1, replacement=False)
            print(y_t.shape, log_p_t.shape) 
            
            log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
            y_t = y_t.view(-1).data
            print(y_t.shape) 
          
            samples.append(y_t)

            # Stop if all sentences reach EOS.
            num_eos |= (y_t == lb.Constants.EOS)
            
            sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())
            
            if num_eos.sum() == total_sample_size: break

            emb = self.decoder.word_lut(Variable(y_t))
        print(np.shape(samples)) #(47,)
        print(np.shape( sample_scores))
        #exit(0)
        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        print(samples.shape) #torch.Size([47, 40])  sl X bs
        print(outputs.shape) #torch.Size([47, 40, 256]) sl X bs X rnn_hidden
        #Filter the sentence here based on reward
        eos_id= lb.Constants.EOS
        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        _completed_outputs = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        print(batch_size)
        print(outputs[:,0,:])
        for t, y_t in enumerate(samples):
                print("==="*5)
                print(y_t)
                for i, sampled_word_id in enumerate(y_t):
                    sampled_word_id = sampled_word_id.cpu().item()
                    src_sent_id = i % batch_size
                    sample_id = i // batch_size
                    #print("="*3)
                    print(t,i)
                    print(src_sent_id)
                    print(sample_id)
                    print(sampled_word_id)
                    if t == 0 or _completed_samples[src_sent_id][sample_id][-1] != eos_id:
                        _completed_samples[src_sent_id][sample_id].append(sampled_word_id)
                    print("*")
                    print(_completed_samples) 
                        
                #print(dicts["tgt"].convertToLabelss(_completed_samples[src_sent_id][sample_id]))
                print("^"*3)
                print(_completed_samples)      
        #print(_completed_samples) 
             
        completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
        
        for src_sent_id in range(batch_size): #row is batch
                print("^^"*5)
                for sample_id in range(sample_size): #column 
                    offset = sample_id * batch_size + src_sent_id 
                    print("=="*5)
                    print(_completed_samples[src_sent_id][sample_id])
                    print(dicts["tgt"].convertToLabelss(_completed_samples[src_sent_id][sample_id]))
                    hyp = Hypothesis(value= dicts["tgt"].convertToLabelss(_completed_samples[src_sent_id][sample_id])[:-1],
                                 score=sample_scores[offset].item())
                    completed_samples[src_sent_id][sample_id] = hyp 
                              
                print("**"*5)    
        
        print(np.shape(completed_samples)) #(8, 5, 2)
        #exit(0) 
        return samples, outputs, completed_samples
    def n_sample(self, inputs, max_length, sample_size=5):
        max_length = 20
        sources = inputs[0][0]
        targets = inputs[1]
        batch_size = targets.size(1)
        
        
        targets, init_states = self.initialize(inputs, eval=False)
        print(inputs[0][0].shape, inputs[1].shape) #torch.Size([24, 4]) torch.Size([35, 4])
  
        emb, output, hidden, context = init_states
        print(emb.shape, hidden[0].shape, hidden[1].shape, context.shape, output.shape)
        #torch.Size([4, 128]) torch.Size([1, 4, 256]) torch.Size([1, 4, 256]) torch.Size([4, 24, 256]) torch.Size([4, 256])
        
        #Broadcast to sample n candidate for each input to decoder
        context = context.repeat(sample_size, 1, 1)
        emb = emb.repeat(sample_size, 1)
        h0, h1 = hidden
        h0 = h0.repeat(1,sample_size, 1)
        h1 = h1.repeat(1,sample_size, 1)
        hidden = (h0,h1)
        output= output.repeat(sample_size, 1)
        gold = inputs[0][2]
        print(emb.shape, hidden[0].shape, hidden[1].shape, context.shape, output.shape)
        #torch.Size([20, 128]) torch.Size([1, 20, 256]) torch.Size([1, 20, 256]) torch.Size([20, 24, 256]) torch.Size([20, 256])        
        total_sample_size = sample_size * batch_size
        attention_mask = inputs[0][0].data.eq(lb.Constants.PAD).t()
        lenght = list(inputs[0][1]) * sample_size
        print(lenght)
        
        src_sent_masks = self.get_attention_mask(context, lenght)
        src_sent_masks = src_sent_masks.byte()
        
        attention_mask = attention_mask.repeat(sample_size, 1)
        #print(src_sent_masks.shape)
        #print(attention_mask.shape)
       
        self.decoder.attn.applyMask(src_sent_masks)
        num_eos = targets[0].data.byte().new(total_sample_size).zero_()
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device='cuda')
        sample_scores = torch.zeros(total_sample_size, device='cuda')
        outputs = []
        samples = []
        
        for i in range(max_length):
           output, hidden = self.decoder.step(emb, output, hidden, context)
           outputs.append(output)
           p_t = F.softmax(self.generator(output))
           log_p_t = torch.log(p_t)

           y_t = p_t.multinomial(1, replacement=False)
          # print(y_t.shape, log_p_t.shape) 
            
           log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
           y_t = y_t.view(-1).data
           samples.append(y_t)

           # Stop if all sentences reach EOS.
           num_eos |= (y_t == lb.Constants.EOS)
            
           sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())
            
           if num_eos.sum() == total_sample_size: break

           emb = self.decoder.word_lut(Variable(y_t))
        print("Sample sample shape")   
        print(np.shape(samples)) #()
        print(np.shape(sample_scores)) #
        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        
        print(samples.shape) #torch.Size([47, 40])  sl X bs
        print(outputs.shape) #torch.Size([47, 40, 256]) sl X bs X rnn_hidden

        eos_id= lb.Constants.EOS
        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        _completed_outputs = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        
        for t, (y_t, h_t) in enumerate(zip(samples, outputs)):  # ([40, 256]) torch.Size([256])
           #print("==="*5)
           #print(y_t)
           #print(h_t.shape)
           for i, (sampled_word_id, hidden_t) in enumerate(zip(y_t, h_t)):
              sampled_word_id = sampled_word_id.cpu().item()  
              src_sent_id = i % batch_size
              sample_id = i // batch_size
              _completed_samples[src_sent_id][sample_id].append(sampled_word_id)
              _completed_outputs[src_sent_id][sample_id].append(hidden_t.tolist())                    
       
        print(np.shape(_completed_samples), np.shape(_completed_outputs), np.shape(gold))  #(8, 5, 47) (8, 5, 47, 256)
        print("Completed")
        print(_completed_samples) 
        print(max_length)      
        return samples, outputs, _completed_samples, gold, sample_scores    
    
              
    def ssample(self, src_sents: List[List[str]], sample_size=5, max_decoding_time_step=100) -> List[Hypothesis]:
        """
        Given a batched list of source sentences, randomly sample hypotheses from the model distribution p(y|x)
        Args:
            src_sents: a list of batched source sentences
            sample_size: sample size for each source sentence in the batch
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        print(src_sents)
        src_sents_var = self.vocab.src.to_input_tensor(src_sents, self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(sent) for sent in src_sents])
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec

        batch_size = len(src_sents)
        total_sample_size = sample_size * len(src_sents)

        # (total_sample_size, max_src_len, src_encoding_size)
        src_encodings = src_encodings.repeat(sample_size, 1, 1)
        src_encodings_att_linear = src_encodings_att_linear.repeat(sample_size, 1, 1)

        src_sent_masks = self.get_attention_mask(src_encodings, [len(sent) for _ in range(sample_size) for sent in src_sents])

        h_tm1 = (h_tm1[0].repeat(sample_size, 1), h_tm1[1].repeat(sample_size, 1))

        att_tm1 = torch.zeros(total_sample_size, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device=self.device)
        sample_scores = torch.zeros(total_sample_size, device=self.device)

        samples = [torch.tensor([self.vocab.tgt['<s>']] * total_sample_size, dtype=torch.long, device=self.device)]

        t = 0
        while t < max_decoding_time_step:
            t += 1

            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            # probabilities over target words 
            p_t = F.softmax(self.readout(att_t), dim=-1)
            log_p_t = torch.log(p_t)

            # (total_sample_size)
            y_t = torch.multinomial(p_t, num_samples=1)
            log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
            y_t = y_t.squeeze(1)

            samples.append(y_t)

            sample_ends |= torch.eq(y_t, eos_id).byte()
            sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())

            if torch.all(sample_ends):
                break

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        for t, y_t in enumerate(samples):
            for i, sampled_word_id in enumerate(y_t):
                sampled_word_id = sampled_word_id.cpu().item()
                src_sent_id = i % batch_size
                sample_id = i // batch_size

                if t == 0 or _completed_samples[src_sent_id][sample_id][-1] != eos_id:
                    _completed_samples[src_sent_id][sample_id].append(sampled_word_id)

        completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
        for src_sent_id in range(batch_size):
            for sample_id in range(sample_size):
                offset = sample_id * batch_size + src_sent_id
                hyp = Hypothesis(value=self.vocab.tgt.indices2words(_completed_samples[src_sent_id][sample_id])[:-1],
                                 score=sample_scores[offset].item())
                completed_samples[src_sent_id][sample_id] = hyp
        print(completed_samples)
        #exit(0)
        return completed_samples                   


