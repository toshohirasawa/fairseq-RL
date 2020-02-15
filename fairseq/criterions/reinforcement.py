# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.sequence_generator import SequenceGenerator
from fairseq import bleu

@register_criterion('reinforcement')
class ReinforcementCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.n_sample = args.criterion_sample_size
        self.pad = task.tgt_dict.pad()
        self.sample_gen = SequenceGenerator(task.tgt_dict, beam_size=args.criterion_sample_size, retain_dropout=True)
        self.greedy_gen = SequenceGenerator(task.tgt_dict, beam_size=1, retain_dropout=True)
        self.scorer = bleu.Scorer(task.tgt_dict.pad(), task.tgt_dict.eos(), task.tgt_dict.unk())

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--criterion-sample-size', type=int, default=5, help='Number of sample size (default: 5)')

    def reword(self, ref, pred):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref.type(torch.IntTensor), pred.type(torch.IntTensor))
        return self.scorer.score()

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Generate baseline/samples
        y_g = self.greedy_gen.generate([model], sample)
        y_hat = self.sample_gen.generate([model], sample)
        ref = sample['target']

        # rewords
        r_g = torch.tensor([self.reword(ref_i, y_g_i[0]['tokens']) for ref_i, y_g_i in zip(ref, y_g)])
        r_hat = torch.tensor([[self.reword(ref_i, y_hat_i_n['tokens']) for y_hat_i_n in y_hat_i] for ref_i, y_hat_i in zip(ref, y_hat)])
        r_d = r_hat - r_g.unsqueeze(-1)

        # scores
        net_input = {
            'src_tokens': sample['net_input']['src_tokens'],
            'src_lengths': sample['net_input']['src_lengths'],
        }
        encoder_out = model.encoder(**net_input)
        bos = sample['net_input']['prev_output_tokens'][:,:1]

        scores = []
        for n in range(self.n_sample):
            output_tokens = [y_hat_i[n]['tokens'] for y_hat_i in y_hat]
            output_tokens = rnn_utils.pad_sequence(output_tokens, batch_first=True, padding_value=self.pad)

            prev_output_tokens = torch.cat([bos, output_tokens], dim=-1)
            net_output = model.decoder(prev_output_tokens, encoder_out=encoder_out)
            
            lprobs = model.get_normalized_probs(net_output, log_probs=True)[:, :-1, :]
            lprobs = lprobs.reshape(-1, lprobs.size(-1))
            lprobs = lprobs[range(lprobs.size(0)), output_tokens.reshape(-1)]
            lprobs = lprobs.reshape(output_tokens.size())
            lprobs = lprobs.sum(dim=-1, keepdim=True)

            scores.append(lprobs)
        
        scores = torch.cat(scores, dim=-1)
        r_d = r_d.to(scores.device)

        loss = ((scores * r_d) / self.n_sample).sum()

        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
