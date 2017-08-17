#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import json
import numpy
import copy
import argparse

import os
import sys
import time
import logging

import itertools

from subprocess import Popen

from collections import OrderedDict

profile = False

from data_iterator import TextIterator
from training_progress import TrainingProgress
from util import *
from theano_util import *
from alignment_util import *

from layers import *
from initializers import *
from optimizers import *
from metrics.scorer_provider import ScorerProvider

from domain_interpolation_data_iterator import DomainInterpolatorTextIterator

# batch preparation
def prepare_data(batch, maxlen=None, n_words_src=30000,
                 n_words=30000, n_factors=1):

    # x: a list of sentences
    if maxlen is not None:
        new_batch = [[] for _ in range(len(batch))]

        for parallel_sentences in zip(*batch):
            if not any(len(sentence) > maxlen for sentence in parallel_sentences):
                for idx in range(len(new_batch)):
                    new_batch[idx].append(parallel_sentences[idx])

        batch = new_batch

        if any(len(batch) < 1 for batch in batch):
                return None

    n_samples = len(batch[0])
    lengths_x = [len(s) for s in batch[0]]
    maxlen_x = numpy.max(lengths_x) + 1
    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(floatX)
    for idx, s_x in enumerate(batch[0]):
        lengths_x = len(s_x)
        x[:, :lengths_x, idx] = zip(*s_x)
        x_mask[:lengths_x + 1, idx] = 1.

    batch_out = [x, x_mask]

    for target_batch in batch[1:]:
        lengths_y = [len(sentence) for sentence in target_batch]
        maxlen_y = numpy.max(lengths_y) + 1
        batch_out.append(numpy.zeros((maxlen_y, n_samples)).astype('int64'))
        batch_out.append(numpy.zeros((maxlen_y, n_samples)).astype(floatX))
        for idx, s_y in enumerate(target_batch):
            batch_out[-2][:lengths_y[idx], idx] = s_y
            batch_out[-1][:lengths_y[idx]+1, idx] = 1.
    return batch_out

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params = get_layer_param('embedding')(options, params, options['n_words'][0], options['dim_per_factor'], options['factors'], suffix='')
    #
    if not options['tie_encoder_decoder_embeddings']:
        for decoder_idx in range(options['outputs']):
            params = get_layer_param('embedding')(options, params, options['n_words'][decoder_idx+1], options['dim_word'], suffix=pp('_dec', decoder_idx))

    # encoder: bidirectional RNN
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              recurrence_transition_depth=options['enc_recurrence_transition_depth'])
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              recurrence_transition_depth=options['enc_recurrence_transition_depth'])
    if options['enc_depth'] > 1:
        for level in range(2, options['enc_depth'] + 1):
            prefix_f = pp('encoder', level)
            prefix_r = pp('encoder_r', level)

            if level <= options['enc_depth_bidirectional']:
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_f,
                                                             nin=options['dim'],
                                                             dim=options['dim'],
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_r,
                                                             nin=options['dim'],
                                                             dim=options['dim'],
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])
            else:
                params = get_layer_param(options['encoder'])(options, params,
                                                             prefix=prefix_f,
                                                             nin=options['dim'] * 2,
                                                             dim=options['dim'] * 2,
                                                             recurrence_transition_depth=options['enc_recurrence_transition_depth'])


    ctxdim = 2 * options['dim']

    dec_state = options['dim']
    if options['decoder_deep'].startswith('lstm'):
        dec_state *= 2

        # init_state, init_cell

    params = get_layer_param('ff')(options, params, prefix='ff_state',
                                   nin=ctxdim, nout=dec_state)

    # iterating over the outputs, different name for each layer belonging to different decoder
    for decoder_idx in range(options['outputs']):
        decoder_idx = str(decoder_idx)

        # decoder
        params = get_layer_param(options['decoder'])(options, params,
                                                  prefix=pp('decoder', decoder_idx),
                                                  nin=options['dim_word'],
                                                  dim=options['dim'],
                                                  dimctx=ctxdim,
                                                  recurrence_transition_depth=options['dec_base_recurrence_transition_depth'])

        # deeper layers of the decoder
        if options['dec_depth'] > 1:
            if options['dec_deep_context']:
                input_dim = options['dim'] + ctxdim
            else:
                input_dim = options['dim']

            for level in range(2, options['dec_depth'] + 1):
                decoder_name = pp('decoder', decoder_idx)
                params = get_layer_param(options['decoder_deep'])(options, params,
                                                prefix=pp(decoder_name, level),
                                                nin=input_dim,
                                                dim=options['dim'],
                                                dimctx=ctxdim,
                                                recurrence_transition_depth=options['dec_high_recurrence_transition_depth'])

        # readout
        params = get_layer_param('ff')(options, params, prefix=pp('ff_logit_lstm', decoder_idx),
                                    nin=options['dim'], nout=options['dim_word'],
                                    ortho=False)
        params = get_layer_param('ff')(options, params, prefix=pp('ff_logit_prev', decoder_idx),
                                    nin=options['dim_word'],
                                    nout=options['dim_word'], ortho=False)
        params = get_layer_param('ff')(options, params, prefix=pp('ff_logit_ctx', decoder_idx),
                                    nin=ctxdim, nout=options['dim_word'],
                                    ortho=False)

        params = get_layer_param('ff')(options, params, prefix=pp('ff_logit', decoder_idx),
                                    nin=options['dim_word'],
                                    nout=options['n_words'][int(decoder_idx)+1],
                                    weight_matrix = not options['tie_decoder_embeddings'],
                                    followed_by_softmax=True)

    return params


# bidirectional RNN encoder: take input x (optionally with mask), and produce sequence of context vectors (ctx)
def build_encoder(tparams, options, dropout, x_mask=None, sampling=False):

    x = tensor.tensor3('x', dtype='int64')
    # source text; factors 1; length 5; batch size 10
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')

    # for the backward rnn, we just need to invert x
    xr = x[:, ::-1]
    if x_mask is None:
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    # word embedding for forward rnn (source)
    emb = get_layer_constr('embedding')(tparams, x, suffix='', factors= options['factors'])

    # word embedding for backward rnn (source)
    embr = get_layer_constr('embedding')(tparams, xr, suffix='', factors= options['factors'])

    if options['use_dropout']:
        source_dropout = dropout((n_timesteps, n_samples, 1), options['dropout_source'])
        if not sampling:
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
        emb *= source_dropout

        if sampling:
            embr *= source_dropout
        else:
            # we drop out the same words in both directions
            embr *= source_dropout[::-1]


    ## level 1
    proj = get_layer_constr(options['encoder'])(tparams, emb, options, dropout,
                                                prefix='encoder',
                                                mask=x_mask,
                                                dropout_probability_below=options['dropout_embedding'],
                                                dropout_probability_rec=options['dropout_hidden'],
                                                recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                truncate_gradient=options['encoder_truncate_gradient'],
                                                profile=profile)
    projr = get_layer_constr(options['encoder'])(tparams, embr, options, dropout,
                                                 prefix='encoder_r',
                                                 mask=xr_mask,
                                                 dropout_probability_below=options['dropout_embedding'],
                                                 dropout_probability_rec=options['dropout_hidden'],
                                                 recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                 truncate_gradient=options['encoder_truncate_gradient'],
                                                 profile=profile)

    # discard LSTM cell state
    if options['encoder'].startswith('lstm'):
        proj[0] = get_slice(proj[0], 0, options['dim'])
        projr[0] = get_slice(projr[0], 0, options['dim'])

    ## bidirectional levels before merge
    for level in range(2, options['enc_depth_bidirectional'] + 1):
        prefix_f = pp('encoder', level)
        prefix_r = pp('encoder_r', level)

        # run forward on previous backward and backward on previous forward
        input_f = projr[0][::-1]
        input_r = proj[0][::-1]

        proj = get_layer_constr(options['encoder'])(tparams, input_f, options, dropout,
                                                    prefix=prefix_f,
                                                    mask=x_mask,
                                                    dropout_probability_below=options['dropout_hidden'],
                                                    dropout_probability_rec=options['dropout_hidden'],
                                                    recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                    truncate_gradient=options['encoder_truncate_gradient'],
                                                    profile=profile)
        projr = get_layer_constr(options['encoder'])(tparams, input_r, options, dropout,
                                                     prefix=prefix_r,
                                                     mask=xr_mask,
                                                     dropout_probability_below=options['dropout_hidden'],
                                                     dropout_probability_rec=options['dropout_hidden'],
                                                     recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                     truncate_gradient=options['encoder_truncate_gradient'],
                                                     profile=profile)

        # discard LSTM cell state
        if options['encoder'].startswith('lstm'):
            proj[0] = get_slice(proj[0], 0, options['dim'])
            projr[0] = get_slice(projr[0], 0, options['dim'])

        # residual connections
        if level > 1:
            proj[0] += input_f
            projr[0] += input_r

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    ## forward encoder layers after bidirectional layers are concatenated
    for level in range(options['enc_depth_bidirectional'] + 1, options['enc_depth'] + 1):

        ctx += get_layer_constr(options['encoder'])(tparams, ctx, options, dropout,
                                                   prefix=pp('encoder', level),
                                                   mask=x_mask,
                                                   dropout_probability_below=options['dropout_hidden'],
                                                   dropout_probability_rec=options['dropout_hidden'],
                                                   recurrence_transition_depth=options['enc_recurrence_transition_depth'],
                                                   truncate_gradient=options['encoder_truncate_gradient'],
                                                   profile=profile)[0]

    return x, ctx


# RNN decoder (including embedding and feedforward layer before output)
def build_decoder(tparams, options, y, ctx, init_state, dropout, x_mask=None, y_mask=None, sampling=False, pctx_=None, decoder_idx='0', shared_vars=None):
    opt_ret = dict()

    # tell RNN whether to advance just one step at a time (for sampling),
    # or loop through sequence (for training)
    if sampling:
        one_step=True
    else:
        one_step=False

    if options['use_dropout']:
        if sampling:
            target_dropout = dropout(dropout_probability=options['dropout_target'])
        else:
            n_timesteps_trg = y.shape[0]
            n_samples = y.shape[1]
            target_dropout = dropout((n_timesteps_trg, n_samples, 1), options['dropout_target'])
            target_dropout = tensor.tile(target_dropout, (1, 1, options['dim_word']))

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else pp('_dec', decoder_idx)
    emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
    if options['use_dropout']:
        emb *= target_dropout

    if sampling:
        emb = tensor.switch(y[:, None] < 0,
            tensor.zeros((1, options['dim_word'])),
            emb)
    else:
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options, dropout,
                                            prefix=pp('decoder', decoder_idx),
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            pctx_=pctx_,
                                            one_step=one_step,
                                            init_state=init_state[0],
                                            recurrence_transition_depth=options['dec_base_recurrence_transition_depth'],
                                            dropout_probability_below=options['dropout_embedding'],
                                            dropout_probability_ctx=options['dropout_hidden'],
                                            dropout_probability_rec=options['dropout_hidden'],
                                            truncate_gradient=options['decoder_truncate_gradient'],
                                            profile=profile)
    # hidden states of the decoder gru
    next_state = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # we return state of each layer
    if sampling:
        ret_state = [next_state.reshape((1, next_state.shape[0], next_state.shape[1]))]
    else:
        ret_state = None

    if options['dec_depth'] > 1:
        for level in range(2, options['dec_depth'] + 1):

            # don't pass LSTM cell state to next layer
            if options['decoder'].startswith('lstm'):
                next_state = get_slice(next_state, 0, options['dim'])

            if options['dec_deep_context']:
                if sampling:
                    axis=1
                else:
                    axis=2
                input_ = tensor.concatenate([next_state, ctxs], axis=axis)
            else:
                input_ = next_state
            decoder_name = pp('decoder', decoder_idx)
            out_state = get_layer_constr(options['decoder_deep'])(tparams, input_, options, dropout,
                                              prefix=pp(decoder_name, level),
                                              mask=y_mask,
                                              context=ctx,
                                              context_mask=x_mask,
                                              pctx_=None, #TODO: we can speed up sampler by precomputing this
                                              one_step=one_step,
                                              init_state=init_state[level-1],
                                              dropout_probability_below=options['dropout_hidden'],
                                              dropout_probability_rec=options['dropout_hidden'],
                                              recurrence_transition_depth=options['dec_high_recurrence_transition_depth'],
                                              truncate_gradient=options['decoder_truncate_gradient'],
                                              profile=profile)[0]

            if sampling:
                ret_state.append(out_state.reshape((1, proj[0].shape[0], proj[0].shape[1])))

            # don't pass LSTM cell state to next layer
            if options['decoder'].startswith('lstm'):
                out_state = get_slice(out_state, 0, options['dim'])

            # residual connection
            next_state += out_state

    # don't pass LSTM cell state to next layer
    elif options['decoder'].startswith('lstm'):
        next_state = get_slice(next_state, 0, options['dim'])

    if sampling:
        if options['dec_depth'] > 1:
            ret_state = tensor.concatenate(ret_state, axis=0)
        else:
            ret_state = ret_state[0]

    # hidden layer taking RNN state, previous word embedding and context vector as input
    # (this counts as the first layer in our deep output, which is always on)
    logit_lstm = get_layer_constr('ff')(tparams, next_state, options, dropout,
                                    dropout_probability=options['dropout_hidden'],
                                    prefix=pp('ff_logit_lstm', decoder_idx),
                                    activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options, dropout,
                                    dropout_probability=options['dropout_embedding'],
                                    prefix=pp('ff_logit_prev', decoder_idx),
                                    activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options, dropout,
                                   dropout_probability=options['dropout_hidden'],
                                   prefix=pp('ff_logit_ctx', decoder_idx),
                                   activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    # last layer
    logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
    logit = get_layer_constr('ff')(tparams, logit, options, dropout,
                            dropout_probability=options['dropout_hidden'],
                            prefix=pp('ff_logit', decoder_idx), activ='linear', W=logit_W, followed_by_softmax=True)

    return logit, opt_ret, ret_state

# build a training model
def build_model(tparams, options, scoring=False):

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy_floatX(0.))
    dropout = dropout_constr(options, use_noise, trng, sampling=False)

    x_mask = tensor.matrix('x_mask', dtype=floatX)
    #y = tensor.matrix('y', dtype='int64')
    #y_mask = tensor.matrix('y_mask', dtype=floatX)
    #y = tensor.ltensor3('y')
    #y_mask_type = tensor.TensorType(floatX, (False,)*3)
    #y_mask = y_mask_type('y_mask')
    # source text length 5; batch size 10
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype(floatX)
    # target text length 8; batch size 10
    #y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    #y_mask.tag.test_value = numpy.ones(shape=(2, 8, 10)).astype(floatX)

    x, ctx = build_encoder(tparams, options, dropout, x_mask, sampling=False)
    n_samples = x.shape[2]

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoders state
    init_state = get_layer_constr('ff')(tparams, ctx_mean, options, dropout,
                                        dropout_probability=options['dropout_hidden'],
                                        prefix='ff_state', activ='tanh')

    # every decoder RNN layer gets its own copy of the init state
    init_state = init_state.reshape([1, init_state.shape[0], init_state.shape[1]])
    if options['dec_depth'] > 1:
        init_state = tensor.tile(init_state, (options['dec_depth'], 1, 1))

    inps = [x, x_mask]

    if scoring:
        start = options['outputs'] - 1
    else:
        start = 0

    for decoder_idx in range(start, options['outputs']):
        decoder_idx = str(decoder_idx)

        y = tensor.matrix(pp('y', decoder_idx), dtype='int64')
        y_mask = tensor.matrix(pp('y_mask', decoder_idx), dtype=floatX)
        y.tag.test_value = (numpy.random.rand(8, 10) * 100).astype('int64')
        y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype(floatX)
        y_flat = y.flatten()
        #da capire un'attimo la cosa del vocabolario!
        y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'][-1] + y_flat

        logit, opt_ret, _ = build_decoder(tparams, options, y, ctx, init_state, dropout, x_mask=x_mask, y_mask=y_mask, decoder_idx=decoder_idx, sampling=False)

        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))

        # cost
        cost_dec = -tensor.log(probs.flatten()[y_flat_idx])
        cost_dec = cost_dec.reshape([y.shape[0], y.shape[1]])
        cost_dec = (cost_dec * y_mask).sum(0)

        inps.append(y)
        inps.append(y_mask)

        if decoder_idx == '0':
            cost = cost_dec
        else:
            cost += cost_dec

    return trng, use_noise, inps, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):

    dropout = dropout_constr(options, use_noise, trng, sampling=True)

    x, ctx = build_encoder(tparams, options, dropout, x_mask=None, sampling=True)
    n_samples = x.shape[2]

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options, dropout,
                                        dropout_probability=options['dropout_hidden'],
                                        prefix='ff_state', activ='tanh')

    # every decoder RNN layer gets its own copy of the init state
    init_state = init_state.reshape([1, init_state.shape[0], init_state.shape[1]])
    if options['dec_depth'] > 1:
        init_state = tensor.tile(init_state, (options['dec_depth'], 1, 1))

    logging.info('Building f_init...')
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    logging.info('Done')

    init_state_old = init_state
    init_state = tensor.tensor3('init_state', dtype=floatX)
    if theano.config.compute_test_value != 'off':
        init_state.tag.test_value = numpy.random.rand(*init_state_old.tag.test_value.shape).astype(floatX)

    inps = [ctx, init_state]
    f_next = []
    for decoder_idx in range(options['outputs']):
        decoder_idx = str(decoder_idx)

        # x: 1 x 1
        y = tensor.vector(pp('y_sampler', decoder_idx), dtype='int64')
        y.tag.test_value = -1 * numpy.ones((10,)).astype('int64')

        logit, opt_ret, ret_state = build_decoder(tparams, options, y, ctx, init_state, dropout, x_mask=None, y_mask=None, decoder_idx=decoder_idx, sampling=True)

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)

        # sample from softmax distribution to get the sample
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        outs = [next_probs, next_sample, ret_state]

        if return_alignment:
            outs.append(opt_ret['dec_alphas'])

        logging.info('Building {}..'.format(pp('f_next', decoder_idx)))
        tmp_inps = [y] + inps
        f_next.append(theano.function(tmp_inps, outs, name=pp('f_next', decoder_idx), profile=profile))
        logging.info('Done')

    return f_init, f_next


# minimum risk cost
# assumes cost is the negative sentence-level log probability
# and each sentence in the minibatch is a sample of the same source sentence
def mrt_cost(cost, y_mask, options):
    loss = tensor.vector('loss', dtype=floatX)
    alpha = theano.shared(numpy_floatX(options['mrt_alpha']))

    if options['mrt_ml_mix'] > 0:
        ml_cost = cost[0]

        # remove reference for MRT objective unless enabled
        if not options['mrt_reference']:
            cost = cost[1:]

    cost *= alpha

    #get normalized probability
    cost = tensor.nnet.softmax(-cost)[0]

    # risk: expected loss
    if options['mrt_ml_mix'] > 0 and not options['mrt_reference']:
        cost *= loss[1:]
    else:
        cost *= loss


    cost = cost.sum()

    if options['mrt_ml_mix'] > 0:
        #normalize ML by length (because MRT is length-invariant)
        ml_cost /= y_mask[:,0].sum(0)
        ml_cost *= options['mrt_ml_mix']
        cost += ml_cost

    return cost, loss


# build a sampler that produces samples in one theano function
def build_full_sampler(tparams, options, use_noise, trng, greedy=False):

    dropout = dropout_constr(options, use_noise, trng, sampling=True)

    if greedy:
        x_mask = tensor.matrix('x_mask', dtype=floatX)
        x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype(floatX)
    else:
        x_mask = None

    x, ctx = build_encoder(tparams, options, dropout, x_mask, sampling=True)
    n_samples = x.shape[2]

    if x_mask:
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
    else:
        ctx_mean = ctx.mean(0)

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options, dropout,
                                    dropout_probability=options['dropout_hidden'],
                                    prefix='ff_state', activ='tanh')

    # every decoder RNN layer gets its own copy of the init state
    init_state = init_state.reshape([1, init_state.shape[0], init_state.shape[1]])
    if options['dec_depth'] > 1:
        init_state = tensor.tile(init_state, (options['dec_depth'], 1, 1))

    if greedy:
        init_w = tensor.alloc(numpy.int64(-1), n_samples)
    else:
        k = tensor.iscalar("k")
        k.tag.test_value = 12
        init_w = tensor.alloc(numpy.int64(-1), k*n_samples)

        ctx = tensor.tile(ctx, [k, 1])

        init_state = tensor.tile(init_state, [1, k, 1])

    # projected context
    assert ctx.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(ctx*dropout(dropout_probability=options['dropout_hidden']), tparams[pp('decoder', 'Wc_att')]) +\
        tparams[pp('decoder', 'b_att')]

    def decoder_step(y, init_state, ctx, pctx_, *shared_vars):

        logit, opt_ret, ret_state = build_decoder(tparams, options, y, ctx, init_state, dropout, x_mask=x_mask, y_mask=None, sampling=True, pctx_=pctx_, shared_vars=shared_vars)

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)

        if greedy:
            next_sample = next_probs.argmax(1)
        else:
            # sample from softmax distribution to get the sample
            next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # do not produce words after EOS
        next_sample = tensor.switch(
                      tensor.eq(y,0),
                      0,
                      next_sample)

        return [next_sample, ret_state, next_probs[:, next_sample].diagonal()], \
               theano.scan_module.until(tensor.all(tensor.eq(next_sample, 0))) # stop when all outputs are 0 (EOS)


    decoder_prefixes = ['decoder']
    if options['dec_depth'] > 1:
        for level in range(2, options['dec_depth'] + 1):
            decoder_prefixes.append(pp('decoder', level))

    shared_vars = []
    for prefix in decoder_prefixes:
        shared_vars.extend([tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_tt')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]])

    n_steps = tensor.iscalar("n_steps")
    n_steps.tag.test_value = 50

    (sample, state, probs), updates = theano.scan(decoder_step,
                        outputs_info=[init_w, init_state, None],
                        non_sequences=[ctx, pctx_]+shared_vars,
                        n_steps=n_steps, truncate_gradient=options['decoder_truncate_gradient'])

    logging.info('Building f_sample...')
    if greedy:
        inps = [x, x_mask, n_steps]
    else:
        inps = [x, k, n_steps]
    outs = [sample, probs]
    f_sample = theano.function(inps, outs, name='f_sample', updates=updates, profile=profile)
    logging.info('Done')

    return f_sample



# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
# In the case of multi-output it will sample only from the last decoder.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False,
               return_hyp_graph=False):

    # k is the beam size we have
    if k > 1 and argmax:
        assert not stochastic, \
            'Beam search does not support stochastic sampling with argmax'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
    hyp_graph = None
    if stochastic:
        if argmax:
            sample_score = 0
        live_k=k
    else:
        live_k = 1

    if return_hyp_graph:
        from hypgraph import HypGraph
        hyp_graph = HypGraph()

    dead_k = 0

    hyp_samples=[ [] for i in xrange(live_k) ]
    word_probs=[ [] for i in xrange(live_k) ]
    hyp_scores = numpy.zeros(live_k).astype(floatX)
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    dec_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)

        # to more easily manipulate batch size, go from (layers, batch_size, dim) to (batch_size, layers, dim)
        ret[0] = numpy.transpose(ret[0], (1,0,2))

        next_state[i] = numpy.tile( ret[0] , (live_k, 1, 1))
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((live_k,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    # sampling using only the decoder for the translation task
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])

            # for theano function, go from (batch_size, layers, dim) to (layers, batch_size, dim)
            next_state[i] = numpy.transpose(next_state[i], (1,0,2))

            inps = [next_w, ctx, next_state[i]]
            ret = f_next[i][-1](*inps)

            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            # to more easily manipulate batch size, go from (layers, batch_size, dim) to (batch_size, layers, dim)
            next_state[i] = numpy.transpose(next_state[i], (1,0,2))

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:
            #batches are not supported with argmax: output data structure is different
            if argmax:
                nw = sum(next_p)[0].argmax()
                sample.append(nw)
                sample_score += numpy.log(next_p[0][0, nw])
                if nw == 0:
                    break
            else:
                # FIXME: sampling is currently performed according to the last model only
                nws = next_w_tmp
                cand_scores = numpy.array(hyp_scores)[:, None] - numpy.log(next_p[-1])
                probs = next_p[-1]

                for idx,nw in enumerate(nws):
                    hyp_samples[idx].append(nw)


                hyp_states=[]
                for ti in xrange(live_k):
                    hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                    hyp_scores[ti]=cand_scores[ti][nws[ti]]
                    word_probs[ti].append(probs[ti][nws[ti]])

                new_hyp_states=[]
                new_hyp_samples=[]
                new_hyp_scores=[]
                new_word_probs=[]
                for hyp_sample,hyp_state, hyp_score, hyp_word_prob in zip(hyp_samples,hyp_states,hyp_scores, word_probs):
                    if hyp_sample[-1] > 0:
                        new_hyp_samples.append(copy.copy(hyp_sample))
                        new_hyp_states.append(copy.copy(hyp_state))
                        new_hyp_scores.append(hyp_score)
                        new_word_probs.append(hyp_word_prob)
                    else:
                        sample.append(copy.copy(hyp_sample))
                        sample_score.append(hyp_score)
                        sample_word_probs.append(hyp_word_prob)

                hyp_samples=new_hyp_samples
                hyp_states=new_hyp_states
                hyp_scores=new_hyp_scores
                word_probs=new_word_probs

                live_k=len(hyp_samples)
                if live_k < 1:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = [numpy.array(state) for state in zip(*hyp_states)]
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            #averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype(floatX)
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if return_hyp_graph:
                    word, history = new_hyp_samples[idx][-1], new_hyp_samples[idx][:-1]
                    score = new_hyp_scores[idx]
                    word_prob = new_word_probs[idx][-1]
                    hyp_graph.add(word, history, word_prob=word_prob, cost=score)
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(copy.copy(new_hyp_samples[idx]))
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(copy.copy(new_hyp_samples[idx]))
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(copy.copy(new_hyp_states[idx]))
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    # dump every remaining one
    if not argmax and live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])
            sample_word_probs.append(word_probs[idx])
            if return_alignment:
                alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment, hyp_graph


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalization_alpha=0.0, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for batch in iterator:
        #ensure consistency in number of factors
        if len(batch[0][0][0]) != options['factors']:
            logging.error('Mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(batch[0])

        batch = prepare_data(batch, n_factors=options['factors'])

        ### in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(batch)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(*batch)

        # normalize scores according to output length
        if normalization_alpha:
            for y_mask in batch[3::2]:
                adjusted_lengths = numpy.array([numpy.count_nonzero(s) ** normalization_alpha for s in y_mask.T])
                pprobs /= adjusted_lengths

        for pp in pprobs:
            probs.append(pp)

        logging.debug('%d samples computed' % (n_done))

    return numpy.array(probs), alignments_json


def train(dim_word=512,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          enc_depth=1, # number of layers in the encoder
          dec_depth=1, # number of layers in the decoder
          enc_recurrence_transition_depth=1, # number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru)
          dec_base_recurrence_transition_depth=2, # number of GRU transition operations applied in the first layer of the decoder. Minimum is 2. (Only applies to gru_cond)
          dec_high_recurrence_transition_depth=1, # number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru)
          dec_deep_context=False, # include context vectors in deeper layers of the decoder
          enc_depth_bidirectional=None, # first n encoder layers are bidirectional (default: all)
          factors=1, # input factors
          outputs=1, # outputs for multi-task learning
          dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          decoder_deep='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=1000,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0., # L2 regularization penalty towards original weights
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.0001,  # learning rate
          n_words=[None],  # list of vocabulary size, one size per all source dictionaries, one per each target dictionary
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=10000,
          saveFreq=30000,   # save the parameters after every saveFreq updates
          sampleFreq=10000,   # generate some samples after every sampleFreq
          datasets=[ # path to training datasets (source and target)
                    None,
                    None],
          valid_datasets=[None, # path to validation datasets (source and target)
                          None],
          dictionaries=[ # path to dictionaries (json file created with ../data/build_dictionary.py). One dictionary per input factor; last dictionary is target-side dictionary.
                        None,
                        None],
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
          reload_=False,
          reload_training_progress=True, # reload trainig progress (only used if reload_ is True)
          overwrite=False,
          external_validation_script=None,
          shuffle_each_epoch=True,
          sort_by_length=True,
          use_domain_interpolation=False, # interpolate between an out-domain training corpus and an in-domain training corpus
          domain_interpolation_min=0.1, # minimum (initial) fraction of in-domain training data
          domain_interpolation_max=1.0, # maximum fraction of in-domain training data
          domain_interpolation_inc=0.1, # interpolation increment to be applied each time patience runs out, until maximum amount of interpolation is reached
          domain_interpolation_indomain_datasets=[None, None], # in-domain parallel training corpus (source and target)
          anneal_restarts=0, # when patience run out, restart with annealed learning rate X times before early stopping
          anneal_decay=0.5, # decay learning rate by this amount on each restart
          maxibatch_size=20, #How many minibatches to load at one time
          objective="CE", #CE: cross-entropy; MRT: minimum risk training (see https://www.aclweb.org/anthology/P/P16/P16-1159.pdf)
          mrt_alpha=0.005,
          mrt_samples=100,
          mrt_samples_meanloss=10,
          mrt_reference=False,
          mrt_loss="SENTENCEBLEU n=4", # loss function for minimum risk training
          mrt_ml_mix=0, # interpolate mrt loss with ML loss
          model_version=0.1, #store version used for training for compatibility
          prior_model=None, # Prior model file, used for MAP
          tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
          tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
          encoder_truncate_gradient=-1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
          decoder_truncate_gradient=-1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
          layer_normalisation=False, # layer normalisation https://arxiv.org/abs/1607.06450
          weight_normalisation=False, # normalize weights
    ):

    # Model options
    model_options = OrderedDict(sorted(locals().copy().items()))

    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            logging.error('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(dictionaries) == factors + outputs) # one dictionary per source and target
    assert(len(datasets) == outputs + 1) # one dataset per output
    assert(len(n_words) == outputs + 1) # one dict len per sources, one per each target
    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    assert(prior_model != None and (os.path.exists(prior_model)) or (map_decay_c==0.0)) # MAP training requires a prior model file

    assert(enc_recurrence_transition_depth >= 1) # enc recurrence transition depth must be at least 1.
    assert(dec_base_recurrence_transition_depth >= 2) # dec base recurrence transition depth must be at least 2.
    assert(dec_high_recurrence_transition_depth >= 1) # dec higher recurrence transition depth must be at least 1.

    if model_options['enc_depth_bidirectional'] is None:
        model_options['enc_depth_bidirectional'] = model_options['enc_depth']
    # first layer is always bidirectional; make sure people don't forget to increase enc_depth as well
    assert(model_options['enc_depth_bidirectional'] >= 1 and model_options['enc_depth_bidirectional'] <= model_options['enc_depth'])

    if model_options['dec_depth'] > 1 and model_options['decoder'].startswith('lstm') != model_options['decoder_deep'].startswith('lstm'):
        logging.error('cannot mix LSTM and GRU in decoder')
        logging.error('decoder: {0}'.format(model_options['decoder']))
        logging.error('decoder_deep: {0}'.format(model_options['decoder_deep']))
        sys.exit(1)

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    if n_words is [None]:
        n_words = [len(dict_) for dict_ in worddicts]
        model_options['n_words'] = n_words

    # all the words dicts must be equal.
    if tie_encoder_decoder_embeddings:
        assert (min(n_words) == max(n_words)), "When tying encoder and decoder embeddings, source and target vocabulary size must the same"
        if any(worddicts[0] != worddict for worddict in worddicts[1:]):
            logging.warning("Encoder-decoder embedding tying is enabled with different source and target dictionaries. This is usually a configuration error")

    if model_options['objective'] == 'MRT':
        # in CE mode parameters are updated once per batch; in MRT mode parameters are updated once
        # per pair of train sentences (== per batch of samples), so we set batch_size to 1 to make
        # model saving, validation, etc trigger after the same number of updates as before
        logging.info('Running in MRT mode, minibatch size set to 1 sentence')
        batch_size = 1

    # initialize training progress
    training_progress = TrainingProgress()
    best_p = None
    best_opt_p = None
    training_progress.bad_counter = 0
    training_progress.anneal_restarts_done = 0
    training_progress.uidx = 0
    training_progress.eidx = 0
    training_progress.estop = False
    training_progress.history_errs = []
    training_progress.domain_interpolation_cur = domain_interpolation_min if use_domain_interpolation else None
    # reload training progress
    training_progress_file = saveto + '.progress.json'
    if reload_ and reload_training_progress and os.path.exists(training_progress_file):
        logging.info('Reloading training progress')
        training_progress.load_from_json(training_progress_file)
        if (training_progress.estop == True) or (training_progress.eidx > max_epochs) or (training_progress.uidx >= finish_after):
            logging.warning('Training is already complete. Disable reloading of training progress (--no_reload_training_progress) or remove or modify progress file (%s) to train anyway.' % training_progress_file)
            return numpy.inf

    # adjust learning rate if we resume process that has already entered annealing phase
    if training_progress.anneal_restarts_done > 0:
        lrate *= anneal_decay**training_progress.anneal_restarts_done

    logging.info('Loading data')
    if use_domain_interpolation:
        logging.info('Using domain interpolation with initial ratio %s, final ratio %s, increase rate %s' % (training_progress.domain_interpolation_cur, domain_interpolation_max, domain_interpolation_inc))
        train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         skip_empty=True,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         indomain_source=domain_interpolation_indomain_datasets[0],
                         indomain_target=domain_interpolation_indomain_datasets[1],
                         interpolation_rate=training_progress.domain_interpolation_cur,
                         use_factor=(factors > 1),
                         maxibatch_size=maxibatch_size)
    else:
        train = TextIterator(datasets,
                         dictionaries,
                         n_words_dicts=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         skip_empty=False,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         factors=factors,
                         outputs=outputs,
                         maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets,
                            dictionaries,
                            n_words_dicts=n_words,
                            batch_size=valid_batch_size,
                            factors=factors,
                            outputs=outputs,
                            maxlen=maxlen)
    else:
        valid = None

    comp_start = time.time()

    logging.info('Building model')
    params = init_params(model_options)

    optimizer_params = {}
    # prepare parameters
    if reload_ and os.path.exists(saveto):
        logging.info('Reloading model parameters')
        params = load_params(saveto, params)
        logging.info('Reloading optimizer parameters')
        try:
            logging.info('trying to load optimizer params from {0} or {1}'.format(saveto + '.gradinfo', saveto + '.gradinfo.npz'))
            optimizer_params = load_optimizer_params(saveto + '.gradinfo', optimizer)
        except IOError:
            logging.warning('{0}(.npz) not found. Trying to load optimizer params from {1}(.npz)'.format(saveto + '.gradinfo', saveto))
            optimizer_params = load_optimizer_params(saveto, optimizer)
    elif prior_model:
        logging.info('Initializing model parameters from prior')
        params = load_params(prior_model, params)

    # load prior model if specified
    if prior_model:
        logging.info('Loading prior model parameters')
        params = load_params(prior_model, params, with_prefix='prior_')

    tparams = init_theano_params(params)

    trng, use_noise, \
        inps, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)

    #inps = [x, x_mask] + y + y_mask

    if validFreq or sampleFreq:
        logging.info('Building sampler')
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)
    if model_options['objective'] == 'MRT':
        logging.info('Building MRT sampler')
        f_sampler = build_full_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    logging.info('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=profile)
    logging.info('Done')

    if model_options['objective'] == 'CE':
        cost = cost.mean()
    elif model_options['objective'] == 'MRT':
        #MRT objective function
        cost, loss = mrt_cost(cost, y_mask, model_options)
        inps += [loss]
    else:
        logging.error('Objective must be one of ["CE", "MRT"]')
        sys.exit(1)

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy_floatX(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            init_value = tparams['prior_' + kk]
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay

    updated_params = OrderedDict(tparams)

    # don't update prior model parameters
    if prior_model:
        updated_params = OrderedDict([(key,value) for (key,value) in updated_params.iteritems() if not key.startswith('prior_')])

    logging.info('Computing gradient...')
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    logging.info('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    logging.info('Building optimizers...')
    f_update, optimizer_tparams = eval(optimizer)(lr, updated_params,
                                                                 grads, inps, cost,
                                                                 profile=profile,
                                                                 optimizer_params=optimizer_params)
    logging.info('Done')

    logging.info('Total compilation time: {0:.1f}s'.format(time.time() - comp_start))

    if validFreq == -1 or saveFreq == -1 or sampleFreq == -1:
        logging.info('Computing number of training batches')
        num_batches = len(train)
        logging.info('There are {} batches in the train set'.format(num_batches))

        if validFreq == -1:
            validFreq = num_batches
        if saveFreq == -1:
            saveFreq = num_batches
        if sampleFreq == -1:
            sampleFreq = num_batches

    logging.info('Optimization')

    #save model options
    json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)

    valid_err = None

    cost_sum = 0
    cost_batches = 0
    last_disp_samples = 0
    last_words = 0
    ud_start = time.time()
    p_validation = None

    for training_progress.eidx in xrange(training_progress.eidx, max_epochs):
        n_samples = 0

        for batch in train:
            training_progress.uidx += 1
            use_noise.set_value(1.)

            #ensure consistency in number of factors
            if len(batch[0]) and len(batch[0][0]) and len(batch[0][0][0]) != factors:
                logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(factors, len(x[0][0])))
                sys.exit(1)

            xlen = len(batch[0])
            n_samples += xlen

            if model_options['objective'] == 'CE':

                batch = prepare_data(batch, maxlen=maxlen, n_factors=factors)

                if batch is None:
                    logging.warning('Minibatch with zero sample under length %d' % maxlen)
                    training_progress.uidx -= 1
                    continue

                cost_batches += 1
                last_disp_samples += xlen
                last_words_tmp = 0

                for idx in range(1, len(batch), 2):
                    last_words_tmp += numpy.sum(batch[idx])

                last_words += 2*last_words_tmp/len(batch)

                # compute cost, grads and update parameters
                cost = f_update(lrate, *batch)

                cost_sum += cost

            elif model_options['objective'] == 'MRT':
                assert maxlen is not None and maxlen > 0

                xy_pairs = [(x_i, y_i) for (x_i, y_i) in zip(x, y) if len(x_i) < maxlen and len(y_i) < maxlen]
                if not xy_pairs:
                    training_progress.uidx -= 1
                    continue

                for x_s, y_s in xy_pairs:

                    # add EOS and prepare factored data
                    x, _, _, _ = prepare_data([x_s], [y_s], maxlen=None,
                                              n_factors=factors,
                                              n_words_src=n_words_src, n_words=n_words)

                    # draw independent samples to compute mean reward
                    if model_options['mrt_samples_meanloss']:
                        use_noise.set_value(0.)
                        samples, _ = f_sampler(x, model_options['mrt_samples_meanloss'], maxlen)
                        use_noise.set_value(1.)

                        samples = [numpy.trim_zeros(item) for item in zip(*samples)]

                        # map integers to words (for character-level metrics)
                        samples = [seqs2words(sample, worddicts_r[-1]) for sample in samples]
                        ref = seqs2words(y_s, worddicts_r[-1])

                        #scorers expect tokenized hypotheses/references
                        ref = ref.split(" ")
                        samples = [sample.split(" ") for sample in samples]

                        # get negative smoothed BLEU for samples
                        scorer = ScorerProvider().get(model_options['mrt_loss'])
                        scorer.set_reference(ref)
                        mean_loss = numpy.array(scorer.score_matrix(samples), dtype=floatX).mean()
                    else:
                        mean_loss = 0.

                    # create k samples
                    use_noise.set_value(0.)
                    samples, _ = f_sampler(x, model_options['mrt_samples'], maxlen)
                    use_noise.set_value(1.)

                    samples = [numpy.trim_zeros(item) for item in zip(*samples)]

                    # remove duplicate samples
                    samples.sort()
                    samples = [s for s, _ in itertools.groupby(samples)]

                    # add gold translation [always in first position]
                    if model_options['mrt_reference'] or model_options['mrt_ml_mix']:
                        samples = [y_s] + [s for s in samples if s != y_s]

                    # create mini-batch with masking
                    x, x_mask, y, y_mask = prepare_data([x_s for _ in xrange(len(samples))], samples,
                                                                    maxlen=None,
                                                                    n_factors=factors,
                                                                    n_words_src=n_words_src,
                                                                    n_words=n_words)

                    cost_batches += 1
                    last_disp_samples += xlen
                    last_words += (numpy.sum(x_mask) + numpy.sum(y_mask))/2.0

                    # map integers to words (for character-level metrics)
                    samples = [seqs2words(sample, worddicts_r[-1]) for sample in samples]
                    y_s = seqs2words(y_s, worddicts_r[-1])

                    #scorers expect tokenized hypotheses/references
                    y_s = y_s.split(" ")
                    samples = [sample.split(" ") for sample in samples]

                    # get negative smoothed BLEU for samples
                    scorer = ScorerProvider().get(model_options['mrt_loss'])
                    scorer.set_reference(y_s)
                    loss = mean_loss - numpy.array(scorer.score_matrix(samples), dtype=floatX)

                    # compute cost, grads and update parameters
                    cost = f_update(lrate, x, x_mask, y, y_mask, loss)

                    cost_sum += cost

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                logging.warning('NaN detected')
                return 1., 1., 1.

            # verbose
            if numpy.mod(training_progress.uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                sps = last_disp_samples / float(ud)
                wps = last_words / float(ud)
                cost_avg = cost_sum / float(cost_batches)
                logging.info(
                    'Epoch {epoch} Update {update} Cost {cost} UD {ud} {sps} {wps}'.format(
                        epoch=training_progress.eidx,
                        update=training_progress.uidx,
                        cost=cost_avg,
                        ud=ud,
                        sps="{0:.2f} sents/s".format(sps),
                        wps="{0:.2f} words/s".format(wps)
                    )
                )
                ud_start = time.time()
                cost_batches = 0
                last_disp_samples = 0
                last_words = 0
                cost_sum = 0

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(training_progress.uidx, saveFreq) == 0:
                logging.info('Saving the best model...')
                if best_p is not None:
                    params = best_p
                    optimizer_params = best_opt_p
                else:
                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

                save(params, optimizer_params, training_progress, saveto)
                logging.info('Done')

                # save with uidx
                if not overwrite:
                    logging.info('Saving the model at iteration {}...'.format(training_progress.uidx))
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], training_progress.uidx)

                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    save(params, optimizer_params, training_progress, saveto_uidx)
                    logging.info('Done')


            # generate some samples with the model and display them
            if sampleFreq and numpy.mod(training_progress.uidx, sampleFreq) == 0:
                # FIXME: random selection?
                # sampling just from the last decoder, the main translation task
                x = batch[0]
                x_mask = batch[1]
                y = batch[-2]
                y_mask = batch[-1]
                for jj in xrange(numpy.minimum(5, x.shape[2])):
                    stochastic = True
                    x_current = x[:, :, jj][:, :, None]

                    # remove padding
                    x_current = x_current[:,:x_mask.astype('int64')[:, jj].sum(),:]

                    sample, score, sample_word_probs, alignment, hyp_graph = gen_sample([f_init], [f_next],
                                               x_current,
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False,
                                               return_hyp_graph=False)
                    print 'Source ', jj, ': ',
                    for pos in range(x.shape[1]):
                        if x[0, pos, jj] == 0:
                            break
                        for factor in range(factors):
                            vv = x[factor, pos, jj]
                            if vv in worddicts_r[factor]:
                                sys.stdout.write(worddicts_r[factor][vv])
                            else:
                                sys.stdout.write('UNK')
                            if factor+1 < factors:
                                sys.stdout.write('|')
                            else:
                                sys.stdout.write(' ')
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample[0]
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if valid is not None and validFreq and numpy.mod(training_progress.uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                training_progress.history_errs.append(float(valid_err))

                if training_progress.uidx == 0 or valid_err <= numpy.array(training_progress.history_errs).min():
                    best_p = unzip_from_theano(tparams, excluding_prefix='prior_')
                    best_opt_p = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    training_progress.bad_counter = 0
                if valid_err >= numpy.array(training_progress.history_errs).min():
                    training_progress.bad_counter += 1
                    if training_progress.bad_counter > patience:

                        # change mix of in-domain and out-of-domain data
                        if use_domain_interpolation and (training_progress.domain_interpolation_cur < domain_interpolation_max):
                            training_progress.domain_interpolation_cur = min(training_progress.domain_interpolation_cur + domain_interpolation_inc, domain_interpolation_max)
                            logging.info('No progress on the validation set, increasing domain interpolation rate to %s and resuming from best params' % training_progress.domain_interpolation_cur)
                            train.adjust_domain_interpolation_rate(training_progress.domain_interpolation_cur)
                            if best_p is not None:
                                zip_to_theano(best_p, tparams)
                                zip_to_theano(best_opt_p, optimizer_tparams)
                            training_progress.bad_counter = 0

                        # anneal learning rate and reset optimizer parameters
                        elif training_progress.anneal_restarts_done < anneal_restarts:
                            logging.info('No progress on the validation set, annealing learning rate and resuming from best params.')
                            lrate *= anneal_decay
                            training_progress.anneal_restarts_done += 1
                            training_progress.bad_counter = 0

                            # reload best parameters
                            if best_p is not None:
                                zip_to_theano(best_p, tparams)

                            # reset optimizer parameters
                            for item in optimizer_tparams.values():
                                item.set_value(numpy.array(item.get_value()) * 0.)

                        # stop
                        else:
                            logging.info('Valid {}'.format(valid_err))
                            logging.info('Early Stop!')
                            training_progress.estop = True
                            break

                logging.info('Valid {}'.format(valid_err))

                if external_validation_script:
                    logging.info("Calling external validation script")
                    if p_validation is not None and p_validation.poll() is None:
                        logging.info("Waiting for previous validation run to finish")
                        logging.info("If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes")
                        valid_wait_start = time.time()
                        p_validation.wait()
                        logging.info("Waited for {0:.1f} seconds".format(time.time()-valid_wait_start))
                    logging.info('Saving  model...')
                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    save(params, optimizer_params, training_progress, saveto+'.dev')
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    logging.info('Done')
                    p_validation = Popen([external_validation_script])

            # finish after this many updates
            if training_progress.uidx >= finish_after:
                logging.info('Finishing after %d iterations!' % training_progress.uidx)
                training_progress.estop = True
                break

        logging.info('Seen %d samples' % n_samples)

        if training_progress.estop:
            break

    if best_p is not None:
        zip_to_theano(best_p, tparams)
        zip_to_theano(best_opt_p, optimizer_tparams)

    if valid is not None:
        use_noise.set_value(0.)
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
        valid_err = valid_errs.mean()

        logging.info('Valid {}'.format(valid_err))

    if best_p is not None:
        params = copy.copy(best_p)
        optimizer_params = copy.copy(best_opt_p)

    else:
        params = unzip_from_theano(tparams, excluding_prefix='prior_')
        optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

    save(params, optimizer_params, training_progress, saveto)

    return valid_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs='+',
                         help="parallel training corpus (source and (multiple) target)")
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (one per source factor, plus one per target vocabulary)")
    data.add_argument('--model', type=str, default='model.npz', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--reload', action='store_true',  dest='reload_',
                         help="load existing model (if '--model' points to existing model)")
    data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                         help="don't reload training progress (only used if --reload is enabled)")
    data.add_argument('--overwrite', action='store_true',
                         help="write all models to same file")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dim', type=int, default=1000, metavar='INT',
                         help="hidden layer size (default: %(default)s)")
    network.add_argument('--n_words', type=int, default=None, metavar='INT', nargs='+',
                         help="target vocabulary size (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',
                         help="number of decoder layers (default: %(default)s)")

    network.add_argument('--enc_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")
    network.add_argument('--dec_base_recurrence_transition_depth', type=int, default=2, metavar='INT',
                         help="number of GRU transition operations applied in the first layer of the decoder. Minimum is 2.  (Only applies to gru_cond). (default: %(default)s)")
    network.add_argument('--dec_high_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")

    network.add_argument('--dec_deep_context', action='store_true',
                         help="pass context vector (from first layer) to deep decoder layers")
    network.add_argument('--enc_depth_bidirectional', type=int, default=None, metavar='INT',
                         help="number of bidirectional encoder layer; if enc_depth is greater, remaining layers are unidirectional; by default, all layers are bidirectional.")

    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")
    network.add_argument('--outputs', type=int, default=1, metavar='INT',
                         help="number of outputs (decoders) (default: %(default)s)")
    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--use_dropout', action="store_true",
                         help="use dropout layer (default: %(default)s)")
    network.add_argument('--dropout_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for input embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_source', type=float, default=0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")
    network.add_argument('--layer_normalisation', action="store_true",
                         help="use layer normalisation (default: %(default)s)")
    network.add_argument('--weight_normalisation', action="store_true",
                         help=" normalize weights (default: %(default)s)")
    network.add_argument('--tie_encoder_decoder_embeddings', action="store_true", dest="tie_encoder_decoder_embeddings",
                         help="tie the input embeddings of the encoder and the decoder (first factor only). Source and target vocabulary size must the same")
    network.add_argument('--tie_decoder_embeddings', action="store_true", dest="tie_decoder_embeddings",
                         help="tie the input embeddings of the decoder with the softmax output embeddings")
    network.add_argument('--encoder', type=str, default='gru',
                         choices=['gru', 'lstm'],
                         help='encoder recurrent layer (default: %(default)s)')
    network.add_argument('--decoder', type=str, default='gru_cond',
                         choices=['gru_cond', 'lstm_cond'],
                         help='first decoder recurrent layer (default: %(default)s)')
    network.add_argument('--decoder_deep', type=str, default='gru',
                         choices=['gru', 'gru_cond', 'lstm'],
                         help='decoder recurrent layer after first one (default: %(default)s)')

    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--optimizer', type=str, default="adam",
                         choices=['adam', 'adadelta', 'rmsprop', 'sgd', 'sgdmomentum'],
                         help="optimizer (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=10000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--decay_c', type=float, default=0, metavar='FLOAT',
                         help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0, metavar='FLOAT',
                         help="L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument('--objective', choices=['CE', 'MRT'], default='CE',
                         help='training objective. CE: cross-entropy minimization (default); MRT: Minimum Risk Training (https://www.aclweb.org/anthology/P/P16/P16-1159.pdf)')
    training.add_argument('--encoder_truncate_gradient', type=int, default=-1, metavar='INT',
                         help="truncate BPTT gradients in the encoder to this value. Use -1 for no truncation (default: %(default)s)")
    training.add_argument('--decoder_truncate_gradient', type=int, default=-1, metavar='INT',
                         help="truncate BPTT gradients in the encoder to this value. Use -1 for no truncation (default: %(default)s)")

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs='+',
                         help="parallel validation corpus (source and target) (default: %(default)s)")
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validation minibatch size (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--anneal_restarts', type=int, default=0, metavar='INT',
                         help="when patience runs out, restart training INT times with annealed learning rate (default: %(default)s)")
    validation.add_argument('--anneal_decay', type=float, default=0.5, metavar='FLOAT',
                         help="learning rate decay on each restart (default: %(default)s)")
    validation.add_argument('--external_validation_script', type=str, default=None, metavar='PATH',
                         help="location of validation script (to run your favorite metric for validation) (default: %(default)s)")

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")

    mrt = parser.add_argument_group('minimum risk training parameters')
    mrt.add_argument('--mrt_alpha', type=float, default=0.005, metavar='FLOAT',
                         help="MRT alpha (default: %(default)s)")
    mrt.add_argument('--mrt_samples', type=int, default=100, metavar='INT',
                         help="samples per source sentence (default: %(default)s)")
    mrt.add_argument('--mrt_samples_meanloss', type=int, default=10, metavar='INT',
                         help="draw n independent samples to calculate mean loss (which is subtracted from loss) (default: %(default)s)")
    mrt.add_argument('--mrt_loss', type=str, default='SENTENCEBLEU n=4', metavar='STR',
                         help='loss used in MRT (default: %(default)s)')
    mrt.add_argument('--mrt_reference', action="store_true",
                         help='add reference to MRT samples.')
    mrt.add_argument('--mrt_ml_mix', type=float, default=0, metavar='FLOAT',
                     help="mix in ML objective in MRT training with this scaling factor (default: %(default)s)")

    domain_interpolation = parser.add_argument_group('domain interpolation parameters')
    domain_interpolation.add_argument('--use_domain_interpolation', action='store_true',  dest='use_domain_interpolation',
                         help="interpolate between an out-domain training corpus and an in-domain training corpus")
    domain_interpolation.add_argument('--domain_interpolation_min', type=float, default=0.1, metavar='FLOAT',
                         help="minimum (initial) fraction of in-domain training data (default: %(default)s)")
    domain_interpolation.add_argument('--domain_interpolation_max', type=float, default=1.0, metavar='FLOAT',
                         help="maximum fraction of in-domain training data (default: %(default)s)")
    domain_interpolation.add_argument('--domain_interpolation_inc', type=float, default=0.1, metavar='FLOAT',
                         help="interpolation increment to be applied each time patience runs out, until maximum amount of interpolation is reached (default: %(default)s)")
    domain_interpolation.add_argument('--domain_interpolation_indomain_datasets', type=str, metavar='PATH', nargs=2,
                         help="indomain parallel training corpus (source and target)")

    args = parser.parse_args()

    # set up logging
    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    #print vars(args)
    train(**vars(args))

#    Profile peak GPU memory usage by uncommenting next line and enabling theano CUDA memory profiling (http://deeplearning.net/software/theano/tutorial/profiling.html)
#    print theano.sandbox.cuda.theano_allocated()
