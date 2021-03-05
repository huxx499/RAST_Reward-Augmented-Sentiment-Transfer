# -*- coding: utf-8 -*-
import sys, os
import re
import time
#import glob
import tensorflow as tf
import numpy as np
#from utils.data import load_dataset, load_paired_dataset, get_src_samples, get_disc_train_data, get_disc_infer_data, load_raml_data
from utils.data import *
from utils.vocab import load_vocab
from utils.noise import add_noise
from utils.sample import sampling
from seq2sentiseq.main import create_model as s2ss_create_model
import disc.disc_model_bert as disc_model_bert
#from regressor.main import create_model as reg_create_model
from common_options import *
from cycle_options import load_cycle_arguments
from utils import constants
import argparse
from seq2sentiseq.main import inference
from utils.evaluator import BLEUEvaluator
from nltk.translate.bleu_score import sentence_bleu

#import kenlm
#model = kenlm.Model('yelp.bin')

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # ignore warning

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
safe_divide_constant = 0.000001


def get_sentence_bleu(filepath):
    bleu_weight = []
    with open(filepath) as f:
        for line in f:
            text = line.strip().split('\t')
            references, candidates = [text[0]], text[2]
            bleu_2 = sentence_bleu(references, candidates, weights=(1, 1, 0, 0))
            bleu_weight.append(bleu_2)
    return np.array(bleu_weight)


def sigmoid(x, x_trans=0.0, x_scale=1.0, max_y=1):
    value = max_y / (1 + np.exp(-(x - x_trans) * x_scale))
    return value


def main():
    # === Load arguments
    args = load_cycle_arguments()
    if args.task_suffix == 'beta':
        final_model_save_path = args.final_model_save_dir + '-beta=' + str(args.beta) + '/'
        final_tsf_result_dir = args.final_tsf_result_dir + '-beta=' + str(args.beta)
    else:
        final_model_save_path = args.final_model_save_dir + '-' + args.task_suffix + '/'
        final_tsf_result_dir = args.final_tsf_result_dir + '-' + args.task_suffix
    dump_args_to_yaml(args, final_model_save_path) #把参数记录到文件中
    print(args)
    s2ss_args = load_args_from_yaml(args.s2ss_model_save_dir)
    s2ss_args.RL_learning_rate = args.RL_learning_rate  # a smaller learning_rate for RL
    s2ss_args.MLE_learning_rate = args.MLE_learning_rate  # a smaller learning_rate for MLE
    s2ss_args.batch_size = args.batch_size  # a bigger batch_size for RL
    min_seq_len = args.min_seq_len
    max_seq_len = args.max_seq_len


    # === Load global vocab 词到索引的映射表
    vocab, vocab_size = load_vocab(args.vocab_file)
    print("Vocabulary size: %s" % vocab_size)
    vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.vocab_file,  # target vocabulary file(each lines has a word)
        vocab_size=vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)


    # === Load evaluator
    bleu_evaluator = BLEUEvaluator()


    # === Create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)  # limit gpu memory; don't pre-allocate memory; allocate as-needed


    # === Load dataset
    with tf.device("/cpu:0"):  # Input pipeline should always be place on the CPU.

        #Acquire samples from Ps
        raml_train_data = args.raml_train_data + '-tau=' + str(args.tau) + '.txt'
        if not os.path.exists(raml_train_data):
            raml_data = sampling(args.train_template_data, args.vocab_file, raml_train_data, args.sample_size, args.tau)
        else:
            raml_data = load_raml_data(raml_train_data)
        print('pre-sampling complete, len(raml_data) = ', len(raml_data))

        train_data_iterator = load_paired_dataset(raml_train_data, vocab, batch_size=args.batch_size,
                                                  min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        test_data_iterator = load_dataset(args.test_data, vocab, mode=constants.TEST, batch_size=100,
                                          min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        #paired_train_data_iterator = load_paired_dataset(args.pseudo_data, vocab, batch_size=args.batch_size,
                                                         #min_seq_len=min_seq_len, max_seq_len=max_seq_len)

        train_data_next = train_data_iterator.get_next()  # to avoid high number of `Iterator.get_next()` calls
        test_data_next = test_data_iterator.get_next()
        #paired_train_data_next = paired_train_data_iterator.get_next()


    # === Initialize and build Seq2SentiSeq model 创建了三种模型 分别用于训练/测试_greedy/测试_random
    load_model = False if args.no_pretrain else True
    s2ss_train = s2ss_create_model(sess, s2ss_args, constants.TRAIN, vocab_size, load_pretrained_model=load_model) #load预训练模型

    #decode_type_before = s2ss_args.decode_type
    s2ss_args.decode_type = constants.GREEDY #不同的decode_type 对应不同的infer
    s2ss_greedy_infer = s2ss_create_model(sess, s2ss_args, constants.INFER, vocab_size, reuse=True) # infer reuse表示
    #s2ss_args.decode_type = constants.RANDOM
    #s2ss_random_infer = s2ss_create_model(sess, s2ss_args, constants.INFER, vocab_size, reuse=True)
    #s2ss_args.decode_type = decode_type_before

    if args.task_suffix == 'beta':
        disc_model_save_dir = args.disc_model_save_dir + '-beta=' + str(args.beta) + '/'
    else:
        disc_model_save_dir = args.disc_model_save_dir + '-' + args.task_suffix + '/'
    if args.disc_pretrain and args.task_suffix != 'wo-RAT' and args.task_suffix != 'wo-disc':
        # === Pre-train discriminator
        src_fs = get_src_samples(args.first_sample_num, args.train_template_data, args.disc_data_dir)
        first_sample_test_data_iterator = load_dataset(src_fs[0], vocab, mode=constants.TEST, batch_size=100,
                                                 min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        first_sample_test_data_next = first_sample_test_data_iterator.get_next()

        dst_fs = inference(s2ss_greedy_infer, sess=sess, args=s2ss_args, decoder_s=constants.SENT_LIST,
                           src_test_iterator=first_sample_test_data_iterator, src_test_next=first_sample_test_data_next,
                           vocab_rev=vocab_rev, result_dir=args.disc_data_dir)
        
        get_disc_train_data(src_fs[1], dst_fs[0], args.disc_data_dir)
        
        if not os.path.exists(disc_model_save_dir):
            os.makedirs(disc_model_save_dir)
        _ = disc_model_bert.bert_main('train', args.disc_data_dir, disc_model_save_dir, args.disc_pretrain_n_epoch, args.disc_batch_size)
    

    # === Start adversarial training
    for adv_iter in range(args.adv_iters): 

        if args.task_suffix != 'wo-RAT':
            # calculate bleu reward
            raml_bleu_data = args.raml_bleu_data + '-tau=' + str(args.tau) + '.txt'
            if not os.path.exists(raml_bleu_data):
                # calculate bleu reward
                cont_reward = get_sentence_bleu(raml_train_data)
                cont_reward = sigmoid(cont_reward, x_trans=0.3, x_scale=8)
                #print(cont_reward[:10])
                #print(np.min(cont_reward), np.max(cont_reward), np.median(cont_reward), np.mean(cont_reward))
                # to do: write bleu data
                with open(raml_bleu_data, 'w') as f:
                    cont_reward_str = [str(r) for r in cont_reward]
                    f.write('\n'.join(cont_reward_str))
            else:
                #Load bleu data
                with open(raml_bleu_data) as f:
                    cont_reward = np.array([float(line.strip()) for line in f])
                print(np.min(cont_reward), np.max(cont_reward), np.median(cont_reward), np.mean(cont_reward))
            print(len(cont_reward))
            cont_reward += safe_divide_constant

            if args.task_suffix == 'wo-disc':
                reward = cont_reward
                print('length of reward :%d' % len(reward))
            else:
                # calculate senti reward
                sample_size = 1
                if not os.path.exists(args.disc_data_dir):
                    os.makedirs(args.disc_data_dir)
                get_disc_infer_data(args.disc_data_dir, raml_data, sample_size)
                senti_rewards = disc_model_bert.bert_main('predict', args.disc_data_dir, disc_model_save_dir, predict_batch_size=args.disc_pred_batch_size)
                senti_reward_per_sample = np.array(senti_rewards).reshape((-1,2*sample_size)).sum(axis=1)
                senti_reward = sigmoid(senti_reward_per_sample, x_trans=0.5, x_scale=8)
                print(np.min(senti_reward_per_sample), np.max(senti_reward_per_sample), np.median(senti_reward_per_sample), np.mean(senti_reward_per_sample))
                print(np.min(senti_reward), np.max(senti_reward), np.median(senti_reward), np.mean(senti_reward))

                assert len(cont_reward) == len(senti_reward)

                senti_reward += safe_divide_constant

                beta = args.beta #trade-off between two rewards
                reward = (1 + beta * beta) * senti_reward * cont_reward / (beta * beta * senti_reward + cont_reward)
                print('length of reward :%d' % len(reward))

            '''
            # calculate fluency reward
            lm_scores = []
            ppl_scores = []
            for idx,data in enumerate(raml_data):
                sent = data['src']
                lm_scores.append(model.score(' '.join(sent), bos=True, eos=True))
                ppl_scores.append(model.perplexity(' '.join(sent)))
                #if idx<5:
                    #print(sent,' '.join(sent),lm_scores[idx])
            fluency_reward = sigmoid(np.array(lm_scores), x_trans=-21, x_scale=0.4)
            #print(np.min(np.array(lm_scores)), np.max(np.array(lm_scores)), np.median(np.array(lm_scores)), np.mean(np.array(lm_scores)))
            #print(fluency_reward[:10])
            print(np.min(np.array(ppl_scores)), np.max(np.array(ppl_scores)), np.median(np.array(ppl_scores)), np.mean(np.array(ppl_scores)))
            print(np.min(np.array(lm_scores)), np.max(np.array(lm_scores)), np.median(np.array(lm_scores)), np.mean(np.array(lm_scores)))
            print(np.min(fluency_reward), np.max(fluency_reward), np.median(fluency_reward), np.mean(fluency_reward))
            #print(len(fluency_reward))
            #break
            '''
            #assert len(cont_reward) == len(senti_reward) == len(fluency_reward)

            #fluency_reward += safe_divide_constant

            '''
            #reward = (senti_reward + cont_reward + fluency_reward) / 3
            w1 = w2 = w3 = 1
            reward = (w1 + w2 + w3) * cont_reward * senti_reward * fluency_reward / (w1 * senti_reward * fluency_reward + w2 * cont_reward * fluency_reward + w3 * cont_reward * senti_reward)
            print(np.min(reward), np.max(reward), np.median(reward), np.mean(reward))

            print('length of reward :%d' % len(reward))
            #break
            '''

        # === Start train G
        n_batch = -1
        global_step = -1

        for i in range(args.n_epoch):
            print("Epoch:%s" % i)

            sess.run([train_data_iterator.initializer])

            while True:
                n_batch += 1
                global_step += 1
                if n_batch % args.eval_step == 0: #eval
                    print('\n================ N_batch / Global_step (%s / %s): Evaluate on test datasets ================\n'
                          % (n_batch, global_step))
                    dst_fs = inference(s2ss_greedy_infer, sess=sess, args=s2ss_args, decoder_s=constants.SENT_LIST,
                                       src_test_iterator=test_data_iterator, src_test_next=test_data_next,
                                       vocab_rev=vocab_rev, result_dir=final_tsf_result_dir,
                                       step=global_step if args.save_each_step else global_step)
                    t0 = time.time()
                    bleu_scores = bleu_evaluator.score(args.reference, dst_fs[1], all_bleu=True)
                    print("Test(Batch:%d)\tBLEU-1:%.3f\tBLEU-2:%.3f\tBLEU:%.3f\tCost time:%.2f" %
                          (n_batch, bleu_scores[1], bleu_scores[2], bleu_scores[0], time.time() - t0))

                if n_batch % args.save_per_step == 0:
                    print("Save model at dir:", final_model_save_path)
                    s2ss_train.saver.save(sess, final_model_save_path, global_step=global_step)

                try:
                    t0 = time.time()
                    data = sess.run(train_data_next)  # get real data!!
                    batch_size = np.shape(data["source_ids"])[0]
                    #decode_width = s2ss_args.decode_width

                    t0 = time.time()

                    if args.task_suffix != 'wo-RAT':
                        #batched_cont_reward = cont_reward[n_batch*batch_size:(n_batch+1)*batch_size]
                        batched_reward = reward[n_batch*batch_size:(n_batch+1)*batch_size]

                        feed_dict = {s2ss_train.encoder_input: data["source_ids"],
                                     s2ss_train.encoder_input_len: data["source_length"],
                                     s2ss_train.decoder_input: data["target_ids_in"],
                                     s2ss_train.decoder_target: data["target_ids_out"],
                                     s2ss_train.decoder_target_len: data["target_length"] + 1,
                                     s2ss_train.decoder_s: data["target_senti"],
                                     s2ss_train.reward: batched_reward}
                        res = sess.run([s2ss_train.rl_loss, s2ss_train.retrain_op], feed_dict=feed_dict)
                        #sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)
                    
                    if args.task_suffix != 'wo-bt':
                        # baseline #每个句子对应1个tgt_senti
                        greedy_predictions = sess.run(
                            s2ss_greedy_infer.predictions,
                            feed_dict={s2ss_greedy_infer.encoder_input: data["source_ids"],
                                       s2ss_greedy_infer.encoder_input_len: data["source_length"],
                                       s2ss_greedy_infer.decoder_s: data["target_senti"]})

                        mid_ids_bs, mid_ids_in_bs, mid_ids_out_bs, mid_ids_in_out_bs, mid_ids_length_bs = \
                            process_mid_ids(greedy_predictions, min_seq_len, max_seq_len, vocab_size) #处理一些符号
                
                        # Update Seq2SentiSeq with previous model generated data with noise
                        if global_step < 1 :
                            print('$$$Update B use back_trans_noise data')
                        noise_ids, noise_ids_length = add_noise(mid_ids_bs, mid_ids_length_bs)
                        feed_dict = {
                            s2ss_train.encoder_input: noise_ids,
                            s2ss_train.encoder_input_len: noise_ids_length,
                            s2ss_train.decoder_input: data["source_ids_in"],
                            s2ss_train.decoder_target: data["source_ids_out"],
                            s2ss_train.decoder_target_len: data["source_length"] + 1,
                            s2ss_train.decoder_s: data["source_senti"],
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)
                    

                except tf.errors.OutOfRangeError:  # next epoch
                    print("Train---Total N batch:{}\tCost time:{}".format(n_batch, time.time() - t0))
                    n_batch = -1
                    break

                if n_batch % args.eval_step == 0:
                    print('train loss: %.4f' % res[0])

        if args.task_suffix != 'wo-RAT' and args.task_suffix != 'wo-disc':
            # Train discriminator to distinguish real data and generated data
            src_fs = get_src_samples(args.sample_num, args.train_template_data, args.disc_data_dir)
            sample_test_data_iterator = load_dataset(src_fs[0], vocab, mode=constants.TEST, batch_size=100,
                                                     min_seq_len=min_seq_len, max_seq_len=max_seq_len)
            sample_test_data_next = sample_test_data_iterator.get_next()

            dst_fs = inference(s2ss_greedy_infer, sess=sess, args=s2ss_args, decoder_s=constants.SENT_LIST,
                               src_test_iterator=sample_test_data_iterator, src_test_next=sample_test_data_next,
                               vocab_rev=vocab_rev, result_dir=args.disc_data_dir)
            get_disc_train_data(src_fs[1], dst_fs[0], args.disc_data_dir)

            _ = disc_model_bert.bert_main('train', args.disc_data_dir, disc_model_save_dir, args.disc_n_epoch, args.disc_batch_size)
        
        
    

if __name__ == "__main__":
    main()
