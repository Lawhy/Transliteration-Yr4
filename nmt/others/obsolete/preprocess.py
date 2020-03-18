"""
Generate a vectorizer from the training data and print the summary of the dataset
"""

from utils.dataset import NMTDataset
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the vectorizer for NMT.')
    parser.add_argument('-tra_src', dest='tra_src', type=str, help='the path to the source training dataset')
    parser.add_argument('-tra_tgt', dest='tra_tgt', type=str, help='the path to the target training dataset')
    parser.add_argument('-val_src', dest='val_src', type=str, help='the path to the source validation dataset')
    parser.add_argument('-val_tgt', dest='val_tgt', type=str, help='the path to the target validation dataset')
    parser.add_argument('-tst_src', dest='tst_src', type=str, help='the path to the source testing dataset')
    parser.add_argument('-tst_tgt', dest='tst_tgt', type=str, help='the path to the target testing dataset')
    parser.add_argument('-save', dest='save', type=str, help='the dir to save the vectorizer, e.g. ./vecab.json')
    # e.g. 'data/en2ch.train.src', 'data/en2ch.train.tgt'
    args = parser.parse_args()
    print('Generating the vectorizer from training data...')
    dataset = NMTDataset.load_dataset_and_make_vectorizer(args.tra_src, args.tra_tgt, \
                                                          args.val_src, args.val_tgt, \
                                                          args.tst_src, args.tst_tgt)
    dataset.save_vectorizer(args.save)
    print('-------------- Summary --------------')
    print('training_data_size:', dataset.train_size)
    print('validation_data_size:', dataset.validation_size)
    print('testing_data_size:', dataset.test_size)
    print('-------------- Vectorizer -----------')
    print('source_vocab_size:', len(dataset._vectorizer.source_vocab.to_serializable()['token_to_idx']))
    print('target_vocab_size:', len(dataset._vectorizer.target_vocab.to_serializable()['token_to_idx']))
