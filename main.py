import argparse

from file_utils import *








def main():
    corpus_words, corpus_tags, indeces_untagged_words = arrange_corpus(train_path, dev_path, test_path, PADDING_START_END_length, pos_tags_learning)










if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="learning POS tags UNSUPERVISED!!"
    )

    argparser.add_argument("--train_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu", help="train_path.")
    argparser.add_argument("--dev_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu", help="dev_path.")
    argparser.add_argument("--test_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu", help="test_path.")
    argparser.add_argument("--PADDING_START_END_length", type=int, default=2, help="PADDING_START_END_length")

    args = argparser.parse_args()

    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path
    PADDING_START_END_length = args.PADDING_START_END_length

    pos_tags_learning = [".", ","]


    main()