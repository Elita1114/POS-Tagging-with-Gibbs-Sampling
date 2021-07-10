import argparse

from file_utils import *








def main():
    corpus_words, corpus_tags = read_data("data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu", PADDING_START_END_length=2, pos_tags_learning= pos_tags_learning)










if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="learning POS tags UNSUPERVISED!!"
    )

    argparser.add_argument("--train_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu", help="train_path.")
    argparser.add_argument("--dev_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu", help="dev_path.")
    argparser.add_argument("--test_path", type=str, default="data/ud-treebanks-v2.8/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu", help="test_path.")
    args = argparser.parse_args()

    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path

    pos_tags_learning = [".", ","]


    main()