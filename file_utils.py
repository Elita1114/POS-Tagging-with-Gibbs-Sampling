from tqdm import tqdm
from constants import START_PAD, END_PAD, field_values_dictionary


def arrange_corpus(train_path, dev_path, test_path, PADDING_START_END_length, pos_tags_learning):
    corpus_words_train, corpus_tags_train, indeces_untagged_words_train, lemmas_train = read_data(train_path,
                                                                                                  PADDING_START_END_length=PADDING_START_END_length,
                                                                                                  pos_tags_learning=pos_tags_learning)
    corpus_words_dev, corpus_tags_dev, indeces_untagged_words_dev, lemmas_dev = read_data(dev_path,
                                                                                          PADDING_START_END_length=PADDING_START_END_length,
                                                                                          pos_tags_learning=pos_tags_learning,
                                                                                          start_index=len(
                                                                                              corpus_tags_train))
    corpus_words = corpus_words_train + corpus_words_dev
    corpus_tags = corpus_tags_train + corpus_tags_dev
    indeces_untagged_words = indeces_untagged_words_train + indeces_untagged_words_dev
    lemmas = lemmas_train.union(lemmas_dev)
    corpus_words_test, corpus_tags_test, indeces_untagged_words_test, lemmas_test = read_data(test_path,
                                                                                              PADDING_START_END_length=PADDING_START_END_length,
                                                                                              pos_tags_learning=pos_tags_learning,
                                                                                              start_index=len(
                                                                                                  corpus_tags))
    corpus_words += corpus_words_test
    corpus_tags += corpus_tags_test
    indeces_untagged_words += indeces_untagged_words_test
    lemmas = lemmas.union(lemmas_test)
    return corpus_words, corpus_tags, indeces_untagged_words, lemmas


def read_data(corpus_file_path, PADDING_START_END_length, pos_tags_learning, start_index=0):
    """
       Splits the corpus_words data into list of sentences.
       Also collects all the unique lemmas.
       Each sentence is a list of dictionaries.
       Each dictionary represents a word's fields.
       The field_names are the keys and the field values are the dictionary values.
       :param corpus_file_path: the path to the corpus_words file.
       :return: the corpus_words data splited into list of sentences,
                and the unique lemmas in the corpus_words
       """
    flag_end_sentence = False
    indeces_untagged_words = list()

    corpus_words = list()
    corpus_tags_gold = list()
    lemmas = set()

    corpus_words += ([START_PAD] * PADDING_START_END_length)
    corpus_tags_gold += ([START_PAD] * PADDING_START_END_length)

    with open(corpus_file_path, 'r', encoding='utf-8') as corpus_file:

        # read the file line by line
        # with this reading method we load little components of the corpus_words
        # and we preventing memory errors
        for line in tqdm(corpus_file):
            # delete '\n's in the end of the line
            if line[0] == "" or line[0] == "#":
                continue
            # if the line is a space line between sentences
            # move to the next sentence
            elif line == '\n':
                # append the sentence into the sentences matrix
                # this matrix contains the data about all the sentences
                # each sentence contains the data about its words
                if flag_end_sentence:
                    continue
                flag_end_sentence = True
                corpus_words += ([END_PAD] * PADDING_START_END_length)
                corpus_words += ([START_PAD] * PADDING_START_END_length)
                corpus_tags_gold += ([END_PAD] * PADDING_START_END_length)
                corpus_tags_gold += ([START_PAD] * PADDING_START_END_length)

            # a normal line containing a word fields
            else:
                # split the line into the filed values
                line = line.strip('\n')
                field_values = line.split('\t')

                corpus_words.append(field_values[field_values_dictionary["LEMMA"]])
                corpus_tags_gold.append(field_values[field_values_dictionary["POSTAG"]])
                lemmas.add(field_values[field_values_dictionary["LEMMA"]])
                flag_end_sentence = False
                if field_values[field_values_dictionary["POSTAG"]] in pos_tags_learning:
                    indeces_untagged_words.append(len(corpus_tags_gold) - 1 + start_index)

    return corpus_words[:-2], corpus_tags_gold[:-2], indeces_untagged_words, lemmas
