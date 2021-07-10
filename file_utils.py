from tqdm import tqdm
from constants import START_PAD, END_PAD, field_values_dictionary




def read_data(corpus_file_path, PADDING_START_END_length, pos_tags_learning):
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

    corpus_words = list()
    corpus_tags = list()

    corpus_words += ([START_PAD] * PADDING_START_END_length)
    corpus_tags += ([START_PAD] * PADDING_START_END_length)

    with open(corpus_file_path, 'r', encoding='utf-8') as corpus_file:
        sentence = list()

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
                corpus_tags += ([END_PAD] * PADDING_START_END_length)
                corpus_tags += ([START_PAD] * PADDING_START_END_length)

            # a normal line containing a word fields
            else:
                # split the line into the filed values
                line = line.strip('\n')
                field_values = line.split('\t')

                corpus_words.append(field_values[field_values_dictionary["LEMMA"]])
                corpus_tags.append(field_values[field_values_dictionary["POSTAG"]])
                flag_end_sentence = False


    return corpus_words[:-2], corpus_tags[:-2]
