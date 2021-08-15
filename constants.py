"""This file contains constant values we use."""
from typing import Dict, Tuple


START_PAD: str = "SSSTARTTT"
END_PAD: str = "EEENDDD"


field_values_dictionary: Dict[str, int] = {
    'STEM': 1,
    'LEMMA': 2,
    'POSTAG': 4
}


pos_tags_list: Tuple[str, ...] = (
'JJ', '_', 'NNP', 'VBN', 'NNS', 'RB', 'VBD', 'JJS', 'RBR', 'PRP$', 'RBS', 'EEENDDD', 'WP$', '.', 'VBP', ':', 'JJR',
'WRB', 'TO', "''", 'CD', ',', 'POS', 'VB', 'IN', 'DT', 'PRP', 'NNPS', 'UH', 'CC', '-LRB-', 'VBG', 'EX', 'WDT', 'MD',
'RP', '-RRB-', 'VBZ', 'SYM', 'FW', 'PDT', 'WP', 'NN', '``', 'LS', 'SSSTARTTT'
)





""""
field_names = ['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG',
               'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
"""