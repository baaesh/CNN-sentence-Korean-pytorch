from typing import Dict, List

from overrides import overrides

from allennlp.common.tqdm import Tqdm
from allennlp.data import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer

from konlpy.tag import Okt


@Tokenizer.register("morph")
class MorphTokenizer(Tokenizer):

    @overrides
    def tokenize(self, text: str):
        okt = Okt()
        morphs = okt.morphs(text)
        sentiment = morphs.pop()
        idx = morphs.pop(0)
        sentence_len = len(morphs)
        return list(map(Token, morphs)), sentence_len, sentiment


@Tokenizer.register("bpe")
class BpeTokenizer(Tokenizer):

    @overrides
    def tokenize(self, text: str):
        sentence = []
        words = text.split()
        sentiment = words.pop()
        for i, word in enumerate(words):
            if i == 0:
                pass
            else:
                sentence.append(' ')
            tokens = word.split('@@')
            sentence += tokens
        sentence_len = len(sentence)
        return list(map(Token, sentence)), sentence_len, sentiment


@DatasetReader.register("Kor")
class KorDatasetReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 min_len: int = 5,
                 max_len: int = 100,
                 tqdm: bool = True) -> None:
        super(KorDatasetReader, self).__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or\
                               {"tokens":
                                    SingleIdTokenIndexer()}
        self._min_len = min_len
        self._max_len = max_len
        self._tqdm = tqdm

    def set_total_instances(self, file_path):
        with open(file_path, 'r', encoding="utf8") as data_file:
            self.total_instances = len(data_file.readlines())

    def _read(self, file_path):
        with open(file_path, 'r', encoding="utf8") as data_file:
            lines = data_file.readlines()
            self.total_instances = len(lines)
            if self._tqdm:
                lines = Tqdm.tqdm(lines)
            for line_num, line in enumerate(lines):
                line = line.strip("\n")
                if not line:
                    continue
                tokenized_sentence, sentence_len, sentiment = \
                    self._tokenizer.tokenize(line)
                if sentence_len > self._max_len or sentence_len < self._min_len:
                    continue
                yield self.text_to_instance(tokenized_sentence, sentiment)

    def text_to_instance(self, tokenized_sentence, sentiment) -> Instance:
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        label_field = LabelField(sentiment)
        fields = {'sentence': sentence_field, 'label': label_field}
        return Instance(fields)