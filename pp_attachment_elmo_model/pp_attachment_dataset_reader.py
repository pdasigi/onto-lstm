"""
Dataset reader for PPAttachment prediction task.
"""

from typing import Dict, List

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import TextField, IndexField
from allennlp.data import Token, Instance


@DatasetReader.register("pp_attachment")
class PPAttachmentReader(DatasetReader):
    """
    Reads in processed PPAttachment data. Assumes a tsv file where each line has two columns, the
    first is a number indicating the index of the true head word, and the second is a phrase, where
    each word is POS tagged. We'll ignore the POS tags for now.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        for line in open(file_path):
            head_index_string, phrase = line.split("\t")
            # The indices in the file start from 1.
            head_index = int(head_index_string) - 1
            phrase_tokens = [Token(t.split("_")[0]) for t in phrase.split()]
            instance = self.text_to_instance(phrase_tokens, head_index)
            yield instance

    def text_to_instance(self,
                         tokens: List[Token],
                         head_index: int) -> Instance:
        # pylint: disable=arguments-differ
        phrase_field = TextField(tokens, token_indexers=self._token_indexers)
        head_index_field = IndexField(head_index, phrase_field)
        fields = {"phrase": phrase_field, "correct_head_index": head_index_field}
        return Instance(fields)
