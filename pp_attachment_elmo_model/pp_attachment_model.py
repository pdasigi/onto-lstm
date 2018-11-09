from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("pp_attachment")
class PPAttachmentModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 phrase_embedder: TextFieldEmbedder,
                 phrase_encoder: Seq2SeqEncoder,
                 predictor_feedforward: FeedForward,
                 embedding_dropout: float = 0.0,
                 encoder_dropout: float = 0.0) -> None:
        super().__init__(vocab)
        self._phrase_embedder = phrase_embedder
        self._phrase_encoder = phrase_encoder
        self._predictor_feedforward = predictor_feedforward
        self._accuracy = CategoricalAccuracy()
        self._embedding_dropout = embedding_dropout
        self._encoder_dropout = encoder_dropout
        assert self._predictor_feedforward.get_output_dim() == 1, \
              "The predictor feedforward module should output a single logit!"

    def forward(self,
                phrase: Dict[str, torch.LongTensor],
                correct_head_index: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_phrase = self._phrase_embedder(phrase)
        if self._embedding_dropout > 0.0:
            embedded_phrase = torch.nn.functional.dropout(embedded_phrase, self._embedding_dropout,
                                                          self.training)
        batch_size, phrase_length, _ = embedded_phrase.shape
        num_heads = phrase_length - 2
        phrase_mask = get_text_field_mask(phrase)
        encoded_phrase = self._phrase_encoder(embedded_phrase, phrase_mask)
        if self._encoder_dropout > 0.0:
            encoded_phrase = torch.nn.functional.dropout(encoded_phrase, self._encoder_dropout,
                                                         self.training)
        _, _, encoding_dim = encoded_phrase.shape
        if self._phrase_encoder.is_bidirectional():
            # The outputs are concatentated from the encoders running in both directions. We need to
            # sum them.
            encoding_dim = encoding_dim // 2
            encoded_phrase = encoded_phrase[:, :, :encoding_dim] + encoded_phrase[:, :, encoding_dim:]
        # Length of mask - 1 is the index of the last tokens (i.e., the child), and PPs are always
        # two tokens. So preposition's index is len(mask) - 2.
        child_indices = phrase_mask.sum(1).long() - 1
        preposition_indices = child_indices - 1
        head_indices_list = []
        for child_index in child_indices.cpu().data:
            current_head_indices = []
            for i in range(phrase_length):
                if i not in [child_index, child_index - 1]:
                    current_head_indices.append(i)
            head_indices_list.append(current_head_indices)
        head_indices = encoded_phrase.new_tensor(head_indices_list, dtype=torch.long)
        # (batch_size, num_heads, encoding_dim
        child_indices = child_indices.view(-1, 1, 1).expand(batch_size, num_heads, encoding_dim)
        encoded_children = torch.gather(encoded_phrase, 1, child_indices)
        # (batch_size, num_heads, encoding_dim)
        preposition_indices = preposition_indices.view(-1, 1, 1).expand(batch_size,
                                                                        num_heads, encoding_dim)
        encoded_prepositions = torch.gather(encoded_phrase, 1, preposition_indices)
        # (batch_size, num_heads, encoding_dim)
        expanded_head_indices = head_indices.view(batch_size, num_heads, 1).expand(batch_size,
                                                                                   num_heads, encoding_dim)
        encoded_heads = torch.gather(encoded_phrase, 1, expanded_head_indices)
        head_mask = torch.gather(phrase_mask, 1, head_indices)

        # (batch_size, num_heads, encoding_dim * 3)
        prediction_input = torch.cat((encoded_heads, encoded_prepositions, encoded_children), 2)

        # (batch_size, num_heads, 1)
        logits = self._predictor_feedforward(prediction_input)
        logits = logits.squeeze(2)  # (batch_size, num_heads)
        probabilities = masked_softmax(logits, head_mask)
        correct_head_logprobs = torch.log(torch.gather(probabilities, 1, correct_head_index))
        self._accuracy(predictions=probabilities, gold_labels=correct_head_index.squeeze(-1))
        loss = - torch.mean(correct_head_logprobs)
        return {"loss": loss}

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self._accuracy.get_metric(reset=reset)}
