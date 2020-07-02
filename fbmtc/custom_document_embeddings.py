import logging
from typing import List, Union

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

import flair
from flair.data import Sentence
from flair.embeddings.token import TokenEmbeddings, StackedEmbeddings
from flair.embeddings.document import DocumentEmbeddings
from flair.nn import LockedDropout, WordDropout

log = logging.getLogger("flair")


class CustomDocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            hidden_size=128,
            rnn_layers=1,
            reproject_words: bool = True,
            reproject_words_dimension: int = None,
            bidirectional: bool = True,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            fine_tune: bool = True,
            attention_size=100
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False if fine_tune else True

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        self.attention_size = attention_size

        # Word level encoder
        self.rnn = torch.nn.GRU(
            self.embeddings_dimension,
            hidden_size,
            num_layers=rnn_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # Word level attention network (Weights, bias)
        if self.bidirectional:
            self.word_attention = torch.nn.Linear(2 * hidden_size, self.attention_size)
        else:
            self.word_attention = torch.nn.Linear(hidden_size, self.attention_size)
        self.word_context_vector = torch.nn.Linear(self.attention_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.name = "document_gru"

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = (
            LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        )
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push through RNN
        packed = pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        rnn_out, hidden = self.rnn(packed)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(rnn_out.data)  # (n_words, att_size) weights multiplied with hidden state + bias
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)
        # First, take the exponent
        # TODO check whether subtracting max value from att_w makes sense
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)
        packed_att_w = PackedSequence(data=att_w,
                                      batch_sizes=rnn_out.batch_sizes,
                                      sorted_indices=rnn_out.sorted_indices,
                                      unsorted_indices=rnn_out.unsorted_indices)
        att_w, output_lengths = pad_packed_sequence(packed_att_w,
                                                    batch_first=True)  # (n_sentences, max(words_per_sentence))
        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        outputs, _ = pad_packed_sequence(rnn_out,
                                         batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        outputs = outputs * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        outputs = outputs.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        # after-RNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract sentence embeddings
        for sentence_no, length in enumerate(lengths):
            embedding = outputs[sentence_no]

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):
        major, minor, build, *_ = (int(info)
                                   for info in torch.__version__.replace("+", ".").split('.') if info.isdigit())

        # fixed RNN change format for torch 1.4.0
        if major >= 1 and minor >= 4:
            for child_module in self.children():
                if isinstance(child_module, torch.nn.RNNBase):
                    _flat_weights_names = []

                    if child_module.__dict__["bidirectional"]:
                        num_direction = 2
                    else:
                        num_direction = 1
                    for layer in range(child_module.__dict__["num_layers"]):
                        for direction in range(num_direction):
                            suffix = "_reverse" if direction == 1 else ""
                            param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                            if child_module.__dict__["bias"]:
                                param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                            param_names = [
                                x.format(layer, suffix) for x in param_names
                            ]
                            _flat_weights_names.extend(param_names)

                    setattr(child_module, "_flat_weights_names",
                            _flat_weights_names)

                child_module._apply(fn)

        else:
            super()._apply(fn)
