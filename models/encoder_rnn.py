import torch.nn as nn

from .base_rnn import BaseRNN


class EncoderRNN(BaseRNN):
    """Applies a multi-layer RNN to an input sequence.

    Inputs: inputs, input_lengths
        - **inputs**: List of sequences, whose length is the batch size
            and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): List that contains
            the lengths of sequences in the mini-batch, it must be
            provided when using variable length RNN (default: `None`).

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): Tensor containing the
            encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): Tensor
            containing the features in the hidden state `h`

    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='lstm', variable_lengths=False):
        """Constructor for EncoderRNN.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_len (int): A maximum allowed length for the sequence to be
                processed.
            hidden_size (int): The number of features in the hidden state `h`.
            input_dropout_p (float, optional): Dropout probability for the input
                sequence (default: 0).
            dropout_p (float, optional): Dropout probability for the output
                sequence (default: 0).
            n_layers (int, optional): Number of recurrent layers (default: 1).
            bidirectional (bool, optional): if True, becomes a bidirectional
                encoder (defulat False).
            rnn_cell (str, optional): Type of RNN cell (default: gru).
            variable_lengths (bool, optional): If use variable length
                RNN (default: False).
        """
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout_p)
        self.init_weights()

    def init_weights(self):
        """Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_var, input_lengths=None, h0=None):
        """Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): Tensor containing the features of
                the input sequence.
            input_lengths (list of int, optional): A list that contains
                the lengths of sequences in the mini-batch.
            h0 : Tensor containing initial hidden state.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): Variable containing
                the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size):
                Variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded, h0)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=True)
        return output, hidden
