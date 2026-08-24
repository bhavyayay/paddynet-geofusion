"""
model.py

A small Conv1D -> LSTM -> Dense architecture for yield prediction,
following the same architectural style as the original research paper
(Conv1D feature extraction -> LSTM temporal/pattern modeling -> Dense
output) but SCALED DOWN substantially.

WHY SCALED DOWN:
The reference paper trained a 281,489-parameter CNN-LSTM on ~40 years
of dense historical records. This project currently has ~27 real rows
of data. A model with anywhere near that many parameters would simply
memorize the 27 rows (severe overfitting) rather than learn anything
generalizable. This architecture intentionally uses very few filters/
units -- on the order of a few hundred to low thousands of parameters
-- to have any realistic chance of generalizing at this data size.

This should be treated as a PIPELINE PROOF-OF-CONCEPT: it demonstrates
the full architecture and training loop work correctly end-to-end. Its
accuracy numbers should not be treated as trustworthy or production-
ready until trained on substantially more real data.
"""

import torch
import torch.nn as nn


class SmallCNNLSTM(nn.Module):
    """
    Input shape expected: (batch_size, n_features, 1)
      -- we treat each row's features as a length-1 "sequence" per
      feature channel, matching the Conv1D-over-features pattern the
      reference paper used, since our current data doesn't have true
      per-region time series long enough for a real temporal window.

    Architecture:
      Conv1D (small) -> ReLU -> LSTM (small) -> Dense -> single yield output
    """

    def __init__(self, n_features: int, conv_channels: int = 8,
                 lstm_hidden: int = 8, dropout: float = 0.3):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=conv_channels,
            kernel_size=1  # kernel=1 since our "sequence length" per
                            # feature is 1 -- this is really acting as
                            # a learned per-feature transform, not a
                            # true temporal convolution, given the data
        )
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, n_features, 1) -> conv over the "1" dim per feature
        # Reshape to (batch, 1, n_features) so Conv1d treats n_features
        # as the sequence dimension it convolves across.
        x = x.permute(0, 2, 1)  # (batch, 1, n_features)
        x = self.conv1d(x)      # (batch, conv_channels, n_features)
        x = self.relu(x)
        x = self.dropout1(x)

        x = x.permute(0, 2, 1)  # (batch, n_features, conv_channels) for LSTM
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        last_step = self.dropout2(last_step)

        out = self.fc(last_step)  # (batch, 1)
        return out


def build_model(n_features: int) -> SmallCNNLSTM:
    model = build = SmallCNNLSTM(n_features=n_features)
    n_params = sum(p.numel() for p in build.parameters() if p.requires_grad)
    print(f"Built SmallCNNLSTM with {n_params:,} trainable parameters "
          f"(reference paper used 281,489 -- this is intentionally much "
          f"smaller given the current dataset size).")
    return build


if __name__ == "__main__":
    # Quick smoke test -- confirms the architecture runs without a
    # real dataset attached yet.
    model = build_model(n_features=5)
    dummy_input = torch.randn(4, 5, 1)  # batch of 4, 5 features
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect (4, 1)
