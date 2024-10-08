import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(13, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        return self.sigmoid(output)