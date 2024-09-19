# neural_net.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import reduce

# Import configurations if needed
from config import DEVICE, PHONEME_VOCAB, PHONEME_GROUPS, PHONEME_TO_GROUP, MAX_SEQ_LEN

# =========================
# Helper Modules and Layers
# =========================

class WeightStandardizedConv1d(nn.Conv1d):
    """
    Weight Standardization for 1D convolutions.
    Reference: https://arxiv.org/abs/1903.10520
    Weight standardization works synergistically with group normalization.
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

def get_padding(kernel_size):
    return (kernel_size - 1) // 2

class ResidualConvBlock(nn.Module):
    """
    Standard ResNet-style residual convolutional block with Weight Standardization
    and Group Normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, gn=8):
        super(ResidualConvBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            WeightStandardizedConv1d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=get_padding(self.kernel_size),
            ),
            nn.GroupNorm(gn, out_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out

class FrequencyEncoding(nn.Module):
    """
    Frequency Encoding for time steps in diffusion models.
    Similar to positional encodings used in transformers.
    """

    def __init__(self, dim):
        super(FrequencyEncoding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # Shape: (batch_size, dim)

class ChannelAttention(nn.Module):
    """
    Channel Attention module.
    Applies channel-wise attention to the input tensor.
    """

    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 8, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# =======================
# U-Net Down and Up Blocks
# =======================

class UnetDown(nn.Module):
    """
    Downsampling block for U-Net architecture.
    Applies a Residual Convolutional Block followed by Max Pooling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)
        self.pool = nn.MaxPool1d(kernel_size=factor, stride=factor)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x

class UnetUp(nn.Module):
    """
    Upsampling block for U-Net architecture.
    Applies Upsampling followed by a Residual Convolutional Block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode='nearest')
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x

# =========================
# Main Model Implementations
# =========================

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net model for use in diffusion models.
    Incorporates time step embeddings using FrequencyEncoding.
    """

    def __init__(self, in_channels, features, kernel_size=5, gn=8, factor=2):
        super(ConditionalUNet, self).__init__()
        self.in_channels = in_channels
        self.dim = features
        self.time_emb = FrequencyEncoding(features)

        # Downsampling layers
        self.down1 = UnetDown(in_channels, features, kernel_size, gn=gn, factor=factor)
        self.down2 = UnetDown(features, features * 2, kernel_size, gn=gn, factor=factor)
        self.down3 = UnetDown(features * 2, features * 4, kernel_size, gn=gn, factor=factor)

        # Upsampling layers
        self.up3 = UnetUp(features * 4, features * 2, kernel_size, gn=gn, factor=factor)
        self.up2 = UnetUp(features * 4, features, kernel_size, gn=gn, factor=factor)
        self.up1 = UnetUp(features * 2, in_channels, kernel_size, gn=gn, factor=factor)

        # Output layer
        self.out_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x, t):
        """
        Forward pass of the Conditional U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).
            t (torch.Tensor): Time steps tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor.
        """
        # Compute time embeddings
        t_emb = self.time_emb(t).view(-1, self.dim, 1)
        combined_emb = t_emb.expand(-1, -1, x.size(2))  # Expand to match x's spatial dimensions

        # Apply channel attention
        x = self.channel_attention(x)

        # Downsampling path
        dn1 = self.down1(x)
        dn2 = self.down2(dn1 + combined_emb)
        dn3 = self.down3(dn2)

        # Upsampling path
        up3 = self.up3(dn3)
        up2 = self.up2(torch.cat([up3, dn2 + combined_emb], dim=1))
        up1 = self.up1(torch.cat([up2, dn1 + combined_emb], dim=1))

        # Output
        out = self.out_conv(torch.cat([up1, x], dim=1))
        return out

class Diffe(nn.Module):
    """
    U-Net variant for diffusion models.
    Incorporates time step embeddings and outputs intermediate representations.
    """

    def __init__(self, in_channels, features=512, kernel_size=5, gn=8, factor=2):
        super(Diffe, self).__init__()
        self.in_channels = in_channels
        self.dim = features
        self.time_emb = FrequencyEncoding(features)

        # Downsampling layers
        self.down1 = UnetDown(in_channels, features, kernel_size, gn=gn, factor=factor)
        self.down2 = UnetDown(features, features * 2, kernel_size, gn=gn, factor=factor)
        self.down3 = UnetDown(features * 2, features * 4, kernel_size, gn=gn, factor=factor)

        # Upsampling layers
        self.up3 = UnetUp(features * 4, features * 2, kernel_size, gn=gn, factor=factor)
        self.up2 = UnetUp(features * 4, features, kernel_size, gn=gn, factor=factor)
        self.up1 = UnetUp(features * 2, in_channels, kernel_size, gn=gn, factor=factor)

        # Output layer
        self.out_conv = nn.Conv1d(in_channels * 3, in_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x, t):
        """
        Forward pass of the Diffe model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).
            t (torch.Tensor): Time steps tensor of shape (batch_size,).

        Returns:
            tuple: Output tensor, intermediate features, and latent representation.
        """
        # Compute time embeddings
        t_emb = self.time_emb(t).view(-1, self.dim, 1)
        combined_emb = t_emb.expand(-1, -1, x.size(2))

        # Apply channel attention
        x = self.channel_attention(x)

        # Downsampling path
        dn1 = self.down1(x)
        dn2 = self.down2(dn1)
        dn3 = self.down3(dn2)

        # Upsampling path
        up3 = self.up3(dn3)
        up2 = self.up2(torch.cat([up3, dn2], dim=1))
        up1 = self.up1(torch.cat([up2, dn1], dim=1))

        # Output
        out = self.out_conv(torch.cat([up1, x, combined_emb], dim=1))

        # Latent representation
        z = torch.mean(dn3, dim=2)

        return out, dn3, z

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        nn_model (nn.Module): Neural network model to use.
        schedule (dict): Dictionary containing schedule parameters.
        n_T (int): Number of timesteps.
        device (torch.device): Device to run the model on.
    """

    def __init__(self, nn_model, schedule, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.schedule = schedule
        self.n_T = n_T
        self.device = device

    def forward(self, x):
        """
        Forward pass of the DDPM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output of the model, noisy input, noise, and time embeddings.
        """
        batch_size = x.size(0)
        _ts = torch.randint(1, self.n_T, (batch_size,), device=self.device)
        t = _ts / self.n_T

        noise = torch.randn_like(x, device=self.device)
        sqrtab = self.schedule["sqrtab"][_ts].view(-1, 1, 1)
        sqrtmab = self.schedule["sqrtmab"][_ts].view(-1, 1, 1)
        x_t = sqrtab * x + sqrtmab * noise

        output = self.nn_model(x_t, t)
        return output, x_t, noise, t

    def sample(self, noise, num_steps=None):
        """
        Samples from the model.

        Args:
            noise (torch.Tensor): Initial noise tensor.
            num_steps (int, optional): Number of steps to run the reverse diffusion process.

        Returns:
            torch.Tensor: Generated sample.
        """
        self.nn_model.eval()
        x_t = noise
        num_steps = num_steps or self.n_T

        with torch.no_grad():
            for t in reversed(range(num_steps)):
                batch_size = x_t.size(0)
                t_tensor = torch.full((batch_size,), t / self.n_T, device=self.device)
                noise_pred = self.nn_model(x_t, t_tensor)

                # Reverse process
                if t > 0:
                    z = torch.randn_like(x_t, device=self.device)
                else:
                    z = torch.zeros_like(x_t, device=self.device)

                alpha_t = self.schedule["alpha_t"][t]
                beta_t = self.schedule["beta_t"][t]
                alphabar_t = self.schedule["alphabar_t"][t]

                x_t = (
                    (1 / torch.sqrt(alpha_t)) *
                    (x_t - (beta_t / torch.sqrt(1 - alphabar_t)) * noise_pred)
                    + torch.sqrt(beta_t) * z
                )
        return x_t

class EEGPhonemeSeq2Seq(nn.Module):
    """
    Sequence-to-Sequence model for EEG to Phoneme translation.

    Args:
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Hidden size of the GRU layers.
        output_size (int): Size of the output vocabulary.
        num_layers (int): Number of layers in GRU.
        max_output_length (int): Maximum length of the output sequence.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, max_output_length=10):
        super(EEGPhonemeSeq2Seq, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_output_length = max_output_length

        # Encoder: Bidirectional GRU
        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)

        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_size * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Transform decoder output to match decoder input
        self.output_to_decoder_input = nn.Linear(output_size, hidden_size)

    def forward(self, x):
        """
        Forward pass of the Seq2Seq model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            tuple: Output logits and hidden states.
        """
        batch_size = x.size(0)

        # Permute x to (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)

        # Encode
        encoder_output, hidden = self.encoder_gru(x)
        # Combine bidirectional hidden states
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size).sum(dim=1)

        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)

        outputs = []
        hidden_states = []

        eos_token_index = self.output_size - 1  # Assuming last index is EOS token

        for t in range(self.max_output_length):
            # Attention
            attention_scores = self.attention(encoder_output)
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_output)
            context_vector = context_vector.mean(dim=1, keepdim=True)

            # Decoder input
            rnn_input = torch.cat((context_vector, decoder_input), dim=2)

            # Decode
            output, hidden = self.decoder_gru(rnn_input, hidden)

            # Store hidden state
            hidden_states.append(hidden[-1].squeeze(0))

            # Output layer
            output_logits = self.fc(output.squeeze(1))
            outputs.append(output_logits)

            # Check for EOS token
            if (output_logits.argmax(dim=1) == eos_token_index).all():
                break

            # Prepare next decoder input
            decoder_input = self.output_to_decoder_input(output_logits).unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)

        return outputs, hidden_states

class EEGNet(nn.Module):
    def __init__(self, output_features=128):
        super(EEGNet, self).__init__()
        
        # First convolutional block
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(64, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # Depthwise Separable Convolutions
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # Adjusting pooling to keep more of the sequence length
            nn.AvgPool2d(kernel_size=(1, 2)),  # Adjusted pooling
            nn.Dropout(0.5)
        )

        # Second Separable Convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # Adjusting pooling to keep more of the sequence length
            nn.AvgPool2d(kernel_size=(1, 4)),  # Adjusted pooling
            nn.Dropout(0.5)
        )

        # Output features for dn30: (batch_size, output_features=128, seq_length=125)
        self.output_features = output_features

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)

        # Remove extra dimensions and reshape for dn30 output
        x = x.squeeze(-1)  # Assuming the second last dimension needs to be squeezed
        x = x.view(x.size(0), self.output_features, -1)  # (batch_size, features=128, seq_length)

        return x