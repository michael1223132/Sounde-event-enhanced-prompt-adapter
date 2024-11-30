import sys
import os

import torch
from torch import nn
import torchaudio
import pdb
# from TTS.utils.audio.torch_transforms import TorchSTFT
from .base_encoder import BaseEncoder
from .shared_configs import BaseAudioConfig


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetSpeakerEncoder(BaseEncoder):
    """Implementation of the model H/ASP without batch normalization in speaker embedding. This model was proposed in: https://arxiv.org/abs/2009.14153
    Adapted from: https://github.com/clovaai/voxceleb_trainer
    """

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config)
        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # pylint: disable=R0201
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
    

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        x.squeeze_(1)
        # if you torch spec compute it otherwise use the mel spec computed by the AP
        if self.use_torch_spec:
            x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

def read_wav_file(file_path, target_length=16000, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if target_length:
        if waveform.size(1) < target_length:
            padding = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_length]
    return waveform

def main():
    # 定义模型参数
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dim",
        type= int,
        required=False,
        default=80
    )
    parser.add_argument(
        "-p",
        "--proj_dim",
        type=int,
        required=False,
        default=1024
    )
    parser.add_argument(
        "-u",
        "--use_torch_spec",
        type=bool,
        required=False,
        default=True
    )
    args = parser.parse_args()
    audio_config = BaseAudioConfig(
    mel_fmin=50,
    mel_fmax=8000,
    hop_length=256,
    stats_path="scale_stats.npy",
)

    model = ResNetSpeakerEncoder(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        use_torch_spec= True,
        audio_config= audio_config,
    )
    model.eval()

    # 读取并预处理 WAV 文件
    file_paths = [
        "/data4/xiongchenxu/tango/tango/outputs/1721791367_data4_xiongchenxu_tango_tango_saved_1721788470_steps_200_guidance_3.0/output_0.wav",
        "/data4/xiongchenxu/tango/tango/outputs/1721791367_data4_xiongchenxu_tango_tango_saved_1721788470_steps_200_guidance_3.0/output_1.wav"
    ]
    target_length = 32000  # 1 秒钟的音频，对于 16kHz 的采样率

    waveforms = [read_wav_file(file_path, target_length=target_length) for file_path in file_paths]
    waveforms = torch.stack(waveforms)  # (batch_size, 1, T_in)
    
    
    embedding = model(waveforms, l2_norm=True)
    print(embedding)
if __name__ == "__main__":
    main()
