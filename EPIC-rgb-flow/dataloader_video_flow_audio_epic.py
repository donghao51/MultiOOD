from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', eval=False, modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, far_ood=False, datapath=''):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.eval = eval

        if split == 'train' and not eval:
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
            train_pipeline_flow = cfg_flow.data.train.pipeline
            self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)
            val_pipeline_flow = cfg_flow.data.val.pipeline
            self.pipeline_flow = Compose(val_pipeline_flow)

        data1 = []
        if far_ood:
            splits = ['train', 'val', 'test']
            for spl in splits:
                train_file_name = "splits/D3_" + spl + ".pkl"
                train_file = pd.read_pickle(train_file_name)
                for line in train_file:
                    data1.append((line[0], line[1], line[2], line[3], line[4], line[5]))
        else:
            train_file_name = "splits/D3_" + split + "_near_ood.pkl"
            train_file = pd.read_pickle(train_file_name)
            for line in train_file:
                data1.append((line[0], line[1], line[2], line[3], line[4], line[5]))

        self.samples = data1
        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __getitem__(self, index):
        if self.samples[index][0] in ['P22_01', 'P22_02', 'P22_03', 'P22_04']:
            video_path = self.base_path +'rgb/test/D3/'+self.samples[index][0]
            flow_path = self.base_path + 'flow/test/D3/'+self.samples[index][0]
            audio_path = self.base_path + 'rgb/test/D3/' + self.samples[index][0] + '.wav'
        else:
            video_path = self.base_path +'rgb/train/D3/'+self.samples[index][0]
            flow_path = self.base_path + 'flow/train/D3/'+self.samples[index][0]
            audio_path = self.base_path + 'rgb/train/D3/' + self.samples[index][0] + '.wav'
        
        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.samples[index][2] - self.samples[index][1]),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data = self.pipeline(data)

        filename_tmpl_flow = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality_flow = self.cfg_flow.data.train.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples[index][1] / 2)))
        flow = dict(
            frame_dir=flow_path,
            total_frames=int((self.samples[index][2] - self.samples[index][1])/2),
            label=-1,
            start_index=start_index_flow,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow = self.pipeline_flow(flow)

        label1 = self.samples[index][-1]

        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate

        fr_sec = self.samples[index][3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = self.samples[index][4].split(':')
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        start1 = fr_sec / duration * len(samples)
        end1 = stop_sec / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        dur = int(self.sample_dur * 16000)
        resamples = samples[:dur]
        while len(resamples) < dur:
            resamples = np.tile(resamples, 10)[:dur]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        if self.split == 'train' and not self.eval:
            noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
            start1 = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start1:(start1 + self.interval), :] = 0


        return data, flow, spectrogram.astype(np.float32), label1


    def __len__(self):
        return len(self.samples)

