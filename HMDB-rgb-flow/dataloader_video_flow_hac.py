from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import os
import imageio.v3 as iio


class HACDOMAIN(torch.utils.data.Dataset):
    def __init__(self, version='v2',  modality='rgb', cfg=None, cfg_flow=None, datapath='/cluster/work/ibk_chatzi/hao/CharadesEgo/',appen=''):
        self.base_path = datapath
        self.video_list = []
        self.prefix_list = []
        self.label_list = []

        for domain in ['human', 'animal', 'cartoon']:
            if domain == 'animal':
                prefix = 'ActorShift/'
            elif domain == 'human':
                prefix = 'kinetics600/'
            else:
                prefix = 'cartoon/'
            with open(self.base_path + "splits/%s/ActorShift_test_only_%s.csv" % (version, domain)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

            with open(self.base_path + "splits/%s/ActorShift_train_only_%s.csv" % (version, domain)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

        self.domain = domain
        self.modality = modality
        self.appen = appen

        val_pipeline = cfg.data.val.pipeline
        self.pipeline = Compose(val_pipeline)
        val_pipeline_flow = cfg_flow.data.val.pipeline
        self.pipeline_flow = Compose(val_pipeline_flow)
        self.train = False

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.interval = 9
        self.video_path_base = self.base_path + 'ActorShift/'
        if not os.path.exists(self.video_path_base):
            os.mkdir(self.video_path_base)

    def __getitem__(self, index):
        label1 = int(self.label_list[index])
        video_path = self.video_path_base + self.video_list[index] + "/" 
        video_path = video_path + self.video_list[index] + '-'

        video_file = self.base_path + self.prefix_list[index] +'videos/' + self.video_list[index]
        vid = iio.imread(video_file, plugin="pyav")

        frame_num = vid.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.pipeline(data)

        video_file_x = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_x.mp4'
        video_file_y = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_y.mp4'
        
        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")

        frame_num = vid_x.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
        flow = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index_flow,
            video=vid_x,
            video_y=vid_y,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow, frame_inds_flow = self.pipeline_flow(flow)

        return data, flow, label1

    def __len__(self):
        return len(self.video_list)


