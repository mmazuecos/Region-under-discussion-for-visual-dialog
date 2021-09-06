import json

import h5py
import numpy as np
import os
from torch.utils.data import Dataset

from utils.create_subset import create_subset
from utils.datasets.SL.prepro import create_data_file


class N2NDataset(Dataset):
    def __init__(self, split='train', num_turns=None, complete_only=False, **kwargs):
        self.data_args = kwargs

        visual_feat_file = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        tmp_key = split + "_process_file"

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_'+split+'_successful_data.json'
            else:
                data_file_name = 'n2n_'+split+'_all_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(
                data_dir=self.data_args['data_dir'],
                data_file=self.data_args['data_paths'][split],
                data_args=self.data_args,
                vocab_file_name=self.data_args['data_paths']['vocab_file'],
                split=split
            )

        if self.data_args['my_cpu']:
            if not os.path.isfile(os.path.join(self.data_args['data_dir'], 'subset_'+split+'.json')):
                create_subset(data_dir=self.data_args['data_dir'], dataset_file_name=data_file_name, split=split)

        if self.data_args['my_cpu']:
            with open(os.path.join(self.data_args['data_dir'], 'subset_'+split+'.json'), 'r') as f:
                self.n2n_data = json.load(f)
        else:
            with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
                self.n2n_data = json.load(f)

        if num_turns:
            print("Taking only dialogs having {} turns...".format(num_turns))
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if sum([1 for x in v["history_q_lens"] if x != 0]) == num_turns + 1:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        if complete_only:
            print("Taking only complete dialogs...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if v["decider_tgt"] == 1:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        # elif filter == "two_or_more_turns":
        #     print("Taking all turns into account!")
        #     filtered_n2n_data = {}
        #     _id = 0
        #     for k, v in self.n2n_data.items():
        #         if sum([1 for x in v["history_q_lens"] if x != 0]) >=2:
        #             filtered_n2n_data[str(_id)] = v
        #             _id += 1
        #     self.n2n_data = filtered_n2n_data

    def __len__(self):
        return len(self.n2n_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        # Load image features
        image_file = self.n2n_data[idx]['image_file']
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['image'] = ImgFeat
        _data['history'] = np.asarray(self.n2n_data[idx]['history'])
        _data['history_len'] = self.n2n_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.n2n_data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.n2n_data[idx]['target_q'])
        _data['tgt_len'] = self.n2n_data[idx]['tgt_len']
        _data['decider_tgt'] = int(self.n2n_data[idx]['decider_tgt'])
        _data['objects'] = np.asarray(self.n2n_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.n2n_data[idx]['objects'], np.zeros(len(self.n2n_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.n2n_data[idx]['spatials'])
        _data['target_obj'] = self.n2n_data[idx]['target_obj']
        _data['target_cat'] = self.n2n_data[idx]['target_cat']
        _data['game_id'] = self.n2n_data[idx]['game_id']
        _data['bboxes'] = np.asarray(self.n2n_data[idx]['bboxes'])
        _data['image_url'] = self.n2n_data[idx]['image_url']

        return _data
