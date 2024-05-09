# coding: utf-8
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from utils import get_data_path_list


class TextCleaner:
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    def __init__(self, dummy=None):
        symbols = (
            [self._pad]
            + list(self._punctuation)
            + list(self._letters)
            + list(self._letters_ipa)
        )
        self.word_index_dictionary = {symbol: i for i, symbol in enumerate(symbols)}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        root_path,
        sr=24000,
        data_augmentation=False,
        validation=False,
        OOD_data="Data/OOD_texts.txt",
        min_length=50,
    ):
        np.random.seed(1)
        _data_list = [line.strip().split("|") for line in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(OOD_data, "r", encoding="utf-8") as f:
            tl = f.readlines()
        idx = 1 if ".wav" in tl[0].split("|")[0] else 0
        self.ptexts = [t.split("|")[idx] for t in tl]

        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

        wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = self._preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])

        # get OOD text

        ps = ""

        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]

            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)

        return (
            speaker_id,
            acoustic_feature,
            text_tensor,
            ref_text,
            ref_mel_tensor,
            ref_label,
            path,
            wave,
        )

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(os.path.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = self.text_cleaner(text)

        text.insert(0, 0)
        text.append(0)

        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = self._preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]

        return mel_tensor, speaker_id

    def _preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        to_mel = MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor


class Collater(object):
    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ["" for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        for bid, (
            label,
            mel,
            text,
            ref_text,
            ref_mel,
            ref_label,
            path,
            wave,
        ) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref_labels[bid] = ref_label
            waves[bid] = wave

        return (
            waves,
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            output_lengths,
            ref_mels,
        )


def get_dataloaders(dataset_config, batch_size, num_workers, device):
    train_path = dataset_config["train_data"]
    val_path = dataset_config["val_data"]
    min_length = dataset_config["min_length"]
    OOD_data = dataset_config["OOD_data"]
    root_path = dataset_config["root_path"]

    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
    )

    return train_dataloader, val_dataloader, train_list


def build_dataloader(
    path_list,
    root_path,
    validation=False,
    OOD_data="Data/OOD_texts.txt",
    min_length=50,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
):
    dataset = FilePathDataset(
        path_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        **dataset_config
    )
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader
