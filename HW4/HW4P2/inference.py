#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/19/2024 7:51 PM
# @Author  : Loading
import gc
import os
import csv
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.SpeechDataset import SpeechDataset
from datasets.Verify import verify_dataset
from models.myTransformer import Transformer
from utils.mytokenizer import GTokenizer
from utils.train_val import validate_step, test_step
from utils.misc import load_checkpoint


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path) as file:
        return yaml.safe_load(file)


def prepare_data(config: dict, tokenizer: GTokenizer):
    ##Tokenizer##
    Tokenizer = GTokenizer(config['token_type'])
    """加载验证集和测试集数据"""
    train_dataset = SpeechDataset(partition=config['train_partition'], config=config, tokenizer=Tokenizer,
                                  isTrainPartition=True)
    val_dataset = SpeechDataset(config['val_partition'], config, tokenizer, isTrainPartition=False)
    test_dataset = SpeechDataset(config['test_partition'], config, tokenizer, isTrainPartition=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=test_dataset.collate_fn)
    print("\n\nVerifying Datasets")
    max_train_feat, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
    max_val_feat, max_val_transcript = verify_dataset(val_loader, config['val_partition'])
    max_test_feat, max_test_transcript = verify_dataset(test_loader, config['test_partition'])
    # _, max_text_transcript               = verify_dataset(text_loader,  config['unpaired_text_partition'])

    MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
    MAX_TRANS_LEN = max(max_train_transcript, max_val_transcript)
    print(f"Maximum Feat. Length in Entire Dataset      : {MAX_SPEECH_LEN}")
    print(f"Maximum Transcript Length in Entire Dataset : {MAX_TRANS_LEN}")
    print('')
    gc.collect()

    return val_loader, test_loader, MAX_SPEECH_LEN, MAX_TRANS_LEN


def build_model(config: dict, tokenizer: GTokenizer, max_speech_len: int, max_trans_len: int, device: str):
    """初始化模型"""
    model = Transformer(
        input_dim=config['num_feats'],
        time_stride=config['time_stride'],
        feature_stride=config['feature_stride'],
        embed_dropout=config['embed_dropout'],
        d_model=config['d_model'],
        enc_num_layers=config['enc_num_layers'],
        enc_num_heads=config['enc_num_heads'],
        speech_max_len=max_speech_len,
        enc_dropout=config['enc_dropout'],
        dec_num_layers=config['dec_num_layers'],
        dec_num_heads=config['dec_num_heads'],
        d_ff=config['d_ff'],
        dec_dropout=config['dec_dropout'],
        target_vocab_size=tokenizer.VOCAB_SIZE,
        trans_max_len=max_trans_len
    ).to(device)
    return model


def main():
    """推理主函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    # 加载配置文件
    config = load_config("config.yaml")
    print("Config: ", config)

    # 初始化分词器
    tokenizer = GTokenizer(config['token_type'])

    # 数据准备
    val_loader, test_loader, max_speech_len, max_trans_len = prepare_data(config, tokenizer)

    # 加载模型
    model = build_model(config, tokenizer, max_speech_len, max_trans_len, device)
    model, _, _ = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        embedding_load=True,
        encoder_load=True,
        decoder_load=True
    )
    model.eval()
    print("Model loaded successfully.")

    # 验证集推理
    print("Running validation inference...")
    levenshtein_distance, _, wer, cer = validate_step(
        model, val_loader, tokenizer, device, mode='full'
    )
    print(f"Validation Results:\nLevenshtein Distance: {levenshtein_distance}\nWER: {wer}\nCER: {cer}")

    # 测试集推理
    print("Running test inference...")
    predictions = test_step(model, test_loader, tokenizer, device)

    # 保存预测结果到CSV
    csv_file_path = "submission.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Labels"])
        for idx, item in enumerate(predictions):
            writer.writerow([idx, item])
    print(f"Test results saved to {csv_file_path}")


if __name__ == "__main__":
    checkpoint_path = "./no-specaug-fbank-global_mvn_fbank_Transformer_ENC-2-4_DEC-2-4_256_1024_AdamW_CosineAnnealing_token_char/checkpoints/checkpoint-best-loss-modelfull.pth"
    main()
