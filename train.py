import os
import argparse
import numpy as np
from seqeval.metrics import classification_report
import json
import torch 
from utils.data import NERData
from models.bert_model import BERTNER
from models.model_trainer import NERTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', type=str, default='cpu', help='computation device for model (e.g. cpu, gpu:0, gpu:1)')
    parser.add_argument('-sd', '--seeds', type=str, default='8', help='comma-separated seeds for data shuffling and model initialization')
    parser.add_argument('-ts', '--tag_schemes', type=str, default='iobes', help='comma-separated tagging schemes to be considered')
    parser.add_argument('-st', '--splits', type=str, default='80', help='comma-separated training splits to be considered, in percent')
    parser.add_argument('-ds', '--datasets', type=str, default='cell_assembly', help='comma-separated datasets to be considered')
    parser.add_argument('-ml', '--models', type=str, default='matbert', help='comma-separated models to be considered')
    parser.add_argument('-sl', '--sentence_level', action='store_true', help='switch for sentence-level learning instead of paragraph-level')
    parser.add_argument('-bs', '--batch_size', type=int, default=5, help='number of samples in each batch')
    parser.add_argument('-on', '--optimizer_name', type=str, default='rangerlars', help='name of optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.001, help='weight decay for optimizer')
    parser.add_argument('-ne', '--n_epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('-eu', '--embedding_unfreeze', type=int, default=1, help='epoch (index) at which bert embeddings are unfrozen')
    parser.add_argument('-tu', '--transformer_unfreeze', type=str, default='0,12', help='comma-separated number of transformers to unfreeze at each epoch')
    parser.add_argument('-el', '--embedding_learning_rate', type=float, default=1e-4, help='embedding learning rate')
    parser.add_argument('-tl', '--transformer_learning_rate', type=float, default=2e-3, help='transformer learning rate')
    parser.add_argument('-cl', '--classifier_learning_rate', type=float, default=1e-2, help='pooler/classifier learning rate')
    parser.add_argument('-sf', '--scheduling_function', type=str, default='exponential', help='function for learning rate scheduler')
    parser.add_argument('-km', '--keep_model', action='store_true', help='switch for saving the best model parameters to disk')
    parser.add_argument('-pt', '--patience_pt', type=int, default=3, help='patience for early stopping')
    parser.add_argument('-cr', '--criteria_cr1', type=str, default='loss', help='primary criteria for early stopping')
    parser.add_argument('-cr2', '--criteria_cr2', type=str, default='None', help='secondary criteria for early stopping')
    parser.add_argument('-md', '--mode_md', type=str, default='min', help='mode for early stopping (max/min)')
    
    return parser.parse_args()


def setup_environment(args):
    gpu = 'gpu' in args.device
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
    device = 'cuda' if gpu else 'cpu'
    return device, gpu


def get_ner_data(model, scheme, dataset, split, sentence_level, seed):
    
    data_files = {
        'cell_assembly': 'data/cell_assembly.json',
        'cathode_synthesis': 'data/cathode_synthesis.json',
    }
    if dataset not in data_files:
        data_files[dataset.split('/')[-1].split('.')[-2]] = dataset
        dataset = dataset.split('/')[-1].split('.')[-2]

    ner_data = NERData(model, scheme=scheme)
    split_dict = {'train': split/100} if split == 100 else {'test': 0.1, 'valid': 0.00125 * split, 'train': 0.01 * split}
    ner_data.preprocess(data_files[dataset], split_dict, is_file=True, sentence_level=sentence_level, shuffle=True, seed=seed)
    ner_data.create_dataloaders(batch_size=args.batch_size, shuffle=True, seed=seed)
    if split == 100:
        ner_data.dataloaders['valid'] = None
        ner_data.dataloaders['test'] = None
    return ner_data


def create_model_trainer(args, ner_data, seed):


    model_files = {
        'bert': 'bert-base-uncased',
        'scibert': 'allenai/scibert_scivocab_uncased',
        'matbert': './matbert-base-uncased',
        'batterybert': 'batterydata/batterybert-uncased',
    }

    params = (args.datasets, args.patience_pt, args.batch_size, args.n_epoch, args.criteria_cr1, args.criteria_cr2, args.mode_md, args.tag_schemes.lower(), args.models,
              args.optimizer_name, 'sentence' if args.sentence_level else 'paragraph', args.embedding_unfreeze, args.transformer_unfreeze.replace(',', ''), 
              args.embedding_learning_rate, args.transformer_learning_rate, args.classifier_learning_rate, args.weight_decay, args.scheduling_function, seed, args.splits)

    alias = '{}_pa{}_ba{}_ep{}_cr_{}_{}_{}_{}_crf_{}_{}_{}_{}_{}_{:.0e}_{:.0e}_{:.0e}_{:.0e}_{}_{}_{}'.format(*params)
    save_dir = f'./{alias}/'

    print(f'Calculating results for {alias}')

    device = setup_environment(args)[0]
    bert_ner_trainer = NERTrainer(BERTNER(model_file=model_files[args.models], classes=ner_data.classes, scheme=args.tag_schemes.lower(),
                                   seed=seed), device, criteria=args.criteria_cr1, criteria2=args.criteria_cr2, patience=args.patience_pt, mode=args.mode_md)
    return bert_ner_trainer, save_dir, alias


def train_and_evaluate(args, seed, scheme, split, dataset, model, ner_data, bert_ner_trainer, save_dir, alias):


    if os.path.exists(save_dir + 'history.json'):
        print(f'Already trained {alias}')
        with open(save_dir + 'history.json', 'r') as f:
            history = json.load(f)
        if split == 100:
            for i in range(len(history['training'].keys())):
                metrics = {key: np.mean([batch['micro avg']['f1-score'] for batch in history[key][f'epoch_{i}']]) for key in ['training']}
                print(f'{i:<10d}{metrics["training"]:<10.4f}')
        else:
            for i in range(len(history['training'].keys())):
                metrics = {key: np.mean([batch['micro avg']['f1-score'] for batch in history[key][f'epoch_{i}']]) for key in ['training', 'validation']}
                print(f'{i:<10d}{metrics["training"]:<10.4f}{metrics["validation"]:<10.4f}')
    else:
        try:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            bert_ner_trainer.init_optimizer(optimizer_name=args.optimizer_name, elr=args.embedding_learning_rate, tlr=args.transformer_learning_rate, clr=args.classifier_learning_rate, weight_decay=args.weight_decay)
            bert_ner_trainer.train(n_epoch=args.n_epoch, train_iter=ner_data.dataloaders['train'], valid_iter=ner_data.dataloaders['valid'],
                                   embedding_unfreeze=args.embedding_unfreeze, encoder_schedule=[int(num) for num in args.transformer_unfreeze.split(',')],
                                   scheduling_function=args.scheduling_function, save_dir=save_dir, use_cache=False)
            bert_ner_trainer.save_history(history_path=save_dir + 'history.json')
            if ner_data.dataloaders['test'] is not None:
                metrics, test_results = bert_ner_trainer.test(ner_data.dataloaders['test'], test_path=save_dir + 'test.json', state_path=save_dir + 'best.pt')
                annotations = bert_ner_trainer.predict(ner_data.dataloaders['test'], original_data=ner_data.data['test'], predict_path=save_dir + 'predict.json', state_path=save_dir + 'best.pt', return_full_dict=True)
                print(classification_report(test_results['labels'], test_results['predictions'], mode='strict', scheme=bert_ner_trainer.metric_scheme, zero_division=True))
                with open(save_dir + 'predictions.txt', 'w') as f:
                    for entry in annotations:
                        f.write(160 * '=' + '\n')
                        for sentence in entry['tokens']:
                            f.write(160 * '-' + '\n')
                            for word in sentence:
                                f.write(f'{word["text"]:<40}{word["annotation"]:<40}\n')
                            f.write(160 * '-' + '\n')
                        f.write(160 * '-' + '\n')
                        for entity_type in entry['entities'].keys():
                            f.write(f'{entity_type:<20},{", ".join(entry["entities"][entity_type])}\n')
                        f.write(160 * '-' + '\n')
                        f.write(160 * '=' + '\n')
        except Exception as e:
            print('Error encountered:', str(e))




if __name__ == '__main__':
    args = parse_args()
    setup_environment(args)
    
    for seed in [int(seed) for seed in args.seeds.split(',')]:
        for scheme in [str(tag_scheme).upper() for tag_scheme in args.tag_schemes.split(',')]:
            for split in [int(split) for split in args.splits.split(',')]:
                for dataset in [str(dataset) for dataset in args.datasets.split(',')]:
                    ner_data = get_ner_data(args.models, scheme, dataset, split, args.sentence_level, seed)
                    bert_ner_trainer, save_dir, alias = create_model_trainer(args, ner_data, seed)
                    train_and_evaluate(args, seed, scheme, split, dataset, args.models, ner_data, bert_ner_trainer, save_dir, alias)
                    del ner_data
                    del bert_ner_trainer
                    torch.cuda.empty_cache()