import os
import argparse
import numpy as np
from seqeval.metrics import classification_report
import json


def parse_args():
    '''
    Parse command line arguments
        -h for help
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device',
                        help='computation device for model (e.g. cpu, gpu:0, gpu:1)',
                        type=str, default='cpu')
    parser.add_argument('-sd', '--seeds',
                        help='comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)',
                        type=str, default='8')
    parser.add_argument('-ts', '--tag_schemes',
                        help='comma-separated tagging schemes to be considered (e.g. iob1,iob2,iobes)',
                        type=str, default='iobes')
    parser.add_argument('-st', '--splits',
                        help='comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10%% and the validation split will be 1/8 of the training split unless the training split is 100%%',
                        type=str, default='80')
    parser.add_argument('-ds', '--datasets',
                        help='comma-separated datasets to be considered (e.g. cathode_synthesis, cell_assembly)',
                        type=str, default='cell_assembly')
    parser.add_argument('-ml', '--models',
                        help='comma-separated models to be considered (e.g. bert,scibert,matbert,batterybert)',
                        type=str, default='matbert')
    parser.add_argument('-sl', '--sentence_level',
                        help='switch for sentence-level learning instead of paragraph-level',
                        action='store_true')
    parser.add_argument('-bs', '--batch_size',
                        help='number of samples in each batch',
                        type=int, default=5)
    parser.add_argument('-on', '--optimizer_name',
                        help='name of optimizer, add "_lookahead" to implement lookahead on top of optimizer (not recommended for ranger or rangerlars)' "'adamw': AdamW, 'rangerlars': RangerLars,'ralamb': Ralamb, 'ranger': Ranger,'novograd': Novograd, 'radam': RAdam, 'lamb': Lamb",
                        type=str, default='rangerlars')
    parser.add_argument('-wd', '--weight_decay',
                        help='weight decay for optimizer (excluding bias, gamma, and beta)',
                        type=float, default=0.001)
    parser.add_argument('-ne', '--n_epoch',
                        help='number of training epochs',
                        type=int, default=50)
    parser.add_argument('-eu', '--embedding_unfreeze',
                        help='epoch (index) at which bert embeddings are unfrozen',
                        type=int, default=1)
    parser.add_argument('-tu', '--transformer_unfreeze',
                        help='comma-separated number of transformers (encoders) to unfreeze at each epoch',
                        type=str, default='0,12')
    parser.add_argument('-el', '--embedding_learning_rate',
                        help='embedding learning rate',
                        type=float, default=1e-4)
    parser.add_argument('-tl', '--transformer_learning_rate',
                        help='transformer learning rate',
                        type=float, default=2e-3)
    parser.add_argument('-cl', '--classifier_learning_rate',
                        help='pooler/classifier learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-sf', '--scheduling_function',
                        help='function for learning rate scheduler (linear, exponential, or cosine)',
                        type=str, default='exponential')
    parser.add_argument('-km', '--keep_model',
                        help='switch for saving the best model parameters to disk',
                        action='store_true')
    parser.add_argument('-pt', '--patience_pt',
                        help='switch for saving the best model parameters to disk',
                        type=int, default=3)
    parser.add_argument('-cr', '--criteria_cr1',
                        help='ANODE / ACTIVE_MATERIAL / loss / accuracy ',
                        type=str, default='loss')
    parser.add_argument('-cr2', '--criteria_cr2',
                        help='f1-score / None',
                        type=str, default='None')
    parser.add_argument('-md', '--mode_md',
                        help='max/min',
                        type=str, default='min')

    args = parser.parse_args()
    return (args.device, args.seeds, args.tag_schemes, args.splits, args.datasets,
            args.models, args.sentence_level, args.batch_size, args.optimizer_name, args.weight_decay,
            args.n_epoch, args.embedding_unfreeze, args.transformer_unfreeze,
            args.embedding_learning_rate, args.transformer_learning_rate, args.classifier_learning_rate,
            args.scheduling_function, args.keep_model, args.patience_pt, args.mode_md, args.criteria_cr1, args.criteria_cr2)


if __name__ == '__main__':
    # retrieve command line arguments
    (device, seeds, tag_schemes, splits, datasets,
     models, sentence_level, batch_size, optimizer_name, weight_decay,
     n_epoch, embedding_unfreeze, transformer_unfreeze,
     elr, tlr, clr, scheduling_function, keep_model, patience_pt, mode_md, criteria_cr1, criteria_cr2) = parse_args()
    # if gpu
    if 'gpu' in device:
        # set device as cuda and retreive number
        gpu = True
        try:
            d, n = device.split(':')
        except:
            print('ValueError: Improper device format in command-line argument')
        device = 'cuda'
    else:
        gpu = False
    # set environment variable to make only chosen gpu visible
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(n)
    # import torch dependent packages after setting gpu
    import torch
    from utils.data import NERData
    from models.bert_model import BERTNER
    from models.model_trainer import NERTrainer

    # set device and establish deterministic behavior
    torch.device('cuda' if gpu else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # use disk instead of cache for saving model/optimizer state
    use_cache = False
    # convert command line arguments to lists
    seeds = [int(seed) for seed in seeds.split(',')]
    schemes = [str(tag_scheme).upper()
               for tag_scheme in tag_schemes.split(',')]
    splits = [int(split) for split in splits.split(',')]
    datasets = [str(dataset) for dataset in datasets.split(',')]
    models = [str(model) for model in models.split(',')]
    encoder_schedule = [int(num) for num in transformer_unfreeze.split(',')]
    # validate encoder schedule and expand to number of epochs
    if len(encoder_schedule) > n_epoch:
        encoder_schedule = encoder_schedule[:n_epoch]
        print('Provided with encoder schedule longer than number of epochs, truncating')
    elif len(encoder_schedule) < n_epoch:
        encoder_schedule = encoder_schedule + \
            ((n_epoch-len(encoder_schedule))*[0])
    if np.sum(encoder_schedule) > 12:
        encoder_schedule = embedding_unfreeze*[0]+[12]
        print('Provided invalid encoder schedule (too many layers), all encoders will be unlocked with the BERT embeddings')
    # data file dictionary
    data_files = {'cell_assembly': 'T2BR/data/cell_assembly.json',
                  'cathode_synthesis': 'T2BR/data/cathode_synthesis.json'
                  }
    # model file dictionary
    model_files = {'bert': 'bert-base-uncased',
                   'scibert': 'allenai/scibert_scivocab_uncased',
                   'matbert': 'T2BR/bert_model/matbert-base-uncased',
                   'batterybert': 'batterydata/batterybert-uncased',
                   }
    # loop through command line lists
    for seed in seeds:
        for scheme in schemes:
            for split in splits:
                for dataset in datasets:
                    if dataset not in data_files.keys():
                        data_files[dataset.split(
                            '/')[-1].split('.')[-2]] = dataset
                        dataset = dataset.split('/')[-1].split('.')[-2]
                    for model in models:
                        # parameter tumple
                        params = (dataset, patience_pt, batch_size, n_epoch, criteria_cr1, criteria_cr2, mode_md, scheme.lower(), model, optimizer_name,
                                  'sentence' if sentence_level else 'paragraph', embedding_unfreeze, transformer_unfreeze.replace(',', ''), elr, tlr, clr, weight_decay, scheduling_function, seed, split)
                        # alias for save directory
                        alias = '{}_pa{}_ba{}_ep{}_cr_{}_{}_{}_{}_crf_{}_{}_{}_{}_{}_{:.0e}_{:.0e}_{:.0e}_{:.0e}_{}_{}_{}'.format(
                            *params)
                        save_dir = './{}/'.format(
                            alias)
                        print('Calculating results for {}'.format(alias))
                        # initialize ner data and split dictionary
                        ner_data = NERData(model_files[model], scheme=scheme)
                        if split == 100:
                            split_dict = {'train': split/100}
                        else:
                            split_dict = {
                                'test': 0.1, 'valid': 0.00125*split, 'train': 0.01*split}
                        ner_data.preprocess(
                            data_files[dataset], split_dict, is_file=True, sentence_level=sentence_level, shuffle=True, seed=seed)
                        ner_data.create_dataloaders(
                            batch_size=batch_size, shuffle=True, seed=seed)
                        if split == 100:
                            ner_data.dataloaders['valid'] = None
                            ner_data.dataloaders['test'] = None
                        # construct model trainer
                        bert_ner_trainer = NERTrainer(BERTNER(model_file=model_files[model], classes=ner_data.classes, scheme=scheme,
                                                      seed=seed), device, criteria=criteria_cr1, criteria2=criteria_cr2, patience=patience_pt, mode=mode_md)

                        # print classes
                        print('Classes: {}'.format(' '.join(ner_data.classes)))
                        # if test file already exists, skip, otherwise, train
                        succeeded = True
                        if os.path.exists(save_dir+'history.json'):
                            print('Already trained {}'.format(alias))
                            with open(save_dir+'history.json', 'r') as f:
                                history = json.load(f)
                            if split == 100:
                                print('{:<10}{:<10}'.format(
                                    'epoch', 'training'))
                                for i in range(len(history['training'].keys())):
                                    metrics = {key: np.mean(
                                        [batch['micro avg']['f1-score'] for batch in history[key]['epoch_{}'.format(i)]]) for key in ['training']}
                                    print('{:<10d}{:<10.4f}'.format(
                                        i, metrics['training']))
                            else:
                                print('{:<10}{:<10}{:10}'.format(
                                    'epoch', 'training', 'validation'))
                                for i in range(len(history['training'].keys())):
                                    metrics = {key: np.mean(
                                        [batch['micro avg']['f1-score'] for batch in history[key]['epoch_{}'.format(i)]]) for key in ['training', 'validation']}
                                    print('{:<10d}{:<10.4f}{:<10.4f}'.format(
                                        i, metrics['training'], metrics['validation']))
                        else:
                            try:
                                # create directory if it doesn't exist
                                if not os.path.exists(save_dir):
                                    os.mkdir(save_dir)
                                # initialize optimizer
                                bert_ner_trainer.init_optimizer(
                                    optimizer_name=optimizer_name, elr=elr, tlr=tlr, clr=clr, weight_decay=weight_decay)
                                # train model
                                bert_ner_trainer.train(n_epoch=n_epoch, train_iter=ner_data.dataloaders['train'], valid_iter=ner_data.dataloaders['valid'],
                                                       embedding_unfreeze=embedding_unfreeze, encoder_schedule=encoder_schedule, scheduling_function=scheduling_function,
                                                       save_dir=save_dir, use_cache=use_cache)
                                # save model history
                                bert_ner_trainer.save_history(
                                    history_path=save_dir+'history.json')
                                # if cache was used and the model should be kept, the state must be saved directly after loading best parameters
                                if use_cache:
                                    bert_ner_trainer.load_state_from_cache(
                                        'best')
                                    bert_ner_trainer.save_state(
                                        state_path=save_dir+'best.pt')
                                if split == 100:
                                    bert_ner_trainer.save_state(
                                        state_path=save_dir+'best.pt')
                            except Exception as e:
                                succeeded = False
                                print('Error encountered:', str(e))

                        # if test dataloader provided
                        if ner_data.dataloaders['test'] is not None and succeeded:
                            if os.path.exists(save_dir+'best.pt'):
                                # predict test results
                                metrics, test_results = bert_ner_trainer.test(
                                    ner_data.dataloaders['test'], test_path=save_dir+'test.json', state_path=save_dir+'best.pt')
                                # predict classifications
                                annotations = bert_ner_trainer.predict(
                                    ner_data.dataloaders['test'], original_data=ner_data.data['test'], predict_path=save_dir+'predict.json', state_path=save_dir+'best.pt', return_full_dict=True)
                            elif os.path.exists(save_dir+'test.json'):
                                # retrieve test results
                                with open(save_dir+'test.json', 'r') as f:
                                    test = json.load(f)
                                    metrics, test_results = test['metrics'], test['results']
                                # retrieve classifications
                                with open(save_dir+'predict.json', 'r') as f:
                                    annotations = json.load(f)
                            # print classification report over test results
                            print(classification_report(
                                test_results['labels'], test_results['predictions'], mode='strict', scheme=bert_ner_trainer.metric_scheme, zero_division=True))
                            # save tokens/annotations to text file
                            with open(save_dir+'predictions.txt', 'w') as f:
                                for entry in annotations:
                                    f.write(160*'='+'\n')
                                    for sentence in entry['tokens']:
                                        f.write(160*'-'+'\n')
                                        for word in sentence:
                                            f.write('{:<40}{:<40}\n'.format(
                                                word['text'], word['annotation']))
                                        f.write(160*'-'+'\n')
                                    f.write(160*'-'+'\n')
                                    for entity_type in entry['entities'].keys():
                                        f.write('{:<20}{}\n'.format(
                                            entity_type, ', '.join(entry['entities'][entity_type])))
                                    f.write(160*'-'+'\n')
                                    f.write(160*'='+'\n')
                        del ner_data
                        del bert_ner_trainer
                        torch.cuda.empty_cache()
