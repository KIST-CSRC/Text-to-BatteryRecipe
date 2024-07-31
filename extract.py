from models.predict import predict
import os
import pandas as pd

model = 'batterybert'
type = 'cell_assembly'
save_dir = '/home/daeun/miniconda3/daeun_code/T2BR/'

model_files = {'bert': 'bert-base-uncased',
               'scibert': 'allenai/scibert_scivocab_uncased',
               'matbert': 'T2BR/bert_model/matbert-base-uncased',
               'batterybert': 'batterydata/batterybert-uncased'
               }
text_files = {'cathode_synthesis': 'T2BR/data/LFP_synthesis_data',
              'cell_assembly': 'T2BR/data/Battery_cell_manufacturing_data'
              }
best_model_files = {
    'cathode_synthesis': 'T2BR/models/best_model/cathode_synthesis/best.pt',
    'cell_assembly': 'T2BR/models/best_model/cell_assembly/best.pt'}


def collect_text_files(forder_path):
    texts = []
    nums = []
    for file in os.listdir(forder_path):
        nums.append(file)
        file_path = os.path.join(forder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
                texts.append(text)
    return texts[:100], nums[:100]


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

texts, nums = collect_text_files(text_files[type])

annotations = predict(
    texts=texts,
    is_file=False,
    model_file=model_files[model],
    state_path=best_model_files[type],
    predict_path=save_dir+"ner_prediction.txt",
    return_full_dict=False,
    scheme="IOBES",
    batch_size=256,
    device="gpu:0",
    seed=4,
    criteria='loss',
    criteria2='None',
    patience=10,
    mode='min'
)

# Output results by paragraph
with open(save_dir+'ner_prediction.txt', 'w') as f:
    for i in range(len(texts)):
        f.write(nums[i]+'\n\n')
        f.write(texts[i]+'\n\n')
        f.write(str(annotations[i])+'\n\n')

# Organize by title / text / label for each paragraph
nums_df = pd.DataFrame(nums, columns=['title'])
texts_df = pd.DataFrame(texts, columns=['text'])
annotations_df = pd.DataFrame(annotations, columns=['label'])
for i in range(len(annotations)):
    annotations_df = pd.concat([annotations_df, pd.DataFrame(
        {'label': [annotations[i]]})], ignore_index=True)
annotations_df = annotations_df.dropna()
annotations_df = annotations_df.reset_index(drop=True)

para_df = pd.concat([nums_df, texts_df, annotations_df], axis=1)
para_df.to_excel(save_dir+'ner_prediction.xlsx', index=False)
