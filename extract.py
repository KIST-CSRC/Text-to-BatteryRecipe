import os
import pandas as pd
import argparse
from models.predict import predict

model_files = {
    'bert': 'bert-base-uncased',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'matbert': './matbert-base-uncased',
    'batterybert': 'batterydata/batterybert-uncased'
}

text_files = {
    'cathode_synthesis': 'data/LFP_synthesis_data',
    'cell_assembly': 'data/Battery_cell_manufacturing_data'
}

best_model_files = {
    'cathode_synthesis': 'models/best_model/cathode_synthesis/best.pt',
    'cell_assembly': 'models/best_model/cell_assembly/best.pt'
}

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

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    texts, nums = collect_text_files(text_files[args.type])

    annotations = predict(
        texts=texts,
        is_file=False,
        model_file=model_files[args.model],
        state_path=best_model_files[args.type],
        predict_path=os.path.join(args.save_dir, "ner_prediction.txt"),
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
    with open(os.path.join(args.save_dir, 'ner_prediction.txt'), 'w') as f:
        for i in range(len(texts)):
            f.write(nums[i]+'\n\n')
            f.write(texts[i]+'\n\n')
            f.write(str(annotations[i])+'\n\n')

    
    nums_df = pd.DataFrame(nums, columns=['title'])
    texts_df = pd.DataFrame(texts, columns=['text'])
    annotations_df = pd.DataFrame(annotations, columns=['label'])
    
    para_df = pd.concat([nums_df, texts_df, annotations_df], axis=1)
    para_df.to_excel(os.path.join(args.save_dir, 'ner_prediction.xlsx'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, choices=model_files.keys(), help="Choose the model to use.")
    parser.add_argument('--type', type=str, required=True, choices=text_files.keys(), help="Choose the type of text data.")
    parser.add_argument('--save_dir', type=str, default='./result', help="Directory to save the results.")

    args = parser.parse_args()
    main(args)