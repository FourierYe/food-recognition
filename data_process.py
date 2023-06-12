train_full = []

with open('./data/ISIA_Food500/metadata_ISIAFood_500/val_finetune.txt', encoding='utf-8') as file:
    for line in file:
        train_full.append(line)

with open('./data/ISIA_Food500/metadata_ISIAFood_500/train_finetune.txt', encoding='utf-8') as file:
    for line in file:
        train_full.append(line)

with open('./data/ISIA_Food500/metadata_ISIAFood_500/test_public.txt', encoding='utf-8') as file:
    for line in file:
        train_full.append(line)

with open('./data/ISIA_Food500/metadata_ISIAFood_500/train_full2.txt', 'w+', encoding='utf-8') as file:
    for data in train_full:
        file.write(data)
