train_full = []

with open('val_finetune.txt', encoding='utf-8') as file:
    for line in file:
        train_full.append(line)

with open('train_finetune.txt', encoding='utf-8') as file:
    for line in file:
        train_full.append(line)

with open('train_full.txt', 'w+', encoding='utf-8') as file:
    for data in train_full:
        file.write(data)
