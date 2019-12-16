i = 0
f1 = open('train_data', 'a+', encoding='utf-8')
f2 = open('test_data', 'a+', encoding='utf-8')

with open('all_mark.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if i < 60000:
            f2.write(line)
        else:
            f1.write(line)
        i += 1
