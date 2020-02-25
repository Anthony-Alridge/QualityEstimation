preds = []

with open('predictions_1.txt', 'r') as f:
    for line in f:
        if line != '':
            preds.append(str(float(line)))

with open('predictions.txt', 'w') as f:
    f.write('\n'.join(preds))
