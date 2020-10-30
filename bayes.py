import csv
import json

def classify_item(bayes_model, item, missing_value_indicators):
    # for each class, find likelihood item is in that class
    class_likelihoods = []
    for class_label in bayes_model:
        product = class_label['likelihood']
        for attribute in class_label['attributes']:
            title = attribute['title']
            value = item[title]
            if value in missing_value_indicators:
                continue
            # ignore this attribute if value not found in training
            if value not in attribute['likelihoods']:
                continue
            prob = attribute['likelihoods'][value]
            product *= prob
        class_likelihoods.append({
            'class': class_label['title'],
            'likelihood': product,
        })

    highest_likelihood = 0
    # select class with highest score
    for class_label in class_likelihoods:
        if class_label['likelihood'] > highest_likelihood:
            highest_likelihood = class_label['likelihood']
            selected_class = class_label['class']

    return selected_class

def build_bayesian_model(rows, label_attr, columns, ignored_columns, missing_value_indicators):
    model = []
    attributes = [c for c in columns if c not in ignored_columns and c != label_attr]
    # build the list of all values for all attributes
    attribute_values = {}
    for attribute in attributes:
        values = []
        for row in rows:
            value = row[attribute]
            if value not in values and value not in missing_value_indicators:
                values.append(value)
        attribute_values[attribute] = values

    # find all labels
    labels = list(set([item[label_attr] for item in rows]))
    for label in labels:
        items_with_label = [r for r in rows if r[label_attr] == label]
        label = {
            'title': label,
            'likelihood': len(items_with_label) / len(rows),
            'attributes': [],
        }

        # find likelihoods for each attribute
        for attribute in attributes:
            value_counts = {}

            # start with 1: Laplacian correction
            for value in attribute_values[attribute]:
                value_counts[value] = 1

            # count number of times each value occurs
            for row in items_with_label:
                value = row[attribute]
                if value in missing_value_indicators:
                    continue
                value_counts[value] += 1

            items_counted = sum(value_counts.values())

            # calculate 
            value_likelihoods = {}
            for value in value_counts:
                value_likelihoods[value] = value_counts[value] / (items_counted)

            # chart for value counts
            value_list = [{'title': key, 'count': value_counts[key]} for key in value_counts]
            value_list.sort(key=lambda item: -1 * item['count'])
            value_max = max([value['count'] for value in value_list])
            value_chart = []
            for value in value_list:
                num_bars = 1 + round(21 * (value['count'] - 1) / value_max)
                bar_text = f"{'|' * num_bars}  {value['title']} ({value['count']})"
                value_chart.append(bar_text)

            label['attributes'].append({
                'title': attribute,
                'likelihoods': value_likelihoods,
                'value_counts': value_counts,
                'items_counted': items_counted,
                'value_chart': value_chart,
            })
        model.append(label)

    return model

if __name__ == '__main__':

    # get training data
    rows = []
    with open('data/Genes_relation.data', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)

    print(f"Rows in training data: {len(rows)}")

    # build model
    columns = rows[0].keys()
    model = build_bayesian_model(
        rows,
        'Localization',
        columns,
        ['GeneID', 'Function', 'Chromosome'],
        '?'
    )

    print("Model built")

    # save model for inspection
    with open('model.json', 'w') as file:
        file.write(json.dumps(model, indent=4))
        print(f"Model saved to: model.json")

    # get test data
    tests = []
    with open('data/Genes_relation.test', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tests.append(row)

    print(f"Test cases found: {len(tests)}")

    # apply model to test items
    predictions = []
    for test in tests:
        test['prediction'] = classify_item(model, test, ['?'])
        predictions.append(test)

    # save predictions
    with open('predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in predictions:
            writer.writerow([row['GeneID'], row['prediction']])

    # get keys
    keys = {}
    with open('data/keys.txt', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keys[row['GeneID']] = row['Localization']

    # set up stats and count actual occurrences
    stats = {
        'total_tested': len(predictions),
    }
    label_stats = {}
    for key in keys:
        label = keys[key]
        if label not in label_stats:
            label_stats[label] = {
                'actual': 0,
                'predicted': 0,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive': 0,
                'false_negative': 0,
            }

    # check accuracy
    num_total = len(predictions)
    num_correct = 0
    for prediction in predictions:
        correct_label = keys[prediction['GeneID']]
        predicted_label = prediction['prediction']
        label_stats[predicted_label]['predicted'] += 1
        label_stats[correct_label]['actual'] += 1
        if predicted_label == correct_label:
            num_correct += 1
            label_stats[correct_label]['true_positive'] += 1
            for other_label in label_stats:
                if other_label != correct_label:
                    label_stats[other_label]['true_negative'] += 1
        else:
            label_stats[predicted_label]['false_positive'] += 1
            label_stats[correct_label]['false_negative'] += 1
            for other_label in label_stats:
                if other_label != correct_label and other_label != predicted_label:
                    label_stats[other_label]['true_negative'] += 1

    # detailed per-class stats
    for label in label_stats:
        s = label_stats[label]
        s['sum'] = s['false_positive'] + s['false_negative'] + s['true_positive'] + s['true_negative']
        s['accuracy'] = (s['true_positive'] + s['true_negative']) / num_total
        s['sensitivity'] = s['true_positive'] / s['actual']
        s['specificity'] = s['true_negative'] / (num_total - s['actual'])
        try:
            s['precision'] = s['true_positive'] / (s['true_positive'] + s['false_positive'])
        except:
            s['precision'] = None

        try:
            s['recall'] = s['true_positive'] / (s['true_positive'] + s['false_negative'])
        except:
            s['recall'] = None

        try:
            s['f_measure'] = 2 * s['precision'] * s['recall'] / (s['precision'] + s['recall'])
        except:
            s['f_measure'] = None

    # save stats
    stats['labels'] = label_stats
    with open('stats.json', 'w') as file:
        file.write(json.dumps(stats, indent=4))
        print(f"Stats saved to: stats.json")

    print(f"Total tested: {num_total}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {num_correct/num_total}")
