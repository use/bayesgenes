import csv
import json
import pprint

# def processnode(node, attributes_remaining, label_attr):
#     splitting_attr = null
#     highest_gain = 0
#     for attr in attributes_remaining:
#         gain = calc_gain(node.data, attr, label_attr)
#         if gain > highest_gain:
#             splitting_attr = attr

#     partitions = partition_data(node.data, attr)
#     attributes_remaining.remove(attr)
#     for partition in partitions:
#         child = node.addChild(partition.data, partition.attr_value)
#         processnode(child, attributes_remaining, label_attr)
#     return node


item = {
    'color': 'blue',
    'size': 'medium',
}

training_data = {
    'class_column': ['tier'],
    'columns': ['color', 'size', 'weight', 'tier'],
    'data': [
        ['blue', 'large', 'heavy', 'Platinum'],
        ['red', 'small', 'light', 'Bronze'],
    ]
}

bayes_model = [
    {
        'title': 'Platinum',
        'likelihood': .02,
        'attributes': [
            {
                'title': 'color',
                'likelihoods': {
                    'blue': .01,
                    'red': .05,
                },
            },
            {
                'title': 'size',
                'likelihoods': {
                    'large': .5,
                    'medium': .25,
                    'small': .02,
                },
            },
        ],
    },
    {
        'title': 'Bronze',
    },
]

def classify_item(bayes_model, item, ignored_values):
    # for each class, find likelihood item is in that class
    class_likelihoods = []
    for class_label in bayes_model:
        product = class_label['likelihood']
        for attribute in class_label['attributes']:
            title = attribute['title']
            value = item[title]
            if value in ignored_values:
                continue
            # ignore this attribute if value not found in training
            if value not in attribute['likelihoods']:
                continue
            prob = attribute['likelihoods'][value]
            product *= prob
        class_likelihoods.append(
            {
                'class': class_label['title'],
                'likelihood': product,
            }
        )

    selected_class = None
    for class_label in class_likelihoods:
        if selected_class is None or class_label['likelihood'] > selected_class['likelihood']:
            selected_class = class_label

    return selected_class

def build_bayesian_model(rows, label_attr, columns, ignored_columns, ignored_values):
    model = []
    attributes = [c for c in columns if c not in ignored_columns and c != label_attr]
    # build the list of all values for all attributes
    attribute_values = {}
    for attribute in attributes:
        values = []
        for row in rows:
            value = row[attribute]
            if value not in values and value not in ignored_values:
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
        # go through attributes
        for attribute in attributes:
            value_counts = {}

            # start with 1: Laplacian correction
            for value in attribute_values[attribute]:
                value_counts[value] = 1

            # count number of items with this class label for each value on this attribute
            for row in items_with_label:
                value = row[attribute]
                if value in ignored_values:
                    continue
                value_counts[value] += 1

            items_counted = sum(value_counts.values())


            value_likelihoods = {}
            for value in value_counts:
                value_likelihoods[value] = value_counts[value] / (items_counted)

            label['attributes'].append({
                'title': attribute,
                'likelihoods': value_likelihoods,
                'value_counts': value_counts,
                'items_counted': items_counted,
            })
        model.append(label)

    return model

if __name__ == '__main__':
    rows = []
    with open('data/Genes_relation.data', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    print(f"Rows in training data: {len(rows)}")
    print(f"Sample row: {rows[0]}")

    columns = rows[0].keys()
    model = build_bayesian_model(rows, 'Localization', columns, ['GeneID', 'Function'], '?')

    with open('data/model.json', 'w') as file:
        file.write(json.dumps(model, indent=4))

    print("Model built")

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

    # get keys
    keys = {}
    with open('data/keys.txt', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keys[row['GeneID']] = row['Localization']

    num_total = len(predictions)
    num_correct = 0
    label_accuracies = {}
    for prediction in predictions:
        correct_label = keys[prediction['GeneID']]
        if correct_label not in label_accuracies:
            label_accuracies[correct_label] = {
                'actual': 0,
                'predicted': 0,
            }
        label_accuracies[correct_label]['predicted'] += 1
        if prediction['prediction']['class'] == correct_label:
            num_correct += 1
            

    print(f"Total tested: {num_total}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {num_correct/num_total}")
