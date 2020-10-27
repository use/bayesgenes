import csv
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

def classify_item(bayes_model, item):
    # for each class, find likelihood item is in that class
    class_likelihoods = []
    for class_label in bayes_model:
        product = class_label['likelihood']
        for attribute in class_label['attributes']:
            title = attribute['title']
            value = item[title]
            prob = attribute['likelihoods'][value]
            product *= prob
        likelihoods.append(
            {
                'class': class_label['title'],
                'likelihood': product,
            }
        )
    selected_class = null
    for class_label in class_likelihoods:
        if selected_class is null or class_label['likelihood'] > selected_class['likelihood']:
            selected_class = class_label

    return selected_class

def build_bayesian_model(rows, label_attr, columns, ignored_columns, ignored_values):
    model = []
    attributes = [c for c in columns if c not in ignored_columns and c != label_attr]
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
            # count number of items with this class label for each value on this attribute
            for row in items_with_label:
                value = row[attribute]
                if value in ignored_values:
                    continue
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1
            value_likelihoods = {}
            for value in value_counts:
                value_likelihoods[value] = value_counts[value] / len(items_with_label)

            label['attributes'].append({
                'title': attribute,
                'likelihoods': value_likelihoods,
            })
        model.append(label)

    return model

if __name__ == '__main__':
    rows = []
    with open('data/Genes_relation.data', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    print(len(rows))
    print(rows[0])
    columns = rows[0].keys()
    model = build_bayesian_model(rows, 'Localization', columns, ['GeneID', 'Function'], '?')
    pprint.pprint(model)
