# HW#2 Machine Learning 10-601, Meiirbek Islamov
# Decision Tree Learner

# import the necessary libraries
import sys
import numpy as np

args = sys.argv
assert(len(args) == 7)
train_input = args[1] # Path to the training data .tsv file
test_input = args[2] # Path to the test input .tsv file
max_depth = int(args[3]) # Maximum depth to which the tree should be built
train_out = args[4] # Path of output .label file to which the predictions on the training data should be written
test_out = args[5] # Path of output .labels file to which the predictions on the test data should be written
metrics_out = args[6] # Path of the output .txt file to which metrics such as train and test error should be written

# Functions
def read_data(input):
    with open(input, 'r') as f_in:
        next(f_in) # Skip the first row (header)
        lines = f_in.readlines()
    data = np.array([list(l.split()) for l in lines])
    return data

# Entropy
def entropy(attribute):
    values, counts = np.unique(attribute, return_counts = True)
    entropy_list = []
    for i in range(len(values)):
        entropy_list.append(-counts[i]/len(attribute) * np.log2(counts[i]/len(attribute)))
    entropy = np.sum(np.array(entropy_list))
    return entropy

# Information info_gain
def information_gain(feature, label):
    values, counts = np.unique(feature, return_counts=True)
    H_Y = entropy(label)
    index_yes, index_no = [], []
    for i, item in enumerate(feature):
        if item == values[0]:
            index_yes.append(i)
        elif item == values[1]:
            index_no.append(i)

    label_no = np.delete(label, index_yes, axis=0)
    label_yes = np.delete(label, index_no, axis=0)

    label_index = []
    label_index.append(label_yes)
    label_index.append(label_no)

    entropy_feature_list = []
    for i in range(len(values)):
        entropy_feature_list.append(counts[i]/len(feature) * entropy(label_index[i]))
    entropy_feature_sum = np.sum(entropy_feature_list)
    info_gain = H_Y - entropy_feature_sum

    return info_gain

# Decision tree
def decision_tree(data, max_depth):
    num_rows, num_cols = data.shape
    num_features = num_cols - 1

    # Base case

    label = data[:, -1]
    result = np.all(label == label[0])

    if result:
        return label[0]

    elif num_features == 0:
        values,counts = np.unique(label, return_counts=True)
        if len(counts) == 2:
            if counts[0] != counts[1]:
                index_max = np.argmax(counts)
                return values[index_max]
            elif counts[0] == counts[1]:
                sorted_list = sorted(values, reverse=True)
                return sorted_list[0]

        elif len(counts) == 1:
            return values[0]

    elif max_depth == 0:
        values,counts = np.unique(label, return_counts=True)
        if len(counts) == 2:
            if counts[0] != counts[1]:
                index_max = np.argmax(counts)
                return values[index_max]
            elif counts[0] == counts[1]:
                sorted_list = sorted(values, reverse=True)
                return sorted_list[0]

        elif len(counts) == 1:
            return values[0]

    else:

        info_gain = []
        for i in range(num_features):
            info_gain.append(information_gain(data[:, i], label))

        index_highest_feature = np.argmax(info_gain)

        if info_gain[index_highest_feature] > 0:

            highest_feature = data[:, index_highest_feature]

            values,counts = np.unique(highest_feature, return_counts=True)

            index_yes, index_no = [], []
            for i, item in enumerate(highest_feature):
                if item == values[0]:
                    index_yes.append(i)
                elif item == values[1]:
                    index_no.append(i)

            learned_tree = {index_highest_feature:{}}

            max_depth -= 1

            # Recursive case

            subdata_no = np.delete(data, index_yes, axis=0)
            subtree_no = decision_tree(subdata_no, max_depth)

            subdata_yes = np.delete(data, index_no, axis=0)
            subtree_yes = decision_tree(subdata_yes, max_depth)

            learned_tree[index_highest_feature][values[0]] = subtree_yes
            learned_tree[index_highest_feature][values[1]] = subtree_no

            return learned_tree

        else:
            values,counts = np.unique(label, return_counts=True)
            if len(counts) == 2:
                if counts[0] != counts[1]:
                    index_max = np.argmax(counts)
                    return values[index_max]
                elif counts[0] == counts[1]:
                    sorted_list = sorted(values, reverse=True)
                    return sorted_list[0]

            elif len(counts) == 1:
                return values[0]

# Prediction
def predict_decision_tree(learned_tree, data):
    if isinstance(learned_tree, dict):
        node = list(learned_tree.keys())[0]
        child_trees = learned_tree[node]
        feature_value = data[node]
        output_value = child_trees[feature_value]

        if isinstance(output_value,dict):
            return predict_decision_tree(output_value, data)
        else:
            return output_value

    else:
            return learned_tree

def convert_data_to_dict(data):
    data_dict = []
    for i in data[:, :-1]:
        data_dict.append(dict(enumerate(i)))
    return data_dict

def predict_labels(data, learned_tree):
    predicted_labels = []
    for i in data:
        predicted_labels.append(predict_decision_tree(learned_tree, i))
    return predicted_labels

# Error
def calculate_error(label_true, label_predicted):
    n = 0
    for i, item in enumerate(label_true):
        if item != label_predicted[i]:
            n += 1
    error = n/len(label_true)
    return error

# Write labels
def write_labels(predicted_label, filename):
    with open(filename, 'w') as f_out:
        for label in predicted_label:
            f_out.write(str(label) + '\n')

# Write error
def write_error(train_error, test_error, filename):
    with open(filename, 'w') as f_out:
        f_out.write("error(train): " + str(train_error) + "\n")
        f_out.write("error(test): " + str(test_error) + "\n")

# Main body
# Training
train_data = read_data(train_input)
train_data_dict = convert_data_to_dict(train_data)
learned_tree = decision_tree(train_data, max_depth)
predicted_labels_train = predict_labels(train_data_dict, learned_tree)
train_error = calculate_error(train_data[:, -1], predicted_labels_train)
write_labels(predicted_labels_train, train_out)

# Testing
test_data = read_data(test_input)
test_data_dict = convert_data_to_dict(test_data)
predicted_labels_test = predict_labels(test_data_dict, learned_tree)
test_error = calculate_error(test_data[:, -1], predicted_labels_test)
write_labels(predicted_labels_test, test_out)
write_error(train_error, test_error, metrics_out)
