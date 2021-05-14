import sys
import re
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

def main(filename):
	predictions = []
	labels = []
	regex = r'([a-z_A-Z]+)\s\/\s([a-z_A-Z]+)'
	with open(filename, 'r') as f:
		next(f)
		for line in f:
			matches = re.findall(regex, line)
			if matches:
				predictions.append(matches[0][0])
				labels.append(matches[0][1])
	precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
	accuracy = accuracy_score(labels, predictions)
	balanced_accuracy = balanced_accuracy_score(labels, predictions)
	result = {'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy, 'precision': precision, 'recall': recall, 'f1': fscore}
	return result

if __name__ == '__main__':
	print(main(sys.argv[1]))