import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

def load_data(folder_path, queries=None):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Ensuring to read only text files
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    if queries:
                        for query in queries:
                            if query.lower() in content.lower():
                                data.append(content)
                                break  # Breaks to avoid duplicate entries for a file matching multiple queries
                    else:
                        data.append(content)
            except UnicodeDecodeError as e:
                print(f"Error decoding file {filename}: {str(e)}")
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    return data

def encode_data(data, tokenizer, model):
    encoded_data = []
    for i, text in enumerate(data):
        print(f"Encoding example {i + 1}...")
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"][:, :512]
            outputs = model(input_ids)
            logits = outputs.logits
            encoded_data.append(logits)
        except Exception as e:
            print(f"Error encoding example {i + 1}; {str(e)}")
    return encoded_data

def evaluate_model(encoded_data, ground_truth_labels, model):
    logits = torch.cat(encoded_data, dim=0)
    predictions = logits.argmax(dim=1).tolist()
    accuracy = accuracy_score(ground_truth_labels, predictions)
    report = classification_report(ground_truth_labels, predictions, labels = [0,1], target_names=["negative","positive"])
    return accuracy, report

# Paths to your data folders
train_pos_path = "reviews_dataset/train/pos"
train_neg_path = "reviews_dataset/train/neg"
test_pos_path = "reviews_dataset/test/pos"
test_neg_path = "reviews_dataset/test/neg"

queries_to_search = ["great", "disappointing", "awesome"]
train_positive_data = load_data(train_pos_path, queries_to_search)
train_negative_data = load_data(train_neg_path, queries_to_search)
test_positive_data = load_data(test_pos_path, queries_to_search)
test_negative_data = load_data(test_neg_path, queries_to_search)

# Define the model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Encode the data
# encoded_train_positive = encode_data_batch(train_positive_data[:50], tokenizer, model)
# encoded_train_negative = encode_data_batch(train_negative_data[:50], tokenizer, model)
# encoded_test_positive = encode_data_batch(test_positive_data[:50], tokenizer, model)
# encoded_test_negative = encode_data_batch(test_negative_data[:50], tokenizer, model)
# Encode the data
encoded_train_positive = encode_data(train_positive_data[:50], tokenizer, model)
encoded_train_negative = encode_data(train_negative_data[:50], tokenizer, model)
encoded_test_positive = encode_data(test_positive_data[:50], tokenizer, model)
encoded_test_negative = encode_data(test_negative_data[:50], tokenizer, model)
ground_truth_labels = [1] * len(encoded_test_positive) + [0] * len(encoded_test_negative)

# Evaluate the model
accuracy_model, report_model = evaluate_model(encoded_test_positive + encoded_test_negative, ground_truth_labels, model)

# Print the results
print(f"Model Accuracy: {accuracy_model}")
print("Classification Report:\n", report_model)
