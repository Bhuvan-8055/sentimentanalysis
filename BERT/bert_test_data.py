import torch
from sklearn.model_selection import train_test_split

# ✅ Load complete dataset (Fixed Unpickling Error)
tokenized_data = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_tokenized_data.pt", weights_only=False)
labels = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_labels.pt", weights_only=False)
emoji_features = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_emoji_features.pt", weights_only=False)

# ✅ Convert tokenized data to tensors
input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]

# ✅ Split into train & test sets (80% train, 20% test)
train_ids, test_ids, train_mask, test_mask, train_labels, test_labels, train_emoji, test_emoji = train_test_split(
    input_ids, attention_mask, labels, emoji_features, test_size=0.2, random_state=42
)

# ✅ Save training dataset
torch.save({"input_ids": train_ids, "attention_mask": train_mask}, r"C:\Users\sribh\OneDrive\Desktop\bert_train_tokenized.pt")
torch.save(train_labels, r"C:\Users\sribh\OneDrive\Desktop\bert_train_labels.pt")
torch.save(train_emoji, r"C:\Users\sribh\OneDrive\Desktop\bert_train_emoji_features.pt")

# ✅ Save test dataset
torch.save({"input_ids": test_ids, "attention_mask": test_mask}, r"C:\Users\sribh\OneDrive\Desktop\bert_test_tokenized.pt")
torch.save(test_labels, r"C:\Users\sribh\OneDrive\Desktop\bert_test_labels.pt")
torch.save(test_emoji, r"C:\Users\sribh\OneDrive\Desktop\bert_test_emoji_features.pt")

print("\n✅ Dataset Split Complete! Train & Test data saved successfully.")
