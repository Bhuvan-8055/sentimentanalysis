import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ✅ Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Evaluating on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ✅ Load test data (Ensure weights_only=False)
tokenized_test = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_test_tokenized.pt", map_location=device, weights_only=False)
test_labels = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_test_labels.pt", map_location=device, weights_only=False)
test_emoji_features = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_test_emoji_features.pt", map_location=device, weights_only=False)

# ✅ Load the saved model to extract correct num_labels
saved_model = torch.load(r"C:\Users\sribh\OneDrive\Desktop\bert_with_emoji.pth", map_location=device)
num_labels = saved_model["classifier.weight"].shape[0]  # ✅ Automatically extract label count
print(f"🔹 Automatically detected {num_labels} labels from saved model.")

# ✅ Define the same model structure as before
class BertWithEmoji(nn.Module):
    def __init__(self, bert_model_name, num_labels, emoji_feature_size):
        super(BertWithEmoji, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_text = nn.Linear(self.bert.config.hidden_size, 128)  # Text feature layer
        self.fc_emoji = nn.Linear(emoji_feature_size, 32)  # Emoji feature layer
        self.classifier = nn.Linear(128 + 32, num_labels)  # Combined classification layer

    def forward(self, input_ids, attention_mask, emoji_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.fc_text(bert_output.pooler_output)  # Extract text-based features
        emoji_features = self.fc_emoji(emoji_features)  # Process emoji features
        combined_features = torch.cat((text_features, emoji_features), dim=1)  # Concatenate both features
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)  # Classification output
        return logits

# ✅ Model parameters
bert_model_name = "bert-base-uncased"
emoji_feature_size = test_emoji_features.shape[1]

# ✅ Load the trained model
model = BertWithEmoji(bert_model_name, num_labels, emoji_feature_size).to(device)
model.load_state_dict(saved_model)  # ✅ Load saved weights
model.eval()  # ✅ Set model to evaluation mode

# ✅ Create test DataLoader
batch_size = 32
test_dataset = TensorDataset(tokenized_test["input_ids"], tokenized_test["attention_mask"], test_emoji_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ✅ Evaluate the model
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="🔄 Evaluating..."):
        input_ids, attention_mask, emoji_features, batch_labels = [x.to(device) for x in batch]

        logits = model(input_ids, attention_mask, emoji_features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

# ✅ Compute evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

print(f"\n✅ Model Evaluation Complete!")
print(f"🎯 Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", report)
