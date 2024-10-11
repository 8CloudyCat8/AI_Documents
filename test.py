import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from PyQt5 import QtWidgets, QtCore

# Hyperparameters
BATCH_SIZE = 64
EMBED_DIM = 64
EPOCHS = 1
LR = 5.0
MODEL_PATH = 'text_classification_model.pth'
classes = ["World", "Sports", "Business", "Science/Technology"]

# Load the dataset (AG_NEWS)
train_iter, test_iter = AG_NEWS()

# Tokenizer and Vocabulary
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Create text and label pipelines
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Model definition
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Prepare data
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# Load datasets and split into train/validation sets
train_dataset = to_map_style_dataset(train_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Get number of classes
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)

# Function to train model
def train_model(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    def train_epoch(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        return total_acc / total_count

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    for epoch in range(1, EPOCHS + 1):
        train_acc = train_epoch(train_dataloader)
        valid_acc = evaluate(valid_dataloader)
        print(f'Epoch: {epoch}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

# Load model
def load_model(model_class):
    model = model_class(vocab_size, EMBED_DIM, num_class)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Model loaded!")
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        train_model(model)
    return model

# PyQt5 Application
class TextClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text Classification')

        # Layout
        layout = QtWidgets.QVBoxLayout()

        self.model_choice = QtWidgets.QComboBox(self)
        self.model_choice.addItem("Basic Model")
        layout.addWidget(self.model_choice)

        self.input_text = QtWidgets.QLineEdit(self)
        self.input_text.setPlaceholderText('Enter text here...')
        layout.addWidget(self.input_text)

        self.predict_button = QtWidgets.QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.predict_text)
        layout.addWidget(self.predict_button)

        self.train_button = QtWidgets.QPushButton('Train Model', self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.result_label = QtWidgets.QLabel('Prediction will appear here', self)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Load the default model
        self.model = load_model(TextClassificationModel)

    def predict_text(self):
        text = self.input_text.text()
        if text:
            with torch.no_grad():
                text_tensor = torch.tensor(text_pipeline(text))
                output = self.model(text_tensor, torch.tensor([0]))
                predicted_index = output.argmax(1).item()
                predicted_class = classes[predicted_index]
                self.result_label.setText(f'Predicted class: {predicted_class}')
        else:
            self.result_label.setText('Please enter text to classify.')

    def train_model(self):
        # Train the model on button click
        model_class = TextClassificationModel if self.model_choice.currentText() == "Basic Model" else TextClassificationModel
        self.model = model_class(vocab_size, EMBED_DIM, num_class)
        train_model(self.model)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    classifier_app = TextClassifierApp()
    classifier_app.resize(400, 200)
    classifier_app.show()
    sys.exit(app.exec_())
