import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

file_path = 'data.csv'
data = pd.read_csv(file_path)

features = data.drop(columns=['shsat_total_score'])
target = data['shsat_total_score']

categorical_columns = ['race', 'gender', 'middle_school_type', 'favorite_class', 'least_favorite_class', 'dream_school',
                       'learning_about_shsat_date', 'shsat_practice_taken', 'shsat_private_tutor']
numerical_columns = ['annual_household_income', 'gpa']

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_categorical_data = encoder.fit_transform(features[categorical_columns])

features = features.drop(columns=categorical_columns)
encoded_features = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))
features = pd.concat([features, encoded_features], axis=1)

features = features[numerical_columns + list(encoded_features.columns)]

X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


class SHSATModel(nn.Module):
    def __init__(self):
        super(SHSATModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x


model = SHSATModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'shsat_model.pth')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
