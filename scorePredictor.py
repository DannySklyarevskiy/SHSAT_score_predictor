import torch
import pandas as pd
import joblib

from scorePredictorTrainer import SHSATModel, categorical_columns, numerical_columns

model = SHSATModel()
model.load_state_dict(torch.load('shsat_model.pth'))
model.eval()  

encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame({
    'annual_household_income': [50000, 150000],  
    'gpa': [80, 95],                             
    'race': ['Latino', 'White'],                 
    'gender': ['Male', 'Female'],                
    'middle_school_type': ['Public', 'Private'], 
    'favorite_class': ['Science', 'Math'],       
    'least_favorite_class': ['ELA', 'Art'],      
    'learning_about_shsat_date': [2022, 2023],   
    'shsat_practice_taken': [True, True],       
    'shsat_private_tutor': [False, True],        
    'dream_school': ['SIT', 'Stuyvesant']        
})

encoded_new_categorical_data = encoder.transform(new_data[categorical_columns])

new_data = new_data.drop(columns=categorical_columns)
encoded_new_features = pd.DataFrame(encoded_new_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))
new_data = pd.concat([new_data, encoded_new_features], axis=1)

new_data = new_data[numerical_columns + list(encoded_new_features.columns)]

new_data_scaled = scaler.transform(new_data)

new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

with torch.no_grad():
    predictions = model(new_data_tensor)

predicted_scores = predictions.numpy()
print(predicted_scores)
