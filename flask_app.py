from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Load your dataset (replace with your own dataset)
df = pd.read_csv(r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv')
df = df.iloc[:10000]

# Preprocess categorical variables
df['country_code'] = LabelEncoder().fit_transform(df['country'])
df['job_title_code'] = LabelEncoder().fit_transform(df['job_title'])

# Selecting features and target
X = df[['is_hourly', 'hourly_low', 'hourly_high', 'country_code', 'job_title_code']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        is_hourly = bool(request.form['is_hourly'])
        hourly_low = float(request.form['hourly_low'])
        hourly_high = float(request.form['hourly_high'])
        country = request.form['country']
        job_title = request.form['job_title']

        country_code = LabelEncoder().fit_transform([country])[0]
        job_title_code = LabelEncoder().fit_transform([job_title])[0]

        input_data = [[is_hourly, hourly_low, hourly_high, country_code, job_title_code]]

        prediction = model.predict(input_data)

        return render_template('index.html', prediction=f'Predicted Salary: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)