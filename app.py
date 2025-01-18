from flask import Flask,request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('pipe.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    
    # Extract input values
    batting_team = data.get('batting_team')
    bowling_team = data.get('bowling_team')
    runs_left = data.get('runs_left')
    balls_left = data.get('balls_left')
    wickets = data.get('wickets')
    total_runs_x = data.get('total_runs_x')
    crr = data.get('crr')
    rrr = data.get('rrr')

    # Create a DataFrame for the model
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [total_runs_x],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Batting team wins" if prediction >=0.5 else "Bowling team wins"

    # Return the prediction result
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
