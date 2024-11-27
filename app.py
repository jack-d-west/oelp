from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('data/diabetes_dataset.csv')
df_sampled = df.sample(n=10, random_state=42)  # Use random_state for reproducibility
X = df_sampled.iloc[:, :-1]
y = df_sampled['target']

# Train polynomial regression model
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Add noise functions
def add_gaussian_noise(value, epsilon, delta, sensitivity):
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, sigma, size=value.shape)
    return value + noise

def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, len(true_value))
    return true_value + noise

def add_noise_to_X(X, epsilon, noise_type):
    if noise_type == 'Gaussian':
        # Add Gaussian noise to each feature
        noisy_X = add_gaussian_noise(X.values, epsilon, delta=1e-5, sensitivity=1.0)
        return pd.DataFrame(noisy_X, columns=X.columns)
    elif noise_type == 'Laplace':
        # Apply Laplace mechanism to each column of X
        noisy_X = X.apply(lambda col: laplace_mechanism(col.values, sensitivity=1.0, epsilon=epsilon), axis=0)
        return noisy_X

def generate_noisy_y_plot_data(epsilon, noise_type):
    if noise_type == 'Gaussian':
        noisy_y = add_gaussian_noise(y.values, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_y = laplace_mechanism(y.values, 1.0, epsilon)
    return pd.Series(noisy_y)

@app.route('/')
def home():
    return render_template('index.html')  # Render the main page

@app.route('/graph1')
def graph1():
    return render_template('graph1.html')

@app.route('/graph2')
def graph2():
    return render_template('graph2.html')

@app.route('/graph3')
def graph3():
    return render_template('graph3.html')

@app.route('/graph4')
def graph4():
    return render_template('graph4.html')

@app.route('/plot', methods=['POST'])
def generate_plot():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']

    # Generate predictions
    predictions = model.predict(X_poly)
    if noise_type == 'Gaussian':
        noisy_predictions = add_gaussian_noise(predictions, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_predictions = laplace_mechanism(predictions, 1.0, epsilon)

    # Sort predictions and noisy predictions
    sorted_indices = np.argsort(predictions)
    sorted_predictions = predictions[sorted_indices].tolist()
    sorted_noisy_predictions = noisy_predictions[sorted_indices].tolist()

    # Prepare data
    data = {
        'true_values': y.tolist(),
        'original_predictions': sorted_predictions,
        'noisy_predictions': sorted_noisy_predictions
    }
    return jsonify(data)

@app.route('/plot_noisy_x', methods=['POST'])
def generate_plot_noisy_x():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']

    # Polynomial transformation on the original X
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)  # Apply polynomial transformation first
    
    # Add noise to the original X (before transformation)
    noisy_X = add_noise_to_X(X, epsilon, noise_type)

    # Apply polynomial transformation on the noisy data
    poly1 = PolynomialFeatures(degree=5)
    noisy_X_poly = poly1.fit_transform(noisy_X)

    # Train model on original (non-noisy) X_poly
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly, y)
    predictions_no_noise = model_no_noise.predict(X_poly)

    # Train model on noisy X_poly
    model_noisy_x = LinearRegression()
    model_noisy_x.fit(noisy_X_poly, y)
    predictions_noisy_x = model_noisy_x.predict(noisy_X_poly)

    # Sort predictions and true values by noisy predictions
    sorted_indices = np.argsort(predictions_noisy_x)
    sorted_predictions_no_noise = np.array(predictions_no_noise)[sorted_indices].tolist()
    sorted_predictions_noisy_x = np.array(predictions_noisy_x)[sorted_indices].tolist()

    # Prepare data for plotting
    data = {
        'true_values': y.tolist(),
        'predictions_noisy_x': sorted_predictions_noisy_x,
        'predictions_no_noise_x': sorted_predictions_no_noise
    }

    return jsonify(data)

@app.route('/plot_noisy_coeff', methods=['POST'])
def generate_plot_noisy_coeff():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']

    # Polynomial transformation on the original X
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)  # Apply polynomial transformation first

    # Train model on original (non-noisy) X_poly
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly, y)
    coeff = model_no_noise.coef_

    if noise_type == 'Gaussian':
        noisy_coeff = add_gaussian_noise(coeff, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_coeff = laplace_mechanism(coeff, 1.0, epsilon)

    intercept = model_no_noise.intercept_
    predictions_no_noise = np.dot(X_poly, coeff) + intercept
    predictions_noisy = np.dot(X_poly, noisy_coeff) + intercept

    # Sort predictions and true values by noisy predictions
    sorted_indices = np.argsort(predictions_noisy)
    sorted_predictions_noisy = np.array(predictions_noisy)[sorted_indices].tolist()
    sorted_predictions_no_noise = np.array(predictions_no_noise)[sorted_indices].tolist()

    # Prepare data for plotting
    data = {
        'true_values': y.tolist(),
        'predictions_noisy_coeff': sorted_predictions_noisy,
        'predictions_no_noise_coeff': sorted_predictions_no_noise
    }

    return jsonify(data)

@app.route('/plot_noisy_y', methods=['POST'])
def plot_noisy_y():
    data = request.get_json()
    epsilon = data['epsilon']
    noise_type = data['noise_type']

    # Generate noisy y
    noisy_y = generate_noisy_y_plot_data(epsilon, noise_type)

    # Model trained on original y
    X_poly = poly.fit_transform(X)
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly, y)
    predictions_no_noise = model_no_noise.predict(X_poly)

    # Re-train the model on the noisy y values
    model_noisy_y = LinearRegression()
    model_noisy_y.fit(X_poly, noisy_y)
    predictions_noisy_y = model_noisy_y.predict(X_poly)

    sorted_indices = np.argsort(predictions_noisy_y)
    sorted_predictions = predictions_no_noise[sorted_indices].tolist()
    sorted_noisy_predictions = predictions_noisy_y[sorted_indices].tolist()

    return jsonify({
        'true_values': y.tolist(),
        'predictions_noisy_y': sorted_noisy_predictions,
        'predictions_no_noise_y': sorted_predictions
    })


if __name__ == '__main__':
    app.run(debug=True)
