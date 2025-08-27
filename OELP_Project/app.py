from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('data/diabetes_dataset.csv')  # Ensure this file exists in the specified path

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
        noisy_X = add_gaussian_noise(X.values, epsilon, delta=1e-5, sensitivity=1.0)
        return pd.DataFrame(noisy_X, columns=X.columns)
    elif noise_type == 'Laplace':
        noisy_X = X.apply(lambda col: laplace_mechanism(col.values, sensitivity=1.0, epsilon=epsilon), axis=0)
        return noisy_X

def clip_data(data, min_value, max_value):
    return np.clip(data, min_value, max_value)

# Routes for different graphs
@app.route('/')
def home():
    return render_template('i.html')  # Main page

@app.route('/ab')
def abus():
    return render_template('about.html')

@app.route('/r')
def r():
    return render_template('resource.html')

@app.route('/cus')
def cus():
    return render_template('cus.html')

@app.route('/ru')
def ru():
    return render_template('ru.html')

@app.route('/i')
def i():
    return render_template('index.html')

@app.route('/contri')
def cont():
    return render_template('cont.html')

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
    sample_size = int(request.json['sample_size'])  # Default 10 samples
    clipping = request.json.get('clipping', 'yes')
    min_value = float(request.json.get('min_value', -np.inf))
    max_value = float(request.json.get('max_value', np.inf))

    # Sample data
    sampled_data = df.sample(n=sample_size, random_state=42)
    X_sampled = sampled_data.iloc[:, :-1]
    y_sampled = sampled_data['target']

    # Train polynomial regression model
    poly = PolynomialFeatures(degree=5)
    X_poly_sampled = poly.fit_transform(X_sampled)
    model = LinearRegression()
    model.fit(X_poly_sampled, y_sampled)

    # Generate predictions
    predictions = model.predict(X_poly_sampled)
    if noise_type == 'Gaussian':
        noisy_predictions = add_gaussian_noise(predictions, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_predictions = laplace_mechanism(predictions, 1.0, epsilon)

    # Apply clipping
    if clipping.lower() != 'no':
        predictions = clip_data(predictions, min_value, max_value)
        noisy_predictions = clip_data(noisy_predictions, min_value, max_value)

    # Sort predictions
    sorted_indices = np.argsort(predictions)
    sorted_predictions = predictions[sorted_indices].tolist()
    sorted_noisy_predictions = noisy_predictions[sorted_indices].tolist()

    return jsonify({
        'true_values': y_sampled.tolist(),
        'original_predictions': sorted_predictions,
        'noisy_predictions': sorted_noisy_predictions
    })

@app.route('/plot_noisy_x', methods=['POST'])
def generate_plot_noisy_x():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']
    sample_size = int(request.json['sample_size'])  # Default 10 samples
    clipping = request.json.get('clipping', 'yes')
    min_value = float(request.json.get('min_value', -np.inf))
    max_value = float(request.json.get('max_value', np.inf))

    # Sample data
    sampled_data = df.sample(n=sample_size, random_state=42)
    X_sampled = sampled_data.iloc[:, :-1]
    y_sampled = sampled_data['target']

    # Polynomial transformation
    poly = PolynomialFeatures(degree=5)
    X_poly_sampled = poly.fit_transform(X_sampled)
    noisy_X = add_noise_to_X(X_sampled, epsilon, noise_type)
    noisy_X_poly = poly.fit_transform(noisy_X)

    # Train models
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly_sampled, y_sampled)
    predictions_no_noise = model_no_noise.predict(X_poly_sampled)

    model_noisy_x = LinearRegression()
    model_noisy_x.fit(noisy_X_poly, y_sampled)
    predictions_noisy_x = model_noisy_x.predict(noisy_X_poly)

    # Apply clipping
    if clipping.lower() != 'no':
        predictions_no_noise = clip_data(predictions_no_noise, min_value, max_value)
        predictions_noisy_x = clip_data(predictions_noisy_x, min_value, max_value)

    # Sort predictions
    sorted_indices = np.argsort(predictions_noisy_x)
    sorted_predictions_no_noise = predictions_no_noise[sorted_indices].tolist()
    sorted_predictions_noisy_x = predictions_noisy_x[sorted_indices].tolist()

    return jsonify({
        'true_values': y_sampled.tolist(),
        'predictions_noisy_x': sorted_predictions_noisy_x,
        'predictions_no_noise_x': sorted_predictions_no_noise
    })

@app.route('/plot_noisy_coeff', methods=['POST'])
def generate_plot_noisy_coeff():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']
    sample_size = int(request.json['sampling_rate'])  # Default 10 samples
    clipping = request.json.get('clipping', 'yes')
    min_value = float(request.json.get('min_value', -np.inf))
    max_value = float(request.json.get('max_value', np.inf))

    # Sample data
    sampled_data = df.sample(n=sample_size, random_state=42)
    X_sampled = sampled_data.iloc[:, :-1]
    y_sampled = sampled_data['target']

    # Polynomial transformation
    poly = PolynomialFeatures(degree=5)
    X_poly_sampled = poly.fit_transform(X_sampled)

    # Train model on original X_poly
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly_sampled, y_sampled)
    coeff = model_no_noise.coef_

    if noise_type == 'Gaussian':
        noisy_coeff = add_gaussian_noise(coeff, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_coeff = laplace_mechanism(coeff, 1.0, epsilon)

    intercept = model_no_noise.intercept_
    predictions_no_noise = np.dot(X_poly_sampled, coeff) + intercept
    predictions_noisy = np.dot(X_poly_sampled, noisy_coeff) + intercept

    # Apply clipping
    if clipping.lower() != 'no':
        predictions_no_noise = clip_data(predictions_no_noise, min_value, max_value)
        predictions_noisy = clip_data(predictions_noisy, min_value, max_value)

    # Sort predictions
    sorted_indices = np.argsort(predictions_noisy)
    sorted_predictions_noisy = predictions_noisy[sorted_indices].tolist()
    sorted_predictions_no_noise = predictions_no_noise[sorted_indices].tolist()

    return jsonify({
        'true_values': y_sampled.tolist(),
        'predictions_noisy_coeff': sorted_predictions_noisy,
        'predictions_no_noise_coeff': sorted_predictions_no_noise
    })

@app.route('/plot_noisy_y', methods=['POST'])
def plot_noisy_y():
    epsilon = float(request.json['epsilon'])
    noise_type = request.json['noise_type']
    sample_size = int(request.json['sampling_size'])  # Default 10 samples
    clipping = request.json.get('clipping', 'yes')
    min_value = float(request.json.get('min_value', -np.inf))
    max_value = float(request.json.get('max_value', np.inf))

    # Sample data
    sampled_data = df.sample(n=sample_size, random_state=42)
    X_sampled = sampled_data.iloc[:, :-1]
    y_sampled = sampled_data['target']

    # Generate noisy y
    if noise_type == 'Gaussian':
        noisy_y = add_gaussian_noise(y_sampled.values, epsilon, 1e-5, 1.0)
    elif noise_type == 'Laplace':
        noisy_y = laplace_mechanism(y_sampled.values, 1.0, epsilon)

    # Train models on original y and noisy y
    poly = PolynomialFeatures(degree=5)
    X_poly_sampled = poly.fit_transform(X_sampled)
    model_no_noise = LinearRegression()
    model_no_noise.fit(X_poly_sampled, y_sampled)
    predictions_no_noise = model_no_noise.predict(X_poly_sampled)

    model_noisy_y = LinearRegression()
    model_noisy_y.fit(X_poly_sampled, noisy_y)
    predictions_noisy_y = model_noisy_y.predict(X_poly_sampled)

    # Apply clipping
    if clipping.lower() != 'no':
        predictions_no_noise = clip_data(predictions_no_noise, min_value, max_value)
        predictions_noisy_y = clip_data(predictions_noisy_y, min_value, max_value)

    # Sort predictions
    sorted_indices = np.argsort(predictions_noisy_y)
    sorted_predictions_noisy_y = predictions_noisy_y[sorted_indices].tolist()
    sorted_predictions_no_noise = predictions_no_noise[sorted_indices].tolist()

    return jsonify({
        'true_values': y_sampled.tolist(),
        'predictions_noisy_y': sorted_predictions_noisy_y,
        'predictions_no_noise_y': sorted_predictions_no_noise
    })

if __name__ == '__main__':
    app.run(debug=True)
