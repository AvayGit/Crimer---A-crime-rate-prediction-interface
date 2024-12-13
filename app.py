from flask import Flask, render_template, request
import joblib
import plotly.graph_objects as go

app = Flask(__name__)

# Load the saved ARIMA model
arima_model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the form (number of years to predict)
        years_to_predict = int(request.form.get('years'))

        # Validate that the number of years is a positive integer
        if years_to_predict <= 0:
            raise ValueError("The number of years must be positive.")

        # Forecast the crime rates for the next 'years_to_predict' years
        forecast = arima_model.forecast(steps=years_to_predict)
        forecast_values = forecast.tolist()

        # Create a Plotly graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, years_to_predict + 1)),
            y=forecast_values,
            mode='lines+markers',
            name='Predicted Crime Rates'
        ))

        # Convert Plotly figure to HTML
        graph_html = fig.to_html(full_html=False)

        # Return the forecasted values and the Plotly graph
        return render_template('index.html', prediction=forecast_values, years=years_to_predict, graph_html=graph_html)

    except ValueError as e:
        # If there's an error (e.g., invalid input or negative years), display the error message
        return render_template('index.html', error_message=str(e))
