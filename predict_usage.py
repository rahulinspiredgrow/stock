
from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict_usage', methods=['POST'])
def predict_usage():
    try:
        data = request.json
        item_id = data.get('item_id')
        history_data = data.get('history_data')  # List of {'date': 'YYYY-MM-DD', 'quantity': float}
        period_days = data.get('period_days', 6)

        # Validate input
        if not history_data or not item_id or not period_days:
            return jsonify({'error': 'Missing required data'}), 400

        # Prepare data for Prophet
        df = pd.DataFrame(history_data)
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['quantity']
        df = df[['ds', 'y']]

        # Initialize and fit Prophet model
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        model.fit(df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=period_days)
        forecast = model.predict(future)

        # Calculate average predicted daily usage
        predicted_usage = forecast.tail(period_days)['yhat'].mean()

        return jsonify({'item_id': item_id, 'predicted_daily_usage': max(predicted_usage, 0.01)})  # Avoid zero
    except Exception as e:
        logger.error(f"Error in predict_usage: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
