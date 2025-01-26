from flask import Flask

# Initialize the Flask app
app = Flask('ping')

# Define a route for the ping service
@app.route('/ping', methods=['GET'])
def ping():
    return "Pong!"

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)