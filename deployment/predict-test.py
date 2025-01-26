import requests  # Ensure the requests library is imported

url = 'http://localhost:9696/predict'

order_id = 'xyz-123'

order = {
    "customer": "anne_pryor",
    "manufactory": "xerox",
    "product_name_encoded": 1743,
    "segment": "home_office",
    "category": "office_supplies",
    "subcategory": "paper",
    "region": "south",
    "city": "florence",
    "state": "alabama",
    "quantity": 1,
    "sales": 1.7884205679625405,
    "profit_margin": 0.49,
    "order_year": 2021,
    "order_month": 5,
    "ship_year": 2021,
    "ship_month": 5
}

# Send POST request
try:
    response = requests.post(url, json=order)  # Send the request
    response.raise_for_status()  # Check for HTTP request errors

    # Safely handle JSON decoding
    response_json = response.json()

    # Process response
    if 'profit' in response_json and response_json['profit']:
        print(f'This order is profitable: {order_id}')
    else:
        print(f'This order is not profitable: {order_id}')
except requests.exceptions.RequestException as e:
    print(f"An error occurred while making the request: {e}")
except ValueError as e:
    print(f"An error occurred while decoding the response: {e}")

