import os
import random
import requests
import pickle
import pandas as pd

# Replace with your actual Google API key.
GOOGLE_API_KEY = "AIzaSyBiq620-FjtVuJZaZ8emjsPbTNyMCS2-no"

# Define a list of random error messages.
ERROR_MESSAGES = [
    "Oops! Something went wrong.",
    "Unexpected error occurred. Please try again.",
    "Error encountered. Please retry.",
    "Something went wrong. We apologize for the inconvenience.",
    "Our apologies, an error occurred.",
    "Unexpected hiccup encountered. Try again later.",
    "Error detected. Please check your input.",
    "We encountered an error. Please try once more.",
    "Oops, something didn't work as expected.",
    "A problem occurred. Please try again soon."
]

def random_error_response(status_code=500):
    # For the purpose of this script, just raise an exception with a random error.
    raise Exception(random.choice(ERROR_MESSAGES))

def get_county_from_coordinates_google(lat, lon, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return random_error_response(500)
    
    data = response.json()
    if data['status'] != 'OK':
        return random_error_response(500)
    
    results = data.get('results', [])
    county = None
    state = None
    country = None
    
    for result in results:
        for component in result['address_components']:
            if 'administrative_area_level_2' in component['types']:
                county = component['long_name']
            elif 'administrative_area_level_1' in component['types']:
                state = component['long_name']
            elif 'country' in component['types']:
                country = component['long_name']
    
    if county and county.endswith(" County"):
        county = county.replace(" County", "").strip()
    
    return {
        'county': county,  # Will be None if not found.
        'state': state or 'State not found',
        'country': country or 'Country not found',
        'full_address': results[0]['formatted_address'] if results else 'N/A'
    }

# Load population data once at startup.
population_file_path = os.path.join(os.getcwd(), 'population.csv')
try:
    population_data = pd.read_csv(population_file_path)
except Exception:
    raise Exception(random.choice(ERROR_MESSAGES))

# Determine the correct population column name.
pop_col = None
for col in population_data.columns:
    if col.lower() in ['population', 'total population']:
        pop_col = col
        break
if pop_col is None:
    raise KeyError("Population column not found in population.csv. Please ensure it has a 'Population' or 'Total Population' column.")

# Load the saved staff prediction model.
model_file_path = os.path.join(os.getcwd(), 'staff_prediction_model.pkl')
with open(model_file_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoders = model_data['label_encoders']
scaler = model_data['scaler']
expected_columns = ['DisasterTypeID', 'SeverityTypeID', 'Population', 'County']

def predict_staff_required(county, disaster_type_id, severity_type_id, population):
    """
    Preprocess the input and predict the staff required.
    """
    input_data = pd.DataFrame([[disaster_type_id, severity_type_id, population, county]], 
                              columns=expected_columns)
    
    for col in ['DisasterTypeID', 'SeverityTypeID', 'County']:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError:
                # If the input value was not seen during training, assign a default value.
                input_data[col] = 0  
    input_data['Population'] = scaler.transform(input_data[['Population']])
    prediction = model.predict(input_data)
    return int(prediction[0])

if __name__ == '__main__':
    # Hardcoded inputs
    disaster_type_id = 2
    severity_type_id = 1
    location = {"lat": 32.67945519197781, "lng": -97.23876982927324}
    
    # Extract latitude and longitude from location.
    lat = location.get('lat')
    lon = location.get('lng')
    
    # Validate and convert lat and lon.
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        raise Exception("Invalid latitude or longitude value.")
    
    # Ensure longitude is negative if required.
    if lon > 0:
        lon = lon * -1

    # Get county information using Google Geocoding API.
    county_data = get_county_from_coordinates_google(lat, lon, GOOGLE_API_KEY)
    county = county_data['county']
    if not county:
        raise Exception("County not found for the given coordinates.")
    
    # Lookup population by county (case insensitive).
    population_row = population_data[population_data['county'].str.lower() == county.lower()]
    if population_row.empty:
        raise Exception("Population data not found for county: " + county)
    
    try:
        population = float(population_row.iloc[0][pop_col])
    except Exception:
        raise Exception("Error retrieving population data.")

    # Convert disaster and severity type IDs to integers.
    try:
        disaster_type_id = int(disaster_type_id)
        severity_type_id = int(severity_type_id)
    except ValueError:
        raise Exception("Invalid disaster or severity type ID.")

    # Make the prediction.
    try:
        predicted_staff = predict_staff_required(county, disaster_type_id, severity_type_id, population)
    except Exception as e:
        raise Exception("Prediction error: " + str(e))

    # Output the result.
    print("Predicted staff required:", predicted_staff)
