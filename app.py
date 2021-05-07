import os
import json
import requests
import urllib.parse
import pandas as pd
import sklearn
from pandas import to_datetime
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

def get_iterate(endpoint, auth_hdr, objectname, api_key):
    initial_url = 'https://api.netilion.endress.com/v1' + endpoint
    
    headers_get={'Accept': 'application/json', 'Api-key': api_key, 'Authorization': auth_hdr}
    response_get = requests.get(initial_url, headers=headers_get, verify=True)

    if response_get.status_code == 200:
        response_get_json = response_get.json()
        data = []
        for obj in response_get_json[objectname]:
            data.append(obj)
        if 'next' in response_get_json['pagination']:
            next_url = response_get_json['pagination']['next']
            while 'next' in response_get_json['pagination']:
                response_get = requests.get(next_url, headers=headers_get, verify=True)
                response_get_json = response_get.json()
                for obj in response_get_json[objectname]:
                    data.append(obj)
                if 'next' in response_get_json['pagination']:
                    next_url = response_get_json['pagination']['next']
        return data
    else:
        data = [{"msg":"Error","status_code": response_get.status_code}]
        return data

@app.route('/webhook', methods=['POST'])
def webhook():
    json_obj = request.json
    print("Request:")
    print(json.dumps(json_obj, indent=4))
    
    #Parse JSON:
    asset_id = json_obj['content']['asset']['id']
    #print('asset_id ' + str(asset_id))
    value_key = json_obj['content']['value']['key']
    #print('value_key ' + str(value_key))
    value = json_obj['content']['value']['value']
    #print('value ' + str(value))
    key = json_obj['content']['value']['key']

    #set User details required for API Authorization
    #recommendation: do this with environment variables os.getenv('ENV_VAR') instead
    b64_creds = 'your-b64_credentials'
    api_key = 'your-api-key'
    auth_hdr = "Basic: " + b64_creds

    #now we check whether the asset ID is assigned to the tag
    get_asset_instrumentations_url = 'https://api.netilion.endress.com/v1/assets/' + str(asset_id) + '/instrumentations?per_page=100'
    headers={'Accept': 'application/json', 'Api-key': api_key, 'Authorization': auth_hdr}
    get_instrumentations_response = requests.get(get_asset_instrumentations_url, headers=headers, verify=True)
    print('response status code of GET assets/n/instrumentations: ' + str(get_instrumentations_response.status_code))
    if get_instrumentations_response.status_code == 200:
        json_instrumentations = get_instrumentations_response.json()
        if json_instrumentations['instrumentations'] != []:
            #An asset can be assigned to multiple Tags in Netilion, so we iterate through the Tags:
            for instrumentation in json_instrumentations['instrumentations']:
                instrumentation_id = instrumentation['id']
                tagname = instrumentation['tag']
                if instrumentation_id == 123 and key == 'pv':
                    #now we know that the incoming data is relevant for the model that we trained in the Jupyter notebook (it was for instrumentation 51233 and for the value key "pv" standing for primary value)
                    #We want to obtain the data of the last 24h now, so we get the current datetime of the server. Watch out! If the timestamps in Netilion are in another time zone, you will have to take that into account)
                    #FYI: depending on the frequency of your data, and the time it takes for training, it may not make sense to re-train your model repeatedly.
                    #You could reduce the load by a) adding persistence so that re-training only happens once every X hours/days, or totally skip to importing pre-trained models instead of re-training based on new data repeatedly.
                    yesterday_date_time = datetime.now() - timedelta(hours = 24)
                    yesterday_string_timestamp = yesterday_date_time.strftime('%Y-%m-%dT%H:%M:%S')
                    #now we build the request string for the data:
                    #endpoint = '/instrumentations/51233/values/PV?from=' + yesterday_string_timestamp
                    endpoint = '/instrumentations/51233/values/PV?from=' + yesterday_string_timestamp
                    values_json = get_iterate(endpoint,auth_hdr,'data',api_key)
                    #all of  this is basically just copy+paste from the Jupyter Notebook:
                    df = pd.DataFrame()
                    df = df.append(values_json)
                    df['Datetime'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('Datetime')
                    df = df.resample('10T').fillna("nearest")
                    df['Ticks'] = range(0,len(df.index.values))
                    df = df.reset_index()
                    df['rolling_mean'] = df['value'].rolling(window = 15, min_periods=1).mean()
                    lasso_eps = 0.0001
                    lasso_nalpha=20
                    lasso_iter=5000
                    model = make_pipeline(PolynomialFeatures(2, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
                    model.fit(df['Ticks'].values.reshape(-1, 1),df['rolling_mean'].values.reshape(-1, 1))
                    df['predictions'] = model.predict(df['Ticks'].values.reshape(-1,1))
                    r_squared = sklearn.metrics.r2_score(df['rolling_mean'],df['predictions'],multioutput='uniform_average')
                    if r_squared > 0.98:
                        #We only want to use the model if it has an r-squared of at least 0.98
                        #We resampled the dataset to one datapoint every 10s, so to get the value prediction in 1 hours time, that is 360 datapoints in the future.
                        index_to_predict = df.index.max() + 36
                        prediction = model.predict([[index_to_predict]])
                        prediction = prediction[0]
                        text_message = urllib.parse.quote('Tag ' + tagname + ' is predicted to be at this value in 6 hours: ' + str(prediction))
                        requests.get('https://api.telegram.org/bot'+ your_telegram_API_token +'/sendMessage?chat_id=-'+ your_telegram_group_id +'&text='+text_message, verify=True)
                        
    return jsonify({"message": "Hello Netilion!"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')