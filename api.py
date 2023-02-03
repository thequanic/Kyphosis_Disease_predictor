from flask import Flask, jsonify, request
from flask_restful import Resource, Api,reqparse
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
api=Api(app)

CORS(app)

parser = reqparse.RequestParser()
parser.add_argument('age', type=int, required=True, help="Age cannot be left blank!")
parser.add_argument('number', type=int, required=True, help="Number cannot be left blank!")
parser.add_argument('start', type=int, required=True, help="Start cannot be left blank!")


class prediction(Resource):
    def __init__(self):
        with open("E://vsc2.0//GitHub//Kyphosis_Disease_Detection_using_different_classification_models//model.pkl", 'rb') as file:
            self.dtree = pickle.load(file)
    def post(self):
        args = parser.parse_args()
        x=pd.DataFrame({'Age': [args['age']], 'Number': [args['number']], 'Start': [args['start']]})
        y=self.dtree.predict(x)
        return jsonify({'result': str(y[0])})


api.add_resource(prediction, '/predict')
  

if __name__ == '__main__':
  
    app.run(port=3000,debug = True)