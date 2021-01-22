# Importing required libraries
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, jsonify, request, render_template
from flask_restplus import Api, Resource, fields, Namespace
from preprocessing import predict_preprocess
import pandas as pd
import numpy as np
import pickle

# Initializing the flask app
ML_app = Flask(__name__)
api = Api(app = ML_app, version = "1.0", title = "Price Prediction.", description = "Predict price of a property based on user inputs.")

# Loading the saved model
with open('model/model.pkl',"rb") as f:
    model = pickle.load(f)

# Namespace declaration
name_space = Namespace('Predict', description='Machine Learning API')
api.add_namespace(name_space)

# Defining resource fields for documentation
resource_fields = name_space.model('User Input', {
    'borough': fields.String(description='The borough property belongs to.', required=True, enum=['1','2','3','4','5']),
    'neighbour': fields.String(description='The neighbourhood of the property.', required=True, enum=['ALPHABET CITY', 'CHELSEA', 'CHINATOWN', 'CIVIC CENTER', 'CLINTON', 'EAST VILLAGE', 'FASHION', 'FINANCIAL', 'FLATIRON', 'GRAMERCY',
                                                                                                       'GREENWICH VILLAGE-CENTRAL', 'GREENWICH VILLAGE-WEST', 'HARLEM-CENTRAL', 'HARLEM-EAST', 'HARLEM-UPPER', 'HARLEM-WEST',
                                                                                                       'INWOOD', 'JAVITS CENTER', 'KIPS BAY', 'LITTLE ITALY', 'LOWER EAST SIDE', 'MANHATTAN VALLEY', 'MIDTOWN CBD',
                                                                                                       'MIDTOWN EAST', 'MIDTOWN WEST', 'MURRAY HILL', 'ROOSEVELT ISLAND', 'SOHO', 'SOUTHBRIDGE', 'TRIBECA', 'UPPER EAST SIDE (59-79)',
                                                                                                       'UPPER EAST SIDE (79-96)', 'UPPER EAST SIDE (96-110)', 'UPPER WEST SIDE (59-79)', 'UPPER WEST SIDE (79-96)',
                                                                                                       'UPPER WEST SIDE (96-116)', 'WASHINGTON HEIGHTS LOWER', 'WASHINGTON HEIGHTS UPPER', 'BATHGATE', 'BAYCHESTER',
                                                                                                       'BEDFORD PARK/NORWOOD', 'BELMONT', 'BRONX PARK', 'BRONXDALE', 'CASTLE HILL/UNIONPORT', 'CITY ISLAND', 'CITY ISLAND-PELHAM STRIP',
                                                                                                       'CO-OP CITY', 'COUNTRY CLUB', 'CROTONA PARK', 'EAST RIVER', 'EAST TREMONT', 'FIELDSTON', 'FORDHAM',
                                                                                                       'HIGHBRIDGE/MORRIS HEIGHTS', 'HUNTS POINT', 'KINGSBRIDGE HTS/UNIV HTS', 'KINGSBRIDGE/JEROME PARK',
                                                                                                       'MELROSE/CONCOURSE', 'MORRIS PARK/VAN NEST', 'MORRISANIA/LONGWOOD', 'MOTT HAVEN/PORT MORRIS', 'MOUNT HOPE/MOUNT EDEN', 'PARKCHESTER',
                                                                                                       'PELHAM GARDENS', 'PELHAM PARKWAY NORTH', 'PELHAM PARKWAY SOUTH', 'RIVERDALE', 'SCHUYLERVILLE/PELHAM BAY', 'SOUNDVIEW',
                                                                                                       'THROGS NECK', 'WAKEFIELD', 'WESTCHESTER', 'WILLIAMSBRIDGE', 'WOODLAWN', 'BATH BEACH', 'BAY RIDGE', 'BEDFORD STUYVESANT',
                                                                                                       'BENSONHURST', 'BERGEN BEACH', 'BOERUM HILL', 'BOROUGH PARK', 'BRIGHTON BEACH', 'BROOKLYN HEIGHTS', 'BROWNSVILLE',
                                                                                                       'BUSH TERMINAL', 'BUSHWICK', 'CANARSIE', 'CARROLL GARDENS', 'CLINTON HILL', 'COBBLE HILL', 'COBBLE HILL-WEST', 'CONEY ISLAND',
                                                                                                       'CROWN HEIGHTS', 'CYPRESS HILLS', 'DOWNTOWN-FULTON FERRY', 'DOWNTOWN-FULTON MALL', 'DOWNTOWN-METROTECH', 'DYKER HEIGHTS',
                                                                                                       'EAST NEW YORK', 'FLATBUSH-CENTRAL', 'FLATBUSH-EAST', 'FLATBUSH-LEFFERTS GARDEN', 'FLATBUSH-NORTH', 'FLATLANDS',
                                                                                                       'FORT GREENE', 'GERRITSEN BEACH', 'GOWANUS', 'GRAVESEND', 'GREENPOINT', 'KENSINGTON', 'MADISON', 'MANHATTAN BEACH',
                                                                                                       'MARINE PARK', 'MIDWOOD', 'MILL BASIN', 'NAVY YARD', 'OCEAN HILL', 'OCEAN PARKWAY-NORTH', 'OCEAN PARKWAY-SOUTH', 'OLD MILL BASIN',
                                                                                                       'PARK SLOPE', 'PARK SLOPE SOUTH', 'PROSPECT HEIGHTS', 'RED HOOK', 'SEAGATE', 'SHEEPSHEAD BAY', 'SPRING CREEK', 'SUNSET PARK',
                                                                                                       'WILLIAMSBURG-CENTRAL', 'WILLIAMSBURG-EAST', 'WILLIAMSBURG-NORTH', 'WILLIAMSBURG-SOUTH', 'WINDSOR TERRACE', 'WYCKOFF HEIGHTS',
                                                                                                       'AIRPORT LA GUARDIA', 'ARVERNE', 'ASTORIA', 'BAYSIDE', 'BEECHHURST', 'BELLE HARBOR', 'BELLEROSE', 'BRIARWOOD',
                                                                                                       'BROAD CHANNEL', 'CAMBRIA HEIGHTS', 'COLLEGE POINT', 'CORONA', 'DOUGLASTON', 'EAST ELMHURST', 'ELMHURST', 'FAR ROCKAWAY',
                                                                                                       'FLORAL PARK', 'FLUSHING-NORTH', 'FLUSHING-SOUTH', 'FOREST HILLS', 'FRESH MEADOWS', 'GLEN OAKS', 'GLENDALE', 'HAMMELS', 'HILLCREST',
                                                                                                       'HOLLIS', 'HOLLIS HILLS', 'HOLLISWOOD', 'HOWARD BEACH', 'JACKSON HEIGHTS', 'JAMAICA', 'JAMAICA BAY', 'JAMAICA ESTATES',
                                                                                                       'JAMAICA HILLS', 'KEW GARDENS', 'LAURELTON', 'LITTLE NECK', 'LONG ISLAND CITY', 'MASPETH', 'MIDDLE VILLAGE', 'NEPONSIT',
                                                                                                       'OAKLAND GARDENS', 'OZONE PARK', 'QUEENS VILLAGE', 'REGO PARK', 'RICHMOND HILL', 'RIDGEWOOD', 'ROCKAWAY PARK', 'ROSEDALE',
                                                                                                       'SO. JAMAICA-BAISLEY PARK', 'SOUTH JAMAICA', 'SOUTH OZONE PARK', 'SPRINGFIELD GARDENS', 'ST. ALBANS', 'SUNNYSIDE', 'WHITESTONE',
                                                                                                       'WOODHAVEN', 'WOODSIDE', 'ANNADALE', 'ARDEN HEIGHTS', 'ARROCHAR', 'ARROCHAR-SHORE ACRES', 'BULLS HEAD', 'CASTLETON CORNERS',
                                                                                                       'CLOVE LAKES', 'CONCORD', 'CONCORD-FOX HILLS', 'DONGAN HILLS', 'DONGAN HILLS-COLONY', 'DONGAN HILLS-OLD TOWN', 'ELTINGVILLE',
                                                                                                       'EMERSON HILL', 'FRESH KILLS', 'GRANT CITY', 'GRASMERE', 'GREAT KILLS', 'GREAT KILLS-BAY TERRACE', 'GRYMES HILL',
                                                                                                       'HUGUENOT', 'LIVINGSTON', 'MANOR HEIGHTS', 'MARINERS HARBOR', 'MIDLAND BEACH', 'NEW BRIGHTON', 'NEW BRIGHTON-ST. GEORGE',
                                                                                                       'NEW DORP', 'NEW DORP-BEACH', 'NEW DORP-HEIGHTS', 'NEW SPRINGVILLE', 'OAKWOOD', 'OAKWOOD-BEACH', 'PLEASANT PLAINS',
                                                                                                       'PORT IVORY', 'PORT RICHMOND', 'PRINCES BAY', 'RICHMONDTOWN', 'RICHMONDTOWN-LIGHTHS HILL', 'ROSEBANK', 'ROSSVILLE',
                                                                                                       'ROSSVILLE-CHARLESTON', 'ROSSVILLE-PORT MOBIL', 'ROSSVILLE-RICHMOND VALLEY', 'SILVER LAKE', 'SOUTH BEACH',
                                                                                                       'STAPLETON', 'STAPLETON-CLIFTON', 'TODT HILL', 'TOMPKINSVILLE', 'TOTTENVILLE', 'TRAVIS', 'WEST NEW BRIGHTON', 'WESTERLEIGH',
                                                                                                       'WILLOWBROOK', 'WOODROW']),
    'tax_class': fields.String(description='The tax class of the property.', required=True, enum=['1','2','4']),
    'building_class': fields.String(description='The building class of the property.', required=True, enum=['C2', 'C4', 'D9', 'D1', 'C6', 'RR', 'R4', 'S3', 'S4', 'S5', 'R1', 'R8', 'K4', 'A9', 'S2', 'C0', 'R6', 'C3', 'C5', 'C1', 'C7', 'D6',
                                                                                                           'D7', 'R2', 'S9', 'O2', 'O1', 'O3', 'O5', 'K1', 'K2', 'L8', 'L1', 'H1', 'H8', 'H3', 'M1', 'N2', 'K9', 'L9', 'R5', 'GW', 'G2', 'M4',
                                                                                                           'Z9', 'B9', 'D3', 'O6', 'O4', 'L3', 'H2', 'E1', 'D5', 'B3', 'B1', 'C9', 'S1', 'A4', 'A5', 'J8', 'B2', 'F5', 'G5', 'G4', 'V1', 'M9',
                                                                                                           'O8', 'HR', 'G1', 'E7', 'I5', 'I9', 'I4', 'N9', 'H9', 'D4', 'HB', 'J4', 'A7', 'D2', 'O7', 'W3', 'HS', 'H6', 'HH', 'A1', 'F1', 'A2',
                                                                                                           'R3', 'K5', 'F4', 'G9', 'E9', 'I3', 'W4', 'I1', 'A6', 'Q8', 'V0', 'A3', 'W1', 'F2', 'F9', 'GU', 'I6', 'G8', 'P5', 'Y3', 'G7', 'I7',
                                                                                                           'M2', 'K7', 'G0', 'W2', 'P8', 'K6', 'W9', 'S0', 'P9', 'K8', 'P6', 'M3', 'A0', 'E2', 'O9', 'P2', 'H4', 'J1', 'K3', 'W8', 'Z0', 'CM']),
    'zip_code': fields.String(description='The zip code of the area of the property.', required=True, enum=['10009', '10002', '10011', '10001', '10014', '10013', '10007', '10036', '10019', '10018', '10003', '10005', '10004', '10006', '10038', '10010', '10016', '10012',
                                                                                                           '10030', '10031', '10027', '10037', '10035', '10026', '10039', '10029', '10032', '10463', '10034', '10025', '10022', '10017', '10167', '10044', '10280', '10282',
                                                                                                           '10065', '10021', '10075', '10028', '10128', '10023', '10024', '10069', '10033', '10040', '10457', '10458', '10451', '10456', '10469', '10466', '10475', '10468',
                                                                                                           '10467', '10460', '10473', '10472', '10464', '10803', '10465', '10471', '10453', '10452', '10474', '10455', '10462', '10461', '10459', '10454', '10470', '11228',
                                                                                                           '11214', '11220', '11209', '11219', '11221', '11233', '11213', '11206', '11216', '11205', '11238', '11204', '11234', '11217', '11201', '11218', '11232', '11230',
                                                                                                           '11235', '11212', '11237', '11207', '11236', '11239', '11231', '11249', '11224', '11225', '11208', '11226', '11210', '11203', '11229', '11215', '11223', '11222',
                                                                                                           '11211', '11370', '11691', '11692', '11106', '11103', '11377', '11105', '11102', '11101', '11361', '11360', '11357', '11358', '11364', '11694', '11426', '11428',
                                                                                                           '11004', '11001', '11435', '11693', '11411', '11356', '11354', '11420', '11368', '11373', '11363', '11362', '11369', '11378', '11372', '11040', '11355', '11365',
                                                                                                           '11367', '11366', '11375', '11374', '11005', '11385', '11421', '11379', '11432', '11423', '11412', '11427', '11429', '11433', '11414', '11422', '11418', '11415',
                                                                                                           '11413', '11109', '11104', '11416', '11417', '11419', '11436', '11434', '10312', '10305', '10314', '10303', '10310', '10302', '10301', '10304', '10306', '10308', '10309', '10307']),
    'residential units': fields.Integer(description='Number of residential units in the property', required=True, min=0),
    'commercial units': fields.Integer(description='Number of commercial units in the property', required=True, min=0),
    'Land sqft': fields.Integer(description='Land area the property in square foot', required=True, min=10),
    'Gross sqft': fields.Integer(description='Gross area the property in square foot', required=True, min=10),
})

# defining the response of the API for Documentation
response_field = name_space.model('Output_prediction', {
    'Predicted Price': fields.String(description='Predicted Price by the model.')
})

# Class for making prediction which inherit from the Resource model in flask_restful
class MakePrediction(Resource):
    @staticmethod

    @api.response(200, 'Price is predicted.', model = response_field)
    @api.response(400, 'Something went wrong with prediction.')

    @api.expect(resource_fields)

    def post():
        # Fetch attributes from the user inputs
        posted_data = request.get_json()
        posted_data = pd.DataFrame(posted_data, index=[0])

        # Preprocessing user inputs
        data = predict_preprocess(posted_data)

        # Predict for the new features based on the model loaded
        prediction = model.predict(data)

        # converting array output to a string
        prediction = np.array2string(prediction)
        prediction = prediction[-(len(prediction)-1):-1]

        # Return json representation of the prediction
        return jsonify({
            'Predicted price': prediction
        })

# Declare the route to handle the request
api.add_resource(MakePrediction, '/predict')

# Run the app in debug mode
if __name__ == '__main__':
    ML_app.run(debug=True)