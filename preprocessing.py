#Importing required libraries
import pandas as pd
import pickle

# Function to perform preprocessing on predict data
def predict_preprocess(user_data):

    # Define columns for empty dataframe
    with open('model/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)

    # Define an empty dataframe to save user input data
    user_input = pd.DataFrame(columns=model_columns)

    # Defining function to assign value to exact dummy value for user input of 'borough'
    def borough_dummy(val):

        data_dic = {'1': 'BOROUGH_1', '2': 'BOROUGH_2', '3':'BOROUGH_3', '4':'BOROUGH_4', '5':'BOROUGH_5'}

        return data_dic.get(val)

    # Defining function to assign value to exact dummy value for user input of 'neighbourhood'
    def neighbour_dummy(val):

        data_dic = {'AIRPORT LA GUARDIA':'NEIGHBORHOOD_AIRPORT LA GUARDIA', 'ALPHABET CITY': 'NEIGHBORHOOD_ALPHABET CITY', 'ANNADALE': 'NEIGHBORHOOD_ANNADALE', 'ARDEN HEIGHTS': 'NEIGHBORHOOD_ARDEN HEIGHTS','ARROCHAR': 'NEIGHBORHOOD_ARROCHAR',
                    'ARROCHAR-SHORE ACRES': 'NEIGHBORHOOD_ARROCHAR-SHORE ACRES', 'ARVERNE': 'NEIGHBORHOOD_ARVERNE', 'ASTORIA': 'NEIGHBORHOOD_ASTORIA', 'BATH BEACH': 'NEIGHBORHOOD_BATH BEACH', 'BATHGATE': 'NEIGHBORHOOD_BATHGATE',
                    'BAY RIDGE': 'NEIGHBORHOOD_BAY RIDGE', 'BAYCHESTER': 'NEIGHBORHOOD_BAYCHESTER', 'BAYSIDE': 'NEIGHBORHOOD_BAYSIDE', 'BEDFORD PARK/NORWOOD': 'NEIGHBORHOOD_BEDFORD PARK/NORWOOD', 'BEDFORD STUYVESANT': 'NEIGHBORHOOD_BEDFORD STUYVESANT',
                    'BEECHHURST': 'NEIGHBORHOOD_BEECHHURST', 'BELLE HARBOR': 'NEIGHBORHOOD_BELLE HARBOR', 'BELLEROSE': 'NEIGHBORHOOD_BELLEROSE', 'BELMONT': 'NEIGHBORHOOD_BELMONT', 'BENSONHURST': 'NEIGHBORHOOD_BENSONHURST', 'BERGEN BEACH': 'NEIGHBORHOOD_BERGEN BEACH',
                    'BOERUM HILL': 'NEIGHBORHOOD_BOERUM HILL', 'BOROUGH PARK': 'NEIGHBORHOOD_BOROUGH PARK', 'BRIARWOOD': 'NEIGHBORHOOD_BRIARWOOD', 'BRIGHTON BEACH': 'NEIGHBORHOOD_BRIGHTON BEACH', 'BROAD CHANNEL': 'NEIGHBORHOOD_BROAD CHANNEL', 'BRONX PARK': 'NEIGHBORHOOD_BRONX PARK',
                    'BRONXDALE': 'NEIGHBORHOOD_BRONXDALE', 'BROOKLYN HEIGHTS': 'NEIGHBORHOOD_BROOKLYN HEIGHTS', 'BROWNSVILLE': 'NEIGHBORHOOD_BROWNSVILLE', 'BULLS HEAD': 'NEIGHBORHOOD_BULLS HEAD', 'BUSH TERMINAL': 'NEIGHBORHOOD_BUSH TERMINAL',
                    'BUSHWICK': 'NEIGHBORHOOD_BUSHWICK', 'CAMBRIA HEIGHTS': 'NEIGHBORHOOD_CAMBRIA HEIGHTS', 'CANARSIE': 'NEIGHBORHOOD_CANARSIE', 'CARROLL GARDENS': 'NEIGHBORHOOD_CARROLL GARDENS', 'CASTLE HILL/UNIONPORT': 'NEIGHBORHOOD_CASTLE HILL/UNIONPORT',
                    'CASTLETON CORNERS': 'NEIGHBORHOOD_CASTLETON CORNERS', 'CHELSEA': 'NEIGHBORHOOD_CHELSEA', 'CHINATOWN': 'NEIGHBORHOOD_CHINATOWN', 'CITY ISLAND': 'NEIGHBORHOOD_CITY ISLAND', 'CITY ISLAND-PELHAM STRIP': 'NEIGHBORHOOD_CITY ISLAND-PELHAM STRIP',
                    'CIVIC CENTER': 'NEIGHBORHOOD_CIVIC CENTER', 'CLINTON': 'NEIGHBORHOOD_CLINTON', 'CLINTON HILL': 'NEIGHBORHOOD_CLINTON HILL', 'CLOVE LAKES': 'NEIGHBORHOOD_CLOVE LAKES', 'CO-OP CITY': 'NEIGHBORHOOD_CO-OP CITY', 'COBBLE HILL': 'NEIGHBORHOOD_COBBLE HILL',
                    'COBBLE HILL-WEST': 'NEIGHBORHOOD_COBBLE HILL-WEST', 'COLLEGE POINT': 'NEIGHBORHOOD_COLLEGE POINT', 'CONCORD': 'NEIGHBORHOOD_CONCORD', 'CONCORD-FOX HILLS': 'NEIGHBORHOOD_CONCORD-FOX HILLS', 'CONEY ISLAND': 'NEIGHBORHOOD_CONEY ISLAND',
                    'CORONA': 'NEIGHBORHOOD_CORONA', 'COUNTRY CLUB': 'NEIGHBORHOOD_COUNTRY CLUB', 'CROTONA PARK': 'NEIGHBORHOOD_CROTONA PARK', 'CROWN HEIGHTS': 'NEIGHBORHOOD_CROWN HEIGHTS', 'CYPRESS HILLS': 'NEIGHBORHOOD_CYPRESS HILLS', 'DONGAN HILLS': 'NEIGHBORHOOD_DONGAN HILLS',
                    'DONGAN HILLS-COLONY': 'NEIGHBORHOOD_DONGAN HILLS-COLONY', 'DONGAN HILLS-OLD TOWN': 'NEIGHBORHOOD_DONGAN HILLS-OLD TOWN', 'DOUGLASTON': 'NEIGHBORHOOD_DOUGLASTON', 'DOWNTOWN-FULTON FERRY': 'NEIGHBORHOOD_DOWNTOWN-FULTON FERRY',
                    'DOWNTOWN-FULTON MALL': 'NEIGHBORHOOD_DOWNTOWN-FULTON MALL', 'DOWNTOWN-METROTECH': 'NEIGHBORHOOD_DOWNTOWN-METROTECH', 'DYKER HEIGHTS': 'NEIGHBORHOOD_DYKER HEIGHTS', 'EAST ELMHURST': 'NEIGHBORHOOD_EAST ELMHURST', 'EAST NEW YORK': 'NEIGHBORHOOD_EAST NEW YORK',
                    'EAST RIVER': 'NEIGHBORHOOD_EAST RIVER', 'EAST TREMONT': 'NEIGHBORHOOD_EAST TREMONT', 'EAST VILLAGE': 'NEIGHBORHOOD_EAST VILLAGE', 'ELMHURST': 'NEIGHBORHOOD_ELMHURST', 'ELTINGVILLE': 'NEIGHBORHOOD_ELTINGVILLE', 'EMERSON HILL': 'NEIGHBORHOOD_EMERSON HILL',
                    'FAR ROCKAWAY' : 'NEIGHBORHOOD_FAR ROCKAWAY', 'FASHION': 'NEIGHBORHOOD_FASHION', 'FIELDSTON': 'NEIGHBORHOOD_FIELDSTON', 'FINANCIAL': 'NEIGHBORHOOD_FINANCIAL', 'FLATBUSH-CENTRAL': 'NEIGHBORHOOD_FLATBUSH-CENTRAL', 'FLATBUSH-EAST': 'NEIGHBORHOOD_FLATBUSH-EAST',
                    'FLATBUSH-LEFFERTS GARDEN': 'NEIGHBORHOOD_FLATBUSH-LEFFERTS GARDEN', 'FLATBUSH-NORTH': 'NEIGHBORHOOD_FLATBUSH-NORTH', 'FLATIRON': 'NEIGHBORHOOD_FLATIRON', 'FLATLANDS': 'NEIGHBORHOOD_FLATLANDS', 'FLORAL PARK': 'NEIGHBORHOOD_FLORAL PARK',
                    'FLUSHING-NORTH': 'NEIGHBORHOOD_FLUSHING-NORTH', 'FLUSHING-SOUTH': 'NEIGHBORHOOD_FLUSHING-SOUTH', 'FORDHAM': 'NEIGHBORHOOD_FORDHAM', 'FOREST HILLS': 'NEIGHBORHOOD_FOREST HILLS', 'FORT GREENE': 'NEIGHBORHOOD_FORT GREENE', 'FRESH KILLS': 'NEIGHBORHOOD_FRESH KILLS',
                    'FRESH MEADOWS': 'NEIGHBORHOOD_FRESH MEADOWS', 'GERRITSEN BEACH': 'NEIGHBORHOOD_GERRITSEN BEACH', 'GLEN OAKS': 'NEIGHBORHOOD_GLEN OAKS', 'GLENDALE': 'NEIGHBORHOOD_GLENDALE', 'GOWANUS': 'NEIGHBORHOOD_GOWANUS', 'GRAMERCY': 'NEIGHBORHOOD_GRAMERCY',
                    'GRANT CITY': 'NEIGHBORHOOD_GRANT CITY', 'GRASMERE': 'NEIGHBORHOOD_GRASMERE', 'GRAVESEND': 'NEIGHBORHOOD_GRAVESEND', 'GREAT KILLS': 'NEIGHBORHOOD_GREAT KILLS', 'GREAT KILLS-BAY TERRACE': 'NEIGHBORHOOD_GREAT KILLS-BAY TERRACE', 'GREENPOINT': 'NEIGHBORHOOD_GREENPOINT',
                    'GREENWICH VILLAGE-CENTRAL': 'NEIGHBORHOOD_GREENWICH VILLAGE-CENTRAL', 'GREENWICH VILLAGE-WEST': 'NEIGHBORHOOD_GREENWICH VILLAGE-WEST', 'GRYMES HILL': 'NEIGHBORHOOD_GRYMES HILL', 'HAMMELS': 'NEIGHBORHOOD_HAMMELS', 'HARLEM-CENTRAL': 'NEIGHBORHOOD_HARLEM-CENTRAL',
                    'HARLEM-EAST': 'NEIGHBORHOOD_HARLEM-EAST', 'HARLEM-UPPER': 'NEIGHBORHOOD_HARLEM-UPPER', 'HARLEM-WEST': 'NEIGHBORHOOD_HARLEM-WEST', 'HIGHBRIDGE/MORRIS HEIGHTS': 'NEIGHBORHOOD_HIGHBRIDGE/MORRIS HEIGHTS', 'HILLCREST': 'NEIGHBORHOOD_HILLCREST',
                    'HOLLIS': 'NEIGHBORHOOD_HOLLIS', 'HOLLIS HILLS': 'NEIGHBORHOOD_HOLLIS HILLS', 'HOLLISWOOD': 'NEIGHBORHOOD_HOLLISWOOD', 'HOWARD BEACH': 'NEIGHBORHOOD_HOWARD BEACH', 'HUGUENOT': 'NEIGHBORHOOD_HUGUENOT', 'HUNTS POINT': 'NEIGHBORHOOD_HUNTS POINT',
                    'INWOOD': 'NEIGHBORHOOD_INWOOD', 'JACKSON HEIGHTS': 'NEIGHBORHOOD_JACKSON HEIGHTS', 'JAMAICA': 'NEIGHBORHOOD_JAMAICA', 'JAMAICA BAY': 'NEIGHBORHOOD_JAMAICA BAY', 'JAMAICA ESTATES': 'NEIGHBORHOOD_JAMAICA ESTATES', 'JAMAICA HILLS': 'NEIGHBORHOOD_JAMAICA HILLS',
                    'JAVITS CENTER': 'NEIGHBORHOOD_JAVITS CENTER', 'KENSINGTON': 'NEIGHBORHOOD_KENSINGTON', 'KEW GARDENS': 'NEIGHBORHOOD_KEW GARDENS', 'KINGSBRIDGE HTS/UNIV HTS': 'NEIGHBORHOOD_KINGSBRIDGE HTS/UNIV HTS', 'KINGSBRIDGE/JEROME PARK': 'NEIGHBORHOOD_KINGSBRIDGE/JEROME PARK',
                    'KIPS BAY': 'NEIGHBORHOOD_KIPS BAY', 'LAURELTON': 'NEIGHBORHOOD_LAURELTON', 'LITTLE ITALY': 'NEIGHBORHOOD_LITTLE ITALY', 'LITTLE NECK': 'NEIGHBORHOOD_LITTLE NECK', 'LIVINGSTON': 'NEIGHBORHOOD_LIVINGSTON', 'LONG ISLAND CITY': 'NEIGHBORHOOD_LONG ISLAND CITY',
                    'LOWER EAST SIDE': 'NEIGHBORHOOD_LOWER EAST SIDE', 'MADISON': 'NEIGHBORHOOD_MADISON', 'MANHATTAN BEACH': 'NEIGHBORHOOD_MANHATTAN BEACH', 'MANHATTAN VALLEY': 'NEIGHBORHOOD_MANHATTAN VALLEY', 'MANOR HEIGHTS': 'NEIGHBORHOOD_MANOR HEIGHTS', 'MARINE PARK': 'NEIGHBORHOOD_MARINE PARK',
                    'MARINERS HARBOR': 'NEIGHBORHOOD_MARINERS HARBOR', 'MASPETH': 'NEIGHBORHOOD_MASPETH', 'MELROSE/CONCOURSE': 'NEIGHBORHOOD_MELROSE/CONCOURSE', 'MIDDLE VILLAGE': 'NEIGHBORHOOD_MIDDLE VILLAGE', 'MIDLAND BEACH': 'NEIGHBORHOOD_MIDLAND BEACH', 'MIDTOWN CBD': 'NEIGHBORHOOD_MIDTOWN CBD',
                    'MIDTOWN EAST': 'NEIGHBORHOOD_MIDTOWN EAST', 'MIDTOWN WEST': 'NEIGHBORHOOD_MIDTOWN WEST', 'MIDWOOD': 'NEIGHBORHOOD_MIDWOOD', 'MILL BASIN': 'NEIGHBORHOOD_MILL BASIN', 'MORRIS PARK/VAN NEST': 'NEIGHBORHOOD_MORRIS PARK/VAN NEST', 'MORRISANIA/LONGWOOD': 'NEIGHBORHOOD_MORRISANIA/LONGWOOD',
                    'MOTT HAVEN/PORT MORRIS': 'NEIGHBORHOOD_MOTT HAVEN/PORT MORRIS', 'MOUNT HOPE/MOUNT EDEN': 'NEIGHBORHOOD_MOUNT HOPE/MOUNT EDEN', 'MURRAY HILL': 'NEIGHBORHOOD_MURRAY HILL', 'NAVY YARD': 'NEIGHBORHOOD_NAVY YARD', 'NEPONSIT': 'NEIGHBORHOOD_NEPONSIT', 'NEW BRIGHTON': 'NEIGHBORHOOD_NEW BRIGHTON',
                    'NEW BRIGHTON-ST. GEORGE': 'NEIGHBORHOOD_NEW BRIGHTON-ST. GEORGE', 'NEW DORP': 'NEIGHBORHOOD_NEW DORP', 'NEW DORP-BEACH': 'NEIGHBORHOOD_NEW DORP-BEACH', 'NEW DORP-HEIGHTS': 'NEIGHBORHOOD_NEW DORP-HEIGHTS', 'NEW SPRINGVILLE': 'NEIGHBORHOOD_NEW SPRINGVILLE', 'OAKLAND GARDENS': 'NEIGHBORHOOD_OAKLAND GARDENS',
                    'OAKWOOD': 'NEIGHBORHOOD_OAKWOOD', 'OAKWOOD-BEACH': 'NEIGHBORHOOD_OAKWOOD-BEACH', 'OCEAN HILL': 'NEIGHBORHOOD_OCEAN HILL', 'OCEAN PARKWAY-NORTH': 'NEIGHBORHOOD_OCEAN PARKWAY-NORTH', 'OCEAN PARKWAY-SOUTH': 'NEIGHBORHOOD_OCEAN PARKWAY-SOUTH', 'OLD MILL BASIN': 'NEIGHBORHOOD_OLD MILL BASIN', 'OZONE PARK': 'NEIGHBORHOOD_OZONE PARK',
                    'PARK SLOPE': 'NEIGHBORHOOD_PARK SLOPE', 'PARK SLOPE SOUTH': 'NEIGHBORHOOD_PARK SLOPE SOUTH', 'PARKCHESTER': 'NEIGHBORHOOD_PARKCHESTER', 'PELHAM GARDENS': 'NEIGHBORHOOD_PELHAM GARDENS', 'PELHAM PARKWAY NORTH': 'NEIGHBORHOOD_PELHAM PARKWAY NORTH', 'PELHAM PARKWAY SOUTH': 'NEIGHBORHOOD_PELHAM PARKWAY SOUTH',
                    'PLEASANT PLAINS': 'NEIGHBORHOOD_PLEASANT PLAINS', 'PORT IVORY': 'NEIGHBORHOOD_PORT IVORY', 'PORT RICHMOND': 'NEIGHBORHOOD_PORT RICHMOND', 'PRINCES BAY': 'NEIGHBORHOOD_PRINCES BAY', 'PROSPECT HEIGHTS': 'NEIGHBORHOOD_PROSPECT HEIGHTS', 'QUEENS VILLAGE': 'NEIGHBORHOOD_QUEENS VILLAGE',
                    'RED HOOK': 'NEIGHBORHOOD_RED HOOK', 'REGO PARK': 'NEIGHBORHOOD_REGO PARK', 'RICHMOND HILL': 'NEIGHBORHOOD_RICHMOND HILL', 'RICHMONDTOWN': 'NEIGHBORHOOD_RICHMONDTOWN', 'RICHMONDTOWN-LIGHTHS HILL': 'NEIGHBORHOOD_RICHMONDTOWN-LIGHTHS HILL', 'RIDGEWOOD': 'NEIGHBORHOOD_RIDGEWOOD', 'RIVERDALE': 'NEIGHBORHOOD_RIVERDALE',
                    'ROCKAWAY PARK': 'NEIGHBORHOOD_ROCKAWAY PARK', 'ROOSEVELT ISLAND': 'NEIGHBORHOOD_ROOSEVELT ISLAND', 'ROSEBANK': 'NEIGHBORHOOD_ROSEBANK', 'ROSEDALE': 'NEIGHBORHOOD_ROSEDALE', 'ROSSVILLE': 'NEIGHBORHOOD_ROSSVILLE', 'ROSSVILLE-CHARLESTON': 'NEIGHBORHOOD_ROSSVILLE-CHARLESTON', 'ROSSVILLE-PORT MOBIL': 'NEIGHBORHOOD_ROSSVILLE-PORT MOBIL',
                    'ROSSVILLE-RICHMOND VALLEY': 'NEIGHBORHOOD_ROSSVILLE-RICHMOND VALLEY', 'SCHUYLERVILLE/PELHAM BAY': 'NEIGHBORHOOD_SCHUYLERVILLE/PELHAM BAY', 'SEAGATE': 'NEIGHBORHOOD_SEAGATE', 'SHEEPSHEAD BAY': 'NEIGHBORHOOD_SHEEPSHEAD BAY', 'SILVER LAKE': 'NEIGHBORHOOD_SILVER LAKE', 'SO. JAMAICA-BAISLEY PARK': 'NEIGHBORHOOD_SO. JAMAICA-BAISLEY PARK',
                    'SOHO': 'NEIGHBORHOOD_SOHO', 'SOUNDVIEW': 'NEIGHBORHOOD_SOUNDVIEW', 'SOUTH BEACH': 'NEIGHBORHOOD_SOUTH BEACH', 'SOUTH JAMAICA': 'NEIGHBORHOOD_SOUTH JAMAICA', 'SOUTH OZONE PARK': 'NEIGHBORHOOD_SOUTH OZONE PARK', 'SOUTHBRIDGE': 'NEIGHBORHOOD_SOUTHBRIDGE', 'SPRING CREEK': 'NEIGHBORHOOD_SPRING CREEK',
                    'SPRINGFIELD GARDENS': 'NEIGHBORHOOD_SPRINGFIELD GARDENS', 'ST. ALBANS': 'NEIGHBORHOOD_ST. ALBANS', 'STAPLETON': 'NEIGHBORHOOD_STAPLETON', 'STAPLETON-CLIFTON': 'NEIGHBORHOOD_STAPLETON-CLIFTON', 'SUNNYSIDE': 'NEIGHBORHOOD_SUNNYSIDE', 'SUNSET PARK': 'NEIGHBORHOOD_SUNSET PARK', 'THROGS NECK': 'NEIGHBORHOOD_THROGS NECK',
                    'TODT HILL': 'NEIGHBORHOOD_TODT HILL', 'TOMPKINSVILLE': 'NEIGHBORHOOD_TOMPKINSVILLE', 'TOTTENVILLE': 'NEIGHBORHOOD_TOTTENVILLE', 'TRAVIS': 'NEIGHBORHOOD_TRAVIS', 'TRIBECA': 'NEIGHBORHOOD_TRIBECA', 'UPPER EAST SIDE (59-79)': 'NEIGHBORHOOD_UPPER EAST SIDE (59-79)', 'UPPER EAST SIDE (79-96)': 'NEIGHBORHOOD_UPPER EAST SIDE (79-96)',
                    'UPPER EAST SIDE (96-110)': 'NEIGHBORHOOD_UPPER EAST SIDE (96-110)', 'UPPER WEST SIDE (59-79)': 'NEIGHBORHOOD_UPPER WEST SIDE (59-79)', 'UPPER WEST SIDE (79-96)': 'NEIGHBORHOOD_UPPER WEST SIDE (79-96)', 'UPPER WEST SIDE (96-116)': 'NEIGHBORHOOD_UPPER WEST SIDE (96-116)', 'WAKEFIELD': 'NEIGHBORHOOD_WAKEFIELD',
                    'WASHINGTON HEIGHTS LOWER': 'NEIGHBORHOOD_WASHINGTON HEIGHTS LOWER', 'WASHINGTON HEIGHTS UPPER': 'NEIGHBORHOOD_WASHINGTON HEIGHTS UPPER', 'WEST NEW BRIGHTON': 'NEIGHBORHOOD_WEST NEW BRIGHTON', 'WESTCHESTER': 'NEIGHBORHOOD_WESTCHESTER', 'WESTERLEIGH': 'NEIGHBORHOOD_WESTERLEIGH', 'WHITESTONE': 'NEIGHBORHOOD_WHITESTONE',
                    'WILLIAMSBRIDGE': 'NEIGHBORHOOD_WILLIAMSBRIDGE', 'WILLIAMSBURG-CENTRAL': 'NEIGHBORHOOD_WILLIAMSBURG-CENTRAL', 'WILLIAMSBURG-EAST': 'NEIGHBORHOOD_WILLIAMSBURG-EAST', 'WILLIAMSBURG-NORTH': 'NEIGHBORHOOD_WILLIAMSBURG-NORTH', 'WILLIAMSBURG-SOUTH': 'NEIGHBORHOOD_WILLIAMSBURG-SOUTH', 'WILLOWBROOK': 'NEIGHBORHOOD_WILLOWBROOK',
                    'WINDSOR TERRACE': 'NEIGHBORHOOD_WINDSOR TERRACE', 'WOODHAVEN': 'NEIGHBORHOOD_WOODHAVEN', 'WOODLAWN': 'NEIGHBORHOOD_WOODLAWN', 'WOODROW': 'NEIGHBORHOOD_WOODROW', 'WOODSIDE': 'NEIGHBORHOOD_WOODSIDE', 'WYCKOFF HEIGHTS': 'NEIGHBORHOOD_WYCKOFF HEIGHTS'}

        return data_dic.get(val)

    # Defining function to assign value to exact dummy value for user input of 'tax_class'
    def tax_class_dummy(val):

        data_dic = {'1': 'TAX CLASS AT PRESENT_1', '2': 'TAX CLASS AT PRESENT_2', '4':'TAX CLASS AT PRESENT_4'}

        return data_dic.get(val)

    # Defining function to assign value to exact dummy value for user input of 'building_class'
    def building_class_dummy(val):
        data_dic = {'A0': 'BUILDING CLASS AT PRESENT_A0', 'A1': 'BUILDING CLASS AT PRESENT_A1',
                    'A2': 'BUILDING CLASS AT PRESENT_A2', 'A3': 'BUILDING CLASS AT PRESENT_A3',
                    'A4': 'BUILDING CLASS AT PRESENT_A4', 'A5': 'BUILDING CLASS AT PRESENT_A5',
                    'A6': 'BUILDING CLASS AT PRESENT_A6', 'A7': 'BUILDING CLASS AT PRESENT_A7',
                    'A9': 'BUILDING CLASS AT PRESENT_A9', 'B1': 'BUILDING CLASS AT PRESENT_B1',
                    'B2': 'BUILDING CLASS AT PRESENT_B2', 'B3': 'BUILDING CLASS AT PRESENT_B3',
                    'B9': 'BUILDING CLASS AT PRESENT_B9', 'C0': 'BUILDING CLASS AT PRESENT_C0',
                    'C1': 'BUILDING CLASS AT PRESENT_C1', 'C2': 'BUILDING CLASS AT PRESENT_C2',
                    'C3': 'BUILDING CLASS AT PRESENT_C3', 'C4': 'BUILDING CLASS AT PRESENT_C4',
                    'C5': 'BUILDING CLASS AT PRESENT_C5', 'C6': 'BUILDING CLASS AT PRESENT_C6',
                    'C7': 'BUILDING CLASS AT PRESENT_C7', 'C9': 'BUILDING CLASS AT PRESENT_C9',
                    'CM': 'BUILDING CLASS AT PRESENT_CM', 'D1': 'BUILDING CLASS AT PRESENT_D1',
                    'D2': 'BUILDING CLASS AT PRESENT_D2', 'D3': 'BUILDING CLASS AT PRESENT_D3',
                    'D4': 'BUILDING CLASS AT PRESENT_D4', 'D5': 'BUILDING CLASS AT PRESENT_D5',
                    'D6': 'BUILDING CLASS AT PRESENT_D6', 'D7': 'BUILDING CLASS AT PRESENT_D7',
                    'D9': 'BUILDING CLASS AT PRESENT_D9', 'E1': 'BUILDING CLASS AT PRESENT_E1',
                    'E2': 'BUILDING CLASS AT PRESENT_E2', 'E7': 'BUILDING CLASS AT PRESENT_E7',
                    'E9': 'BUILDING CLASS AT PRESENT_E9', 'F1': 'BUILDING CLASS AT PRESENT_F1',
                    'F2': 'BUILDING CLASS AT PRESENT_F2', 'F4': 'BUILDING CLASS AT PRESENT_F4',
                    'F5': 'BUILDING CLASS AT PRESENT_F5', 'F9': 'BUILDING CLASS AT PRESENT_F9',
                    'G0': 'BUILDING CLASS AT PRESENT_G0', 'G1': 'BUILDING CLASS AT PRESENT_G1',
                    'G2': 'BUILDING CLASS AT PRESENT_G2', 'G4': 'BUILDING CLASS AT PRESENT_G4',
                    'G5': 'BUILDING CLASS AT PRESENT_G5', 'G7': 'BUILDING CLASS AT PRESENT_G7',
                    'G8': 'BUILDING CLASS AT PRESENT_G8', 'G9': 'BUILDING CLASS AT PRESENT_G9',
                    'GU': 'BUILDING CLASS AT PRESENT_GU', 'GW': 'BUILDING CLASS AT PRESENT_GW',
                    'H1': 'BUILDING CLASS AT PRESENT_H1', 'H2': 'BUILDING CLASS AT PRESENT_H2',
                    'H3': 'BUILDING CLASS AT PRESENT_H3', 'H4': 'BUILDING CLASS AT PRESENT_H4',
                    'H6': 'BUILDING CLASS AT PRESENT_H6', 'H8': 'BUILDING CLASS AT PRESENT_H8',
                    'H9': 'BUILDING CLASS AT PRESENT_H9', 'HB': 'BUILDING CLASS AT PRESENT_HB',
                    'HH': 'BUILDING CLASS AT PRESENT_HH', 'HR': 'BUILDING CLASS AT PRESENT_HR',
                    'HS': 'BUILDING CLASS AT PRESENT_HS', 'I1': 'BUILDING CLASS AT PRESENT_I1',
                    'I3': 'BUILDING CLASS AT PRESENT_I3', 'I4': 'BUILDING CLASS AT PRESENT_I4',
                    'I5': 'BUILDING CLASS AT PRESENT_I5', 'I6': 'BUILDING CLASS AT PRESENT_I6',
                    'I7': 'BUILDING CLASS AT PRESENT_I7', 'I9': 'BUILDING CLASS AT PRESENT_I9',
                    'J1': 'BUILDING CLASS AT PRESENT_J1', 'J4': 'BUILDING CLASS AT PRESENT_J4',
                    'J8': 'BUILDING CLASS AT PRESENT_J8', 'K1': 'BUILDING CLASS AT PRESENT_K1',
                    'K2': 'BUILDING CLASS AT PRESENT_K2', 'K3': 'BUILDING CLASS AT PRESENT_K3',
                    'K4': 'BUILDING CLASS AT PRESENT_K4', 'K5': 'BUILDING CLASS AT PRESENT_K5',
                    'K6': 'BUILDING CLASS AT PRESENT_K6', 'K7': 'BUILDING CLASS AT PRESENT_K7',
                    'K8': 'BUILDING CLASS AT PRESENT_K8', 'K9': 'BUILDING CLASS AT PRESENT_K9',
                    'L1': 'BUILDING CLASS AT PRESENT_L1', 'L3': 'BUILDING CLASS AT PRESENT_L3',
                    'L8': 'BUILDING CLASS AT PRESENT_L8', 'L9': 'BUILDING CLASS AT PRESENT_L9',
                    'M1': 'BUILDING CLASS AT PRESENT_M1', 'M2': 'BUILDING CLASS AT PRESENT_M2',
                    'M3': 'BUILDING CLASS AT PRESENT_M3', 'M4': 'BUILDING CLASS AT PRESENT_M4',
                    'M9': 'BUILDING CLASS AT PRESENT_M9', 'N2': 'BUILDING CLASS AT PRESENT_N2',
                    'N9': 'BUILDING CLASS AT PRESENT_N9', 'O1': 'BUILDING CLASS AT PRESENT_O1',
                    'O2': 'BUILDING CLASS AT PRESENT_O2', 'O3': 'BUILDING CLASS AT PRESENT_O3',
                    'O4': 'BUILDING CLASS AT PRESENT_O4', 'O5': 'BUILDING CLASS AT PRESENT_O5',
                    'O6': 'BUILDING CLASS AT PRESENT_O6', 'O7': 'BUILDING CLASS AT PRESENT_O7',
                    'O8': 'BUILDING CLASS AT PRESENT_O8', 'O9': 'BUILDING CLASS AT PRESENT_O9',
                    'P2': 'BUILDING CLASS AT PRESENT_P2', 'P5': 'BUILDING CLASS AT PRESENT_P5',
                    'P6': 'BUILDING CLASS AT PRESENT_P6', 'P8': 'BUILDING CLASS AT PRESENT_P8',
                    'P9': 'BUILDING CLASS AT PRESENT_P9', 'Q8': 'BUILDING CLASS AT PRESENT_Q8',
                    'R1': 'BUILDING CLASS AT PRESENT_R1', 'R2': 'BUILDING CLASS AT PRESENT_R2',
                    'R3': 'BUILDING CLASS AT PRESENT_R3', 'R4': 'BUILDING CLASS AT PRESENT_R4',
                    'R5': 'BUILDING CLASS AT PRESENT_R5', 'R6': 'BUILDING CLASS AT PRESENT_R6',
                    'R8': 'BUILDING CLASS AT PRESENT_R8', 'RR': 'BUILDING CLASS AT PRESENT_RR',
                    'S0': 'BUILDING CLASS AT PRESENT_S0', 'S1': 'BUILDING CLASS AT PRESENT_S1',
                    'S2': 'BUILDING CLASS AT PRESENT_S2', 'S3': 'BUILDING CLASS AT PRESENT_S3',
                    'S4': 'BUILDING CLASS AT PRESENT_S4', 'S5': 'BUILDING CLASS AT PRESENT_S5',
                    'S9': 'BUILDING CLASS AT PRESENT_S9', 'V0': 'BUILDING CLASS AT PRESENT_V0',
                    'V1': 'BUILDING CLASS AT PRESENT_V1', 'W1': 'BUILDING CLASS AT PRESENT_W1',
                    'W2': 'BUILDING CLASS AT PRESENT_W2', 'W3': 'BUILDING CLASS AT PRESENT_W3',
                    'W4': 'BUILDING CLASS AT PRESENT_W4', 'W8': 'BUILDING CLASS AT PRESENT_W8',
                    'W9': 'BUILDING CLASS AT PRESENT_W9', 'Y3': 'BUILDING CLASS AT PRESENT_Y3',
                    'Z0': 'BUILDING CLASS AT PRESENT_Z0', 'Z9': 'BUILDING CLASS AT PRESENT_Z9'}

        return data_dic.get(val)

    # Defining function to assign value to exact dummy value for user input of 'zip_code'
    def zip_code_dummy(val):
        data_dic = {'10001': 'ZIP CODE_10001', '10002': 'ZIP CODE_10002', '10003': 'ZIP CODE_10003',
                    '10004': 'ZIP CODE_10004', '10005': 'ZIP CODE_10005', '10006': 'ZIP CODE_10006',
                    '10007': 'ZIP CODE_10007', '10009': 'ZIP CODE_10009', '10010': 'ZIP CODE_10010',
                    '10011': 'ZIP CODE_10011', '10012': 'ZIP CODE_10012', '10013': 'ZIP CODE_10013',
                    '10014': 'ZIP CODE_10014', '10016': 'ZIP CODE_10016', '10017': 'ZIP CODE_10017',
                    '10018': 'ZIP CODE_10018', '10019': 'ZIP CODE_10019', '10021': 'ZIP CODE_10021',
                    '10022': 'ZIP CODE_10022', '10023': 'ZIP CODE_10023', '10024': 'ZIP CODE_10024',
                    '10025': 'ZIP CODE_10025', '10026': 'ZIP CODE_10026', '10027': 'ZIP CODE_10027',
                    '10028': 'ZIP CODE_10028', '10029': 'ZIP CODE_10029', '10030': 'ZIP CODE_10030',
                    '10031': 'ZIP CODE_10031', '10032': 'ZIP CODE_10032', '10033': 'ZIP CODE_10033',
                    '10034': 'ZIP CODE_10034', '10035': 'ZIP CODE_10035', '10036': 'ZIP CODE_10036',
                    '10037': 'ZIP CODE_10037', '10038': 'ZIP CODE_10038', '10039': 'ZIP CODE_10039',
                    '10040': 'ZIP CODE_10040', '10044': 'ZIP CODE_10044', '10065': 'ZIP CODE_10065',
                    '10069': 'ZIP CODE_10069', '10075': 'ZIP CODE_10075', '10128': 'ZIP CODE_10128',
                    '10167': 'ZIP CODE_10167', '10280': 'ZIP CODE_10280', '10282': 'ZIP CODE_10282',
                    '10301': 'ZIP CODE_10301', '10302': 'ZIP CODE_10302', '10303': 'ZIP CODE_10303',
                    '10304': 'ZIP CODE_10304', '10305': 'ZIP CODE_10305', '10306': 'ZIP CODE_10306',
                    '10307': 'ZIP CODE_10307', '10308': 'ZIP CODE_10308', '10309': 'ZIP CODE_10309',
                    '10310': 'ZIP CODE_10310', '10312': 'ZIP CODE_10312', '10314': 'ZIP CODE_10314',
                    '10451': 'ZIP CODE_10451', '10452': 'ZIP CODE_10452', '10453': 'ZIP CODE_10453',
                    '10454': 'ZIP CODE_10454', '10455': 'ZIP CODE_10455', '10456': 'ZIP CODE_10456',
                    '10457': 'ZIP CODE_10457', '10458': 'ZIP CODE_10458', '10459': 'ZIP CODE_10459',
                    '10460': 'ZIP CODE_10460', '10461': 'ZIP CODE_10461', '10462': 'ZIP CODE_10462',
                    '10463': 'ZIP CODE_10463', '10464': 'ZIP CODE_10464', '10465': 'ZIP CODE_10465',
                    '10466': 'ZIP CODE_10466', '10467': 'ZIP CODE_10467', '10468': 'ZIP CODE_10468',
                    '10469': 'ZIP CODE_10469', '10470': 'ZIP CODE_10470', '10471': 'ZIP CODE_10471',
                    '10472': 'ZIP CODE_10472', '10473': 'ZIP CODE_10473', '10474': 'ZIP CODE_10474',
                    '10475': 'ZIP CODE_10475', '10803': 'ZIP CODE_10803', '11001': 'ZIP CODE_11001',
                    '11004': 'ZIP CODE_11004', '11005': 'ZIP CODE_11005', '11040': 'ZIP CODE_11040',
                    '11101': 'ZIP CODE_11101', '11102': 'ZIP CODE_11102', '11103': 'ZIP CODE_11103',
                    '11104': 'ZIP CODE_11104', '11105': 'ZIP CODE_11105', '11106': 'ZIP CODE_11106',
                    '11109': 'ZIP CODE_11109', '11201': 'ZIP CODE_11201', '11203': 'ZIP CODE_11203',
                    '11204': 'ZIP CODE_11204', '11205': 'ZIP CODE_11205', '11206': 'ZIP CODE_11206',
                    '11207': 'ZIP CODE_11207', '11208': 'ZIP CODE_11208', '10209': 'ZIP CODE_11209',
                    '11210': 'ZIP CODE_11210', '11211': 'ZIP CODE_11211', '11212': 'ZIP CODE_11212',
                    '11213': 'ZIP CODE_11213', '11214': 'ZIP CODE_11214', '11215': 'ZIP CODE_11215',
                    '11216': 'ZIP CODE_11216', '11217': 'ZIP CODE_11217', '11218': 'ZIP CODE_11218',
                    '11219': 'ZIP CODE_11219', '11220': 'ZIP CODE_11220', '11221': 'ZIP CODE_11221',
                    '11222': 'ZIP CODE_11222', '11223': 'ZIP CODE_11223', '11224': 'ZIP CODE_11224',
                    '11225': 'ZIP CODE_11225', '11226': 'ZIP CODE_11226', '11228': 'ZIP CODE_11228',
                    '11229': 'ZIP CODE_11229', '11230': 'ZIP CODE_11230', '11231': 'ZIP CODE_11231',
                    '11232': 'ZIP CODE_11232', '11233': 'ZIP CODE_11233', '11234': 'ZIP CODE_11234',
                    '11235': 'ZIP CODE_11235', '11236': 'ZIP CODE_11236', '11237': 'ZIP CODE_11237',
                    '11238': 'ZIP CODE_11238', '11239': 'ZIP CODE_11239', '11249': 'ZIP CODE_11249',
                    '11354': 'ZIP CODE_11354', '11355': 'ZIP CODE_11355', '11356': 'ZIP CODE_11356',
                    '11357': 'ZIP CODE_11357', '11358': 'ZIP CODE_11358', '11360': 'ZIP CODE_11360',
                    '11361': 'ZIP CODE_11361', '11362': 'ZIP CODE_11362', '11363': 'ZIP CODE_11363',
                    '11364': 'ZIP CODE_11364', '11365': 'ZIP CODE_11365', '11366': 'ZIP CODE_11366',
                    '11367': 'ZIP CODE_11367', '11368': 'ZIP CODE_11368', '11369': 'ZIP CODE_11369',
                    '11370': 'ZIP CODE_11370', '11372': 'ZIP CODE_11372', '11373': 'ZIP CODE_11373',
                    '11374': 'ZIP CODE_11374', '11375': 'ZIP CODE_11375', '11377': 'ZIP CODE_11377',
                    '11378': 'ZIP CODE_11378', '11379': 'ZIP CODE_11379', '11385': 'ZIP CODE_11385',
                    '11411': 'ZIP CODE_11411', '11412': 'ZIP CODE_11412', '11413': 'ZIP CODE_11413',
                    '11414': 'ZIP CODE_11414', '11415': 'ZIP CODE_11415', '11416': 'ZIP CODE_11416',
                    '11417': 'ZIP CODE_11417', '11418': 'ZIP CODE_11418', '11419': 'ZIP CODE_11419',
                    '11420': 'ZIP CODE_11420', '11421': 'ZIP CODE_11421', '11422': 'ZIP CODE_11422',
                    '11423': 'ZIP CODE_11423', '11426': 'ZIP CODE_11426', '11427': 'ZIP CODE_11427',
                    '11428': 'ZIP CODE_11428', '11429': 'ZIP CODE_11429', '11432': 'ZIP CODE_11432',
                    '11433': 'ZIP CODE_11433', '11434': 'ZIP CODE_11434', '11435': 'ZIP CODE_11435',
                    '11436': 'ZIP CODE_11436', '11691': 'ZIP CODE_11691', '11462': 'ZIP CODE_11692',
                    '11693': 'ZIP CODE_11693', '11694': 'ZIP CODE_11694'}

        return data_dic.get(val)

    # Collect user input values to one user_input_dic to assign them to user_input dataframe
    user_input_dic = {}
    col_borough = borough_dummy(user_data.iloc[0]['borough'])
    col_neighbor = neighbour_dummy(user_data.iloc[0]['neighbour'])
    col_tax_class = tax_class_dummy(user_data.iloc[0]['tax_class'])
    col_building_class = building_class_dummy(user_data.iloc[0]['building_class'])
    col_zip_code = zip_code_dummy(user_data.iloc[0]['zip_code'])

    user_input_dic[col_borough] = 1
    user_input_dic[col_neighbor] = 1
    user_input_dic[col_tax_class] = 1
    user_input_dic[col_building_class] = 1
    user_input_dic[col_zip_code] = 1
    user_input_dic['RESIDENTIAL UNITS'] = user_data.iloc[0]['residential units']
    user_input_dic['COMMERCIAL UNITS'] = user_data.iloc[0]['commercial units']
    user_input_dic['LAND SQUARE FEET'] = user_data.iloc[0]['Land sqft']
    user_input_dic['GROSS SQUARE FEET'] = user_data.iloc[0]['Gross sqft']

    # Append user_input_dic dictionary to user_input dataframe
    user_input = user_input[0:0]
    user_input = user_input.append(user_input_dic, ignore_index=True)

    # Fill the blank spaces of dataframe with '0'
    user_input = user_input.fillna(0)

    # Converting dataframe to a json file
    #user_input = user_input.to_json(orient="records")
    #user_input = json.loads(user_input)
    #user_input = json.dumps(user_input)

    # Return the resulting json file
    return user_input