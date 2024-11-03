import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Load the trained model and scaler
model = load_model('your_model.h5', custom_objects={'rmse': rmse}) # Replace 'your_model.h5'
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

cleanhouse = pd.read_csv('cleanhouse.csv') # Replace 'cleanhouse.csv'

# Define RMSE function (if not already defined)
def rmse(y_true, y_pred):
    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def predict_price(input_features, cleanhouse_df, model, scaler):
    """Predicts the price based on user inputs."""

    input_df = pd.DataFrame([input_features])

    # Combine user input with cleanhouse data before one-hot encoding
    combined_df = pd.concat([cleanhouse_df, input_df], ignore_index=True)

    categorical_cols = ['Method', 'SellerG', 'CouncilArea', 'Regionname', 'Season', 'Suburb', 'Type']  # Add 'Type' here
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #sparse=False for newer scikit-learn versions

    encoded_categoricals = encoder.fit_transform(combined_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out(categorical_cols))

    numerical_cols = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'Bathroom', 'Car', 'YearBuilt',
                      'Postcode', 'Latitude', 'Longtitude', 'Year', 'Month','Age'] # Add 'Age'

    combined_df = combined_df[numerical_cols].reset_index(drop=True)
    final_df = pd.concat([combined_df, encoded_df], axis=1)


    input_data = final_df.iloc[[-1]].values # Get the last row (user input)

    # Scale numerical features using the trained scaler
    input_data_scaled = scaler.transform(input_data)

    predicted_price = model.predict(input_data_scaled)[0][0]

    return predicted_price

# Streamlit app
st.title("Melbourne Housing Price Prediction")
feature_inputs = {}

feature_inputs['Rooms'] = st.number_input("จำนวนห้อง", min_value=1, max_value=10, value=3)
feature_inputs['Distance'] = st.number_input("ระยะห่างจาก CBD (กม.)", min_value=0.0, value=5.0)
feature_inputs['Landsize'] = st.number_input("ขนาดที่ดิน (ตร.ม.)", min_value=0.0, value=500.0)
feature_inputs['BuildingArea'] = st.number_input("ขนาดพื้นที่สิ่งปลูกสร้าง (ตร.ม.)", min_value=0.0, value=150.0)
feature_inputs['Bathroom'] = st.number_input("จำนวนห้องน้ำ", min_value=0, max_value=10, value=2)
feature_inputs['Car'] = st.number_input("จำนวนที่จอดรถ", min_value=0, max_value=10, value=1)
feature_inputs['YearBuilt'] = st.number_input("ปีที่สร้าง", min_value=1900, max_value=2024, value=2000)
feature_inputs['Postcode'] = st.number_input("รหัสไปรษณีย์", min_value=1000, max_value=9999, value=3000)
feature_inputs['CouncilArea'] = st.selectbox("เขตปกครอง", options=[
  'Yarra City Council', 'Moonee Valley City Council',
  'Port Phillip City Council', 'Darebin City Council',
  'Hobsons Bay City Council', 'Stonnington City Council',
  'Boroondara City Council', 'Monash City Council',
  'Glen Eira City Council', 'Whitehorse City Council',
  'Maribyrnong City Council', 'Bayside City Council',
  'Moreland City Council', 'Manningham City Council',
  'Melbourne City Council', 'Banyule City Council',
  'Brimbank City Council', 'Kingston City Council',
  'Hume City Council', 'Knox City Council', 'Maroondah City Council',
  'Casey City Council', 'Melton City Council',
  'Greater Dandenong City Council', 'Nillumbik Shire Council',
  'Cardinia Shire Council', 'Whittlesea City Council',
  'Frankston City Council', 'Macedon Ranges Shire Council',
  'Yarra Ranges Shire Council', 'Wyndham City Council',
  'Moorabool Shire Council', 'Mitchell Shire Council'
  ])
feature_inputs['Latitude'] = st.number_input("ละติจูด", value=17.0)
feature_inputs['Longtitude'] = st.number_input("ลองติจูด", value=150.0)
feature_inputs['Regionname'] = st.selectbox("ชื่อภูมิภาค", options=[
  'Northern Metropolitan', 'Western Metropolitan',
  'Southern Metropolitan', 'Eastern Metropolitan',
  'South-Eastern Metropolitan', 'Eastern Victoria',
  'Northern Victoria', 'Western Victoria'
  ])
feature_inputs['Season'] = st.selectbox("ฤดูกาล", options=["Spring", "Summer", "Autumn", "Winter"])
feature_inputs['Year'] = st.number_input("ปี", min_value=2000, max_value=2024, value=2024)
feature_inputs['Month'] = st.number_input("เดือน", min_value=1, max_value=12, value=1)
feature_inputs['Type'] = st.selectbox("ประเภทอสังหาริมทรัพย์", options=["House", "Unit", "Townhouse", "Apartment", "Others"])
feature_inputs['Method'] = st.selectbox("วิธีขาย", options=['SS', 'S', 'SP', 'PI', 'VB', 'SN', 'W', 'PN', 'SA'])
feature_inputs['Suburb'] = st.selectbox("ชื่อชุมชน", options=['Abbotsford', 'Airport West', 'Albert Park', 'Alphington',
  'Altona', 'Altona North', 'Armadale', 'Ascot Vale', 'Ashburton',
  'Ashwood', 'Avondale Heights', 'Balaclava', 'Balwyn',
  'Balwyn North', 'Bentleigh', 'Bentleigh East', 'Box Hill',
  'Braybrook', 'Brighton', 'Brighton East', 'Brunswick',
  'Brunswick West', 'Bulleen', 'Burwood', 'Camberwell', 'Canterbury',
  'Carlton North', 'Carnegie', 'Caulfield', 'Caulfield North',
  'Caulfield South', 'Chadstone', 'Clifton Hill', 'Coburg',
  'Coburg North', 'Collingwood', 'Doncaster', 'Eaglemont',
  'Elsternwick', 'Elwood', 'Essendon', 'Essendon North', 'Fairfield',
  'Fitzroy', 'Fitzroy North', 'Flemington', 'Footscray', 'Glen Iris',
  'Glenroy', 'Gowanbrae', 'Hadfield', 'Hampton', 'Hampton East',
  'Hawthorn', 'Heidelberg Heights', 'Heidelberg West', 'Hughesdale',
  'Ivanhoe', 'Kealba', 'Keilor East', 'Kensington', 'Kew',
  'Kew East', 'Kooyong', 'Maidstone', 'Malvern', 'Malvern East',
  'Maribyrnong', 'Melbourne', 'Middle Park', 'Mont Albert',
  'Moonee Ponds', 'Moorabbin', 'Newport', 'Niddrie',
  'North Melbourne', 'Northcote', 'Oak Park', 'Oakleigh South',
  'Parkville', 'Pascoe Vale', 'Port Melbourne', 'Prahran', 'Preston',
  'Reservoir', 'Richmond', 'Rosanna', 'Seddon', 'South Melbourne',
  'South Yarra', 'Southbank', 'Spotswood', 'St Kilda', 'Strathmore',
  'Sunshine', 'Sunshine North', 'Sunshine West', 'Surrey Hills',
  'Templestowe Lower', 'Thornbury', 'Toorak', 'Viewbank', 'Watsonia',
  'West Melbourne', 'Williamstown', 'Williamstown North', 'Windsor',
  'Yallambie', 'Yarraville', 'Aberfeldie', 'Bellfield',
  'Brunswick East', 'Burnley', 'Campbellfield', 'Carlton',
  'East Melbourne', 'Essendon West', 'Fawkner', 'Hawthorn East',
  'Heidelberg', 'Ivanhoe East', 'Jacana', 'Kingsbury', 'Kingsville',
  'Murrumbeena', 'Ormond', 'West Footscray', 'Albion', 'Brooklyn',
  'Glen Huntly', 'Oakleigh', 'Ripponlea', 'Cremorne', 'Docklands',
  'South Kingsville', 'Strathmore Heights', 'Travancore',
  'Caulfield East', 'Seaholme', 'Keilor Park', 'Gardenvale',
  'Princes Hill', 'Altona Meadows', 'Bayswater', 'Bayswater North',
  'Beaumaris', 'Berwick', 'Blackburn', 'Boronia', 'Briar Hill',
  'Broadmeadows', 'Bundoora', 'Burnside Heights', 'Burwood East',
  'Cairnlea', 'Caroline Springs', 'Cheltenham', 'Clarinda',
  'Clayton', 'Craigieburn', 'Cranbourne', 'Croydon', 'Croydon Hills',
  'Dandenong', 'Dandenong North', 'Diamond Creek', 'Dingley Village',
  'Doncaster East', 'Donvale', 'Doreen', 'Eltham', 'Eltham North',
  'Emerald', 'Epping', 'Forest Hill', 'Frankston', 'Frankston North',
  'Frankston South', 'Gisborne', 'Gladstone Park', 'Glen Waverley',
  'Greensborough', 'Greenvale', 'Hallam', 'Healesville', 'Heathmont',
  'Highett', 'Hillside', 'Hoppers Crossing', 'Huntingdale',
  'Keilor Downs', 'Keilor Lodge', 'Keysborough', 'Kings Park',
  'Lalor', 'Lower Plenty', 'Melton', 'Melton West', 'Mernda',
  'Mickleham', 'Mill Park', 'Mitcham', 'Montmorency', 'Montrose',
  'Mordialloc', 'Mount Waverley', 'Narre Warren', 'Noble Park',
  'Nunawading', 'Oakleigh East', 'Parkdale', 'Point Cook',
  'Ringwood', 'Ringwood East', 'Rockbank', 'Rowville', 'Sandringham',
  'Seaford', 'Skye', 'South Morang', 'Springvale',
  'Springvale South', 'St Albans', 'Sunbury', 'Tarneit',
  'Taylors Hill', 'Taylors Lakes', 'Tecoma', 'Templestowe',
  'The Basin', 'Thomastown', 'Truganina', 'Tullamarine', 'Vermont',
  'Wantirna', 'Wantirna South', 'Werribee', 'Westmeadows',
  'Williams Landing', 'Wollert', 'Wyndham Vale', 'Black Rock',
  'Blackburn North', 'Blackburn South', 'Bonbeach', 'Carrum',
  'Chelsea', 'Croydon North', 'Doveton', 'Ferntree Gully',
  'McKinnon', 'Melton South', 'Mentone', 'Mooroolbark', 'Mulgrave',
  'Roxburgh Park', 'Scoresby', 'Seabrook', 'Vermont South',
  'Warrandyte', 'Watsonia North', 'Wheelers Hill', 'Albanvale',
  'Ardeer', 'Attwood', 'Belgrave', 'Carrum Downs', 'Clayton South',
  'Cranbourne North', 'Kilsyth', 'Langwarrin', 'Notting Hill',
  'Patterson Lakes', 'Riddells Creek', 'Ringwood North', 'Sydenham',
  'Aspendale', 'Beaconsfield Upper', 'Chelsea Heights', 'Dallas',
  'Darley', 'Deer Park', 'Keilor', 'Meadow Heights', 'Mount Evelyn',
  'North Warrandyte', 'Pakenham', 'Sandhurst', 'Silvan', 'Wallan',
  'Chirnside Park', 'Croydon South', 'Derrimut', 'Diggers Rest',
  'Edithvale', 'Hampton Park', 'Knoxfield', 'St Helena', 'Upwey',
  'Bacchus Marsh', 'Coolaroo', 'Aspendale Gardens', 'Bullengarook',
  'Deepdene', 'Delahey', 'Hurstbridge', 'Kurunjang', 'Laverton',
  'Lilydale', 'Wonga Park', 'Endeavour Hills', 'Officer',
  'Waterways', 'Beaconsfield', 'Yarra Glen', 'Brookfield',
  'Whittlesea', 'Burnside', 'New Gisborne', 'Plumpton', 'Monbulk',
  'Warranwood', 'Avonsleigh', 'Wildwood', 'Plenty', 'Eumemmerring',
  'Gisborne South', 'Heatherton', 'Research', 'Botanic Ridge',
  'Coldstream', 'Hopetoun Park', 'Eynesbury', 'Wattle Glen',
  'Cranbourne West', 'Clyde North', 'Wandin North', 'Lysterfield',
  'Kalkallo', 'Werribee South'
  ])  # ใช้ชื่อชุมชนตามที่ต้องการ
feature_inputs['SellerG'] = st.st.selectbox("ชื่อผู้ขาย", options=['Jellis', 'Biggin', 'Nelson', 'Collins', 'Philip', 'LITTLE', 'Kay',
  'Marshall', 'Brad', 'Maddison', 'Barry', 'Considine', 'Rendina',
  'Propertyau', 'McDonald', 'Prof.', 'Harcourts', 'hockingstuart',
  'Thomson', 'Buxton', 'Greg', 'RT', "Sotheby's", 'Cayzer',
  'McGrath', 'Brace', 'Miles', 'Love', 'Barlow', 'Sweeney',
  'Village', 'Jas', 'Gunn&Co', 'Burnham', 'Williams', 'Point',
  'Compton', 'FN', 'Raine&Horne', 'Hunter', 'Noel', 'Hodges', 'Gary',
  'Woodards', 'Raine', 'Walshe', 'Alexkarbon', 'Weda', 'Frank',
  'Stockdale', 'Fletchers', 'Tim', 'Buxton/Marshall', 'Ray',
  'Purplebricks', 'Moonee', 'HAR', 'Edward', 'Chisholm', 'RW',
  'North', 'Ascend', 'Christopher', 'Bekdon', 'Mandy', 'R&H',
  'Fletchers/One', 'Assisi', 'One', 'Century', "O'Brien", 'C21',
  'Bayside', 'Anderson', 'Paul', 'Smart', 'First', 'Beller',
  'Matthew', 'Nick', 'Lindellas', 'Allens', 'Bells', 'Trimson',
  'Douglas', 'GL', 'YPA', "Tiernan's", 'Castran', 'Branon', 'J',
  'Rodney', 'Harrington', 'Dingle', 'Holland', 'Grantham',
  'Chambers', 'Pagan', 'Peter', 'hockingstuart/Advantage', 'Parkes',
  'Rounds', 'Ross', 'Rayner', 'Garvey', "O'Donoghues", 'Weast',
  'Kelly', 'Property', "Private/Tiernan's", 'Australian',
  "Abercromby's", 'Whiting', 'iTRAK', 'Del', 'Caine', 'Nicholson',
  'ASL', 'Changing', 'Re', 'RE', 'Walsh', 'Foxtons', 'Darren', 'Ham',
  'Vic', 'Haughton', 'Scott', 'Pride', 'Owen', 'Morleys', 'Wilson',
  'Buxton/Advantage', 'Professionals', 'Joe', 'Red', 'Thomas',
  'hockingstuart/Jellis', 'Craig', 'Naison', 'Sweeney/Advantage',
  'Eview', 'Jason', 'Melbourne', "D'Aprano", 'Morrison', 'Wood',
  'William', 'Coventry', 'Buckingham', 'Domain', 'Nardella', 'LJ',
  'Nguyen', 'Shape', 'Besser', 'Johnston', 'Redina', 'Clairmont',
  'Galldon', 'MICM', 'Elite', 'Buxton/Find', 'W.B.',
  'Harcourts/Barry', 'New', 'Parkinson', 'Geoff',
  'hockingstuart/Barry', 'Blue', 'Steveway', 'hockingstuart/Village',
  'VICPROP', 'Inner', 'Charlton', 'S&L', 'Calder', 'Homes', 'Zahn',
  'Mason', 'Landfield', 'David', 'Prowse', 'Ken', 'Rombotis', 'iOne',
  'hockingstuart/Sweeney', 'JMRE', 'Iconek', 'Crane', 'Leased',
  'Luxton', 'Hooper', 'JRW', 'Oak', 'White', 'Jim', 'Weston',
  '@Realty', 'Reed', 'Oriental', 'Max', 'Lucas', 'Real', 'Jeffrey',
  'Hall', 'buyMyplace', 'WHITEFOX', 'Christou', 'Marvelli', 'Metro',
  "Grant's", 'Hoskins', 'McLennan', 'Reliance', 'PRDNationwide',
  'Only', 'iSell', 'Obrien', 'Millership', 'Appleby', 'Ace',
  'Carter', 'M.J', 'iProperty', 'Triwest', 'Hayeswinckle',
  'Schroeder', 'VICProp', 'REMAX', 'Victory', 'Mindacom', 'Ryder',
  'Win', 'Sanctuary', 'Leeburn', 'Asset', 'Westside', 'LLC',
  'Mitchell', 'Darras', 'U', 'Leyton', 'Conquest', 'Prime',
  'Community', 'Free', 'Veitch', 'John', 'Peake', 'Dixon', 'Sell',
  'Ristic', 'Ash', 'JY', 'Upper', 'Daniel', 'Xynergy', 'VicHomes',
  'TRUE', 'McNaughton', 'Keatings', 'Leading', 'Bullen', 'Better',
  'Boutique', 'Aquire', 'Langwell', 'Gardiner', 'Follett', 'Silver',
  'Wyndham', 'Kaye', 'Nicholls', 'Bowman', 'Emerson', 'Meadows',
  'Just', 'Cooper', 'Open', 'The', 'L', 'Roger', 'Le', 'Joseph',
  'Flannagan', 'McEwing', 'Sandhurst', 'hockingstuart/hockingstuart',
  'Area', 'SN', 'Waterfront', 'P', 'PRD', 'Bombay', 'Quinta', 'T',
  'Rexhepi', 'LJH', 'Collings', 'Munn', 'Knight', 'MJ', 'Justin',
  'Eric', 'H', 'Alex', 'Can', 'Pavilion', 'Avion', 'Charter',
  'Surreal', 'Upside', 'OBrien', 'Meallin', 'Malvern', 'Bradly',
  'McGrath/Langwell', 'Burns', 'Sterling', 'K.R.Peters', 'Boran',
  'Benlor', 'Gold', 'hockingstuart/Biggin', 'Mega', 'Skad',
  'Ruralco', 'Create', 'Janice', 'R', 'G&H', 'Commercial', 'PSP',
  'Maitland', 'Unity', 'Sprint', 'McGrath/Buxton',
  'hockingstuart/Marshall', 'McGrath/First', 'Spencer',
  'voglwalpole', 'Watermark', 'Methven'
  ])  # ใช้ชื่อผู้ขายตามที่ต้องการ
feature_inputs['Age'] = st.number_input("อายุ", value = 30)

if st.button("Predict"):
    try:
        predicted_price = predict_price(feature_inputs, cleanhouse, model, scaler)
        st.success(f"Predicted Price: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


# Save the trained models (after training in your notebook)
model.save('your_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:  # Save the encoder
        pickle.dump(encoder, f)
