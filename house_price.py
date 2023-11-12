import joblib
import sklearn
import numpy as np
import pandas as pd
import streamlit as st
model = joblib.load("hyd_house_price_model.pkl")

Locations = ['Nizampet', 'Hitech City', 'Manikonda', 'Alwal', 'Kukatpally',
       'Gachibowli', 'Tellapur', 'Kokapet', 'Hyder Nagar', 'Mehdipatnam',
       'Narsingi', 'Khajaguda Nanakramguda Road', 'Madhapur',
       'Puppalaguda', 'Begumpet', 'Banjara Hills', 'AS Rao Nagar',
       'Pragathi Nagar Kukatpally', 'Miyapur', 'Mallampet',
       'Nanakramguda', 'Attapur', 'West Marredpally', 'Kompally',
       'Sri Nagar Colony', 'Hakimpet', 'Pocharam', 'Nagole', 'LB Nagar',
       'Meerpet', 'Kachiguda', 'Masab Tank', 'Kondapur', 'Saroornagar',
       'Uppal Kalan', 'Mallapur', 'Rajendra Nagar', 'Beeramguda',
       'Moosapet', 'Bachupally', 'Toli Chowki', 'Lakdikapul', 'Tarnaka',
       'Kistareddypet', 'Hafeezpet', 'Shaikpet', 'Amberpet', 'Kapra',
       'Trimalgherry', 'Habsiguda', 'Sanath Nagar', 'Darga Khaliz Khan',
       'Kothaguda', 'Balanagar', 'Jubilee Hills', 'raidurgam',
       'Murad Nagar', 'Chandanagar', 'East Marredpally', 'Aminpur',
       'Gajularamaram', 'Serilingampally', 'Malkajgiri', 'Mettuguda',
       'Venkat Nagar Colony', 'Kondakal', 'Gopanpally', 'Somajiguda',
       'Nallagandla Gachibowli', 'Krishna Reddy Pet', 'Bolarum',
       'Zamistanpur', 'Madhura Nagar', 'Ghansi Bazaar', 'Chintalakunta',
       'Chinthal Basthi', 'Nallakunta', 'Bowenpally', 'Bandlaguda Jagir',
       'Boduppal', 'Neknampur', 'Appa Junction Peerancheru',
       'Ambedkar Nagar', 'Vanasthalipuram', 'Moula Ali', 'Gandipet',
       'Nacharam', 'Appa Junction', 'Qutub Shahi Tombs', 'Abids',
       'Dilsukh Nagar', 'Quthbullapur', 'Sainikpuri', 'KTR Colony',
       'Bollaram', 'Karmanghat', 'Gajulramaram Kukatpally', 'Uppal',
       'Cherlapalli', 'Himayat Nagar', 'Rhoda Mistri Nagar', 'Chintalmet',
       'Hitex Road', 'ECIL', 'Boiguda', 'ECIL Main Road',
       'ECIL Cross Road', 'Rajbhavan Road Somajiguda',
       'Ramachandra Puram', 'TellapurOsman Nagar Road', 'Mansoorabad',
       'KRCR Colony Road', 'Pragati Nagar', 'Padmarao Nagar',
       'Paramount Colony Toli Chowki', 'BK Guda Internal Road',
       'muthangi', 'Pragathi Nagar', 'Yapral', 'Narayanguda', 'Kollur',
       'Bachupally Road', 'Old Bowenpally', 'Alapathi Nagar',
       'Arvind Nagar Colony', 'Matrusri Nagar', 'Pragathi Nagar Road',
       'Padma Colony', 'Happy Homes Colony', 'Old Nallakunta',
       'Sangeet Nagar', 'NRSA Colony', 'Adibatla', 'Methodist Colony',
       'Ameerpet', 'ALIND Employees Colony', 'Khizra Enclave', 'Medchal',
       'Dammaiguda', 'Suchitra', 'Whitefields', 'Mayuri Nagar',
       'Adda Gutta', 'Miyapur HMT Swarnapuri Colony',
       'Central Excise Colony Hyderabad', 'Basheer Bagh', 'Gopal Nagar',
       'Bachupaly Road Miyapur', 'Kushaiguda', 'Ashok Nagar',
       'Barkatpura', 'Madinaguda', 'Bagh Amberpet', 'new nallakunta',
       'BHEL', 'Sun City', 'Hydershakote', 'BK Guda Road',
       'Nallagandla Road', 'IDPL Colony', 'Ramnagar Gundu',
       'Alkapur township', 'Banjara Hills Road Number 12',
       'Panchavati Colony Manikonda', 'New Maruthi Nagar',
       'Madhavaram Nagar Colony', 'Miyapur Bachupally Road',
       'nizampet road', 'Kokapeta Village', 'HMT Hills', 'Tilak Nagar',
       'Chititra Medchal', 'Isnapur', 'D D Colony', 'DD Colony',
       'Patancheru Shankarpalli Road', 'Patancheru', 'Jhangir Pet',
       'Almasguda', 'Allwyn Colony', 'financial District',
       'Beeramguda Road', 'Pati', 'Karimnagar', 'Kollur Road',
       'Sun City Padmasri Estates', 'Chaitanyapuri', 'Nandagiri Hills',
       'Whitefield', 'Film Nagar', 'Kismatpur', 'Dr A S Rao Nagar Rd',
       'Dullapally', 'KPHB', 'Vivekananda Nagar Colony', 'Ameenpur',
       'Chintradripet', 'Ring Road', 'Saket', 'Kavuri Hills', 'manneguda',
       'Moti Nagar', 'Usman Nagar', 'Shadnagar', 'Bongloor',
       'Mailardevpally', 'Uppalguda', 'Tirumalgiri', 'Chikkadapally',
       'JNTU', 'hyderabad', 'Shamshabad', 'Srisailam Highway',
       'Domalguda', 'Lingampalli', 'Residential Flat Machavaram',
       'Whisper Valley', 'Tukkuguda Airport View Point Road',
       'Santoshnagar', 'Tolichowki', 'Domalguda Road', 'Shankarpalli',
       'Kothapet', 'Baghlingampally', 'Picket', 'Safilguda',
       'Sikh Village', 'Neredmet', 'Macha Bolarum', 'Kowkur',
       'Rakshapuram', 'west venkatapuram', 'Vidyanagar Adikmet',
       'Aushapur', 'Old Alwal', 'Secunderabad Railway Station Road',
       'Balapur', 'Hastinapur', 'chandrayangutta',
       'Balaji Hills Colony Venkatraya Nagar', 'Janachaitanya Colony',
       'Gurramguda', 'Paradise Circle']

labels = [162, 132,   9, 118,  70, 212, 107,  87, 139, 158, 103, 124, 180,
        39,   1, 177, 142, 130, 155,  19, 228,  85,  33, 111, 206,  81,
       175, 151, 120, 138, 134, 113, 197, 221, 131, 184,  37, 145,  24,
       216, 121, 211, 106,  80, 201,  11,  99, 218,  79, 194,  59, 114,
        30, 241, 148,  48,  68,  14,  71, 199, 129, 141,  94, 225, 112,
        76, 152, 117,  42, 233, 126,  74,  51,  53, 154,  45,  32,  40,
       205, 159,  16,  10, 224, 147,  73,  98, 150,  15, 182, 181, 192,
        97,  43, 101,  72, 220,  49,  84, 189,  52,  86,  65,  67,  66,
       183, 186, 213, 133,  96, 179, 167, 170,  22, 238, 176, 232, 157,
       109,  25, 164,   5,  17, 135, 178, 166,  82, 165, 149,   4, 140,
        13,   0, 104, 137,  58, 208, 136, 144,  46,  75,  26, 119,  18,
        35, 127,  27, 239,  21, 209,  88,  23, 153,  89, 187,   6, 168,
       161, 125, 143, 240, 108,  78, 214,  55,  90,  56,  57, 172, 171,
        93,   8,   7, 235,  38, 173, 100, 110, 210,  47, 230, 105,  63,
        64,  95, 227,  12,  54, 190, 193, 237, 146, 223, 200,  44, 128,
       222, 215,  50,  91, 236,  61, 122, 188, 229, 219, 196, 217,  62,
       203, 115,  28, 174, 191, 204, 160, 123, 116, 185, 242, 226,  20,
       163, 198,  31,  83, 234,  29,  60,  92,  77, 169]
location_mapping = dict(zip(Locations, labels))


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://e1.pxfuel.com/desktop-wallpaper/946/805/desktop-wallpaper-pin-on-city-in-hyderabad-city.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 20px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Title

st.markdown('<h1 class="title">House Price Prediction In Hyderabad</h1><br>', unsafe_allow_html=True)

# Sidebar with user input
st.sidebar.header("User Input")

# Number of Bedrooms
num_bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10)

# Area (in square feet)
area = st.sidebar.number_input("Area (in square feet)")

location = st.sidebar.selectbox("Location", Locations)

# Predict Button
predict_button = st.sidebar.button("Predict Price")

# Machine learning model (replace this with your actual model)
def predict_price(area,num_bedrooms,location):
    label_encoded_location = location_mapping.get(location, -1)
    # Check if the location exists in the mapping
    if label_encoded_location == -1:
        return("Location not found in the mapping")
    else:
        price = model.predict([[area,num_bedrooms, label_encoded_location]])
        amount = price[0]
        return st.markdown(f'<h4 class="title1">Price Predicted Succesfully in {location}  </h4><br>', unsafe_allow_html=True),st.write(f'<h3 class="title">Price is ₹f"₹{amount:,.2f}" only/-</h3><br>', unsafe_allow_html=True)

    

if predict_button:
    # Make a prediction
    predicted_price = predict_price(area,num_bedrooms ,location)

    # Display the prediction
    st.markdown('<h4 class="title1">Note: The Price is Predicted As per 2021 Statistics or Data</h4><br>', unsafe_allow_html=True)
    
    

# (Optional) Instructions or information
st.sidebar.markdown("Provide the number of bedrooms, area, and location to predict the house price.")

