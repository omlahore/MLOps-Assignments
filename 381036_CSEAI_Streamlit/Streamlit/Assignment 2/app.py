import streamlit as st
import requests

st.title("Weather Information")

city_name = st.text_input("Enter a city name:", value="London")
api_key = "YOUR_API_KEY"

if st.button("Get Weather"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    data = response.json()

    if data.get("cod") != "404":
        main = data.get("main", {})
        weather_desc = data.get("weather", [{}])[0].get("description", "N/A")
        temp = main.get("temp", "N/A")
        humidity = main.get("humidity", "N/A")

        st.write(f"**Temperature:** {temp} K")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Description:** {weather_desc.capitalize()}")
    else:
        st.error("City not found. Please try again.")
