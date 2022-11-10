import os
import pickle
import geopy
from geopy.geocoders import Nominatim

# https://geopy.readthedocs.io/en/stable/


class GeographyDataManager:
    def __init__(self):
        self.save_path = "state_loc.pkl"

    def generate_us_state_coordinates(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as file:
                state_xy = pickle.load(file)
        else:
            from geopy.geocoders import Nominatim

            loc = Nominatim(user_agent="GetLoc")
            state_xy = {}
            for idx, state in tqdm(enumerate(df.columns.to_list())):
                getLoc = loc.geocode(f"{state}, United States")
                state_xy[state] = (getLoc.longitude, getLoc.latitude)
            with open(self.save_path, "wb") as file:
                pickle.dump(state_xy, file)
        return state_xy
