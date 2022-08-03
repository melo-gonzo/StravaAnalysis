import pdb
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stravalib.client import Client
import time
import pickle
from stravalib.model import Split
import traceback
from strava_payload import *


class Strava:
    def __init__(self, payload, strava_code):
        self.client = Client()
        self.auth_url = "https://www.strava.com/oauth/token"
        self.activites_url = "https://www.strava.com/api/v3/athlete/activities"
        self.payload = payload
        self.strava_code = strava_code
        self.data_exists = os.path.isfile("data.csv")
        self.do_download = True if not self.data_exists else False
        self.fig_dir = "figs/"
        if not os.path.exists(self.fig_dir):
            self.fig_dir = os.makedirs(self.fig_dir)

    def check_if_exits(self):
        return os.path.isfile("data.csv")

    def get_access_token(self):
        code = self.strava_code
        try:
            access_token = self.client.exchange_code_for_token(
                client_id=self.payload["client_id"],
                client_secret=self.payload["client_secret"],
                code=code,
            )
            with open("./access_token.pickle", "wb") as f:
                pickle.dump(access_token, f)
        except Exception as e:
            print(traceback.format_exc())

        with open("./access_token.pickle", "rb") as f:
            access_token = pickle.load(f)

        print("Latest access token read from file:")
        access_token

        if time.time() > access_token["expires_at"]:
            print("Token has expired, will refresh")
            refresh_response = self.client.refresh_access_token(
                client_id=self.payload["client_id"],
                client_secret=self.payload["client_secret"],
                refresh_token=access_token["refresh_token"],
            )
            access_token = refresh_response
            with open("../access_token.pickle", "wb") as f:
                pickle.dump(refresh_response, f)
            print("Refreshed token saved to file")
            self.client.access_token = refresh_response["access_token"]
            self.client.refresh_token = refresh_response["refresh_token"]
            self.client.token_expires_at = refresh_response["expires_at"]

        else:
            print(
                "Token still valid, expires at {}".format(
                    time.strftime(
                        "%a, %d %b %Y %H:%M:%S %Z",
                        time.localtime(access_token["expires_at"]),
                    )
                )
            )
            self.client.access_token = access_token["access_token"]
            self.client.refresh_token = access_token["refresh_token"]
            self.client.token_expires_at = access_token["expires_at"]

    def download_data(self):
        if self.do_download:
            print("Downloading Data ...")
            self.data = self.client.get_activities(limit=1000)

    def organize_data(self):
        if not self.do_download:
            self.data = pd.read_csv("data.csv")
            return
        cols = [
            "name",
            "upload_id",
            "type",
            "start_date_local",
            "distance",
            "moving_time",
            "elapsed_time",
            "total_elevation_gain",
            "elev_high",
            "elev_low",
            "average_speed",
            "max_speed",
            "average_heartrate",
            "max_heartrate",
            "start_latitude",
            "start_longitude",
            "average_cadence",
        ]

        data = []
        for activity in self.data:
            my_dict = activity.to_dict()
            data.append([activity.id] + [my_dict.get(x) for x in cols])

        cols.insert(0, "id")
        df = pd.DataFrame(data, columns=cols)
        df["type"] = df["type"].replace("Walk", "Hike")
        df["distance_km"] = df["distance"] / 1e3
        df["distance_mile"] = df["distance"] * 0.0006214
        df["start_date_local"] = pd.to_datetime(df["start_date_local"])
        df["year"] = df["start_date_local"].dt.year
        df["day_of_week"] = df["start_date_local"].dt.day_name()
        df["month_of_year"] = df["start_date_local"].dt.month
        self.data = df
        self.data.to_csv("data.csv")

    def get_activity_type(self, activity):
        sub = self.data.loc[self.data["type"] == activity]
        return sub

    def get_km_per_min(self, speed):
        min = str(int(np.floor(speed)))
        sec = np.round(60 * (speed - np.floor(speed)), 2)
        if sec < 10:
            sec = "0" + str(sec)
        else:
            sec = str(sec)
        return min + ":" + sec

    def year_histogram(
        self,
        data,
        min=4,
        max=12,
        bins_per_div=6,
        activity_type="Run",
        metric="pace",
        weight_unit="mile",
        use_weights=True,
    ):
        # data = self.get_activity_type(activity_type)
        years = data["year"].unique()[::-1]

        alpha = np.linspace(0.5, 1, len(years))[::-1]
        data[metric] = data[metric][data[metric] >= min]
        data[metric] = data[metric][data[metric] <= max]
        bins = np.linspace(min, max, (max - min) * bins_per_div + 1)
        n, _ = np.histogram(data[metric], bins=bins)

        if use_weights:
            weights = data[weight_unit]
            y_label = weight_unit
        else:
            weights = np.ones(len(data[metric]))
            y_label = activity_type + "s"

        if metric == "pace":
            label_mean = self.get_km_per_min(np.mean(data[metric]))
        else:
            label_mean = str(np.round(np.mean(data[metric]), 2))

        label_mean = " avg: " + label_mean

        plt.hist(
            data[metric],
            bins=bins,
            label=label_mean,
            color="k",
            edgecolor="w",
            weights=weights,
        )
        plt.legend()
        plt.title("All {}s".format(activity_type))
        plt.xlim([min, max])
        plt.ylabel("# {}".format(y_label))
        plt.xlabel("{} {}".format(activity_type, metric))
        plt.savefig(self.fig_dir + "all_{}_hist.png".format(activity_type))

        fig2, ax2 = plt.subplots()
        for idx, year in enumerate(years):
            fig1, ax1 = plt.subplots()
            sub = runs[runs["year"] == year]
            if use_weights:
                weights = sub[weight_unit]
                y_label = weight_unit
            else:
                weights = np.ones(len(sub[metric]))
                y_label = activity_type + "s"

            if metric == "pace":
                label_mean = self.get_km_per_min(np.mean(sub[metric]))
            else:
                label_mean = str(np.round(np.mean(sub[metric]), 2))

            label_mean = " avg: " + label_mean

            ax1.hist(
                sub[metric],
                bins=bins,
                label=str(year) + " " + label_mean,
                color="k",
                edgecolor="w",
                weights=weights,
            )
            ax2.hist(
                sub[metric],
                bins=bins,
                alpha=alpha[idx],
                label=str(year) + " " + label_mean,
                edgecolor="w",
                weights=weights,
            )
            ax1.set_title("{} {}s: {}".format(year, activity_type, len(sub[metric])))
            ax1.set_xlim([min, max])
            ax1.legend()
            ax1.set_ylabel("# {}".format(y_label))

            ax1.set_xlabel("{} {}".format(activity_type, metric))
            fig1.savefig(self.fig_dir + "{}_{}_hist.png".format(year, activity_type))

        ax2.legend()
        ax2.set_title("All {}s by Year".format(activity_type))
        ax2.set_xlim([min, max])
        ax2.set_ylabel("# {}".format(y_label))
        ax2.set_xlabel("{} {}".format(activity_type, metric))
        fig2.savefig(self.fig_dir + "yearly_{}_hist.png".format(activity_type))

    def get_splits(self):
        valid_fields = {
            "id": None,
            "year": None,
            "month_of_year": None,
            "day_of_week": None,
            "s_distance": None,
            "s_elapsed_time": None,
            "s_elevation_difference": None,
            "s_moving_time": None,
            "s_split": None,
            "s_average_speed": None,
            "s_pace_zone": None,
        }
        try:
            splits_df = pd.read_csv("splits.csv")
        except FileNotFoundError:
            splits_df = pd.DataFrame([valid_fields])

        for idx in range(len(runs)):
            activity_num = int(runs.iloc[idx].id)
            if activity_num not in splits_df["id"].values:
                s_data = data.client.get_activity(activity_num)
                splits = s_data.splits_metric
                year = runs.iloc[idx]["year"]
                month_of_year = runs.iloc[idx]["month_of_year"]
                day_of_week = runs.iloc[idx]["day_of_week"]
                if splits is not None:
                    for split in splits:
                        valid_fields = {
                            "id": activity_num,
                            "year": year,
                            "month_of_year": month_of_year,
                            "day_of_week": day_of_week,
                            "s_distance": None,
                            "s_elapsed_time": None,
                            "s_elevation_difference": None,
                            "s_moving_time": None,
                            "s_split": None,
                            "s_average_speed": None,
                            "s_pace_zone": None,
                        }
                        valid_fields["s_distance"] = split.distance.num
                        valid_fields["s_moving_time"] = split.moving_time.seconds
                        ndf = pd.DataFrame([valid_fields])
                        splits_df = pd.concat([splits_df, ndf])
                        # splits_df.append(ndf)

                splits_df.to_csv("splits.csv")


data = Strava(payload, strava_code)
data.get_access_token()
data.download_data()
data.organize_data()
runs = data.get_activity_type("Run")
runs["pace"] = runs["moving_time"] / (runs["distance_mile"]) / 60
try:
    data.get_splits()
except Exception as e:
    print(traceback.format_exc())


##################################################
##################################################
##################################################
##################################################

data.data = pd.read_csv("splits.csv")
runs = data.data.copy()
runs = runs.iloc[1:]
runs["pace"] = runs["s_moving_time"] / (runs["s_distance"] * 0.0006214) / 60
data.year_histogram(
    runs,
    min=4,
    max=12,
    bins_per_div=6,
    activity_type="Run",
    metric="pace",
    weight_unit="None",
    use_weights=False,
)

##################################################
##################################################
##################################################
##################################################
### Make some nice plots
# data.year_histogram(runs, min=4, max=12, bins_per_div=6,
#                     activity_type='Run', metric='pace', weight_unit='distance_mile', use_weights=True)

# data.year_histogram(runs, min=0, max=27, bins_per_div=2,
#                     activity_type='Run', metric='distance_mile', weight_unit='distance_mile', use_weights=True)
