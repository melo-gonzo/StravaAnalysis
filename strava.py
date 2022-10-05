import datetime
import os
import pickle
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stravalib.client import Client
from stravalib.model import Split


from strava_payload import *

USER = "hannah"
payload, strava_code = get_payload(USER)


class Strava:
    def __init__(self, payload, strava_code):
        self.client = Client()
        self.auth_url = "https://www.strava.com/oauth/token"
        self.activites_url = "https://www.strava.com/api/v3/athlete/activities"
        self.payload = payload
        self.strava_code = strava_code
        self.data_exists = os.path.isfile(f"{USER}_data.csv")
        self.do_download = True if not self.data_exists else False
        self.fig_dir = "figs/"
        if not os.path.exists(self.fig_dir):
            self.fig_dir = os.makedirs(self.fig_dir)

    def check_if_exits(self):
        return os.path.isfile(f"{USER}_data.csv")

    def get_access_token(self):
        code = self.strava_code
        try:
            access_token = self.client.exchange_code_for_token(
                client_id=self.payload["client_id"],
                client_secret=self.payload["client_secret"],
                code=code,
            )
            with open(f"./{USER}_access_token.pickle", "wb") as f:
                pickle.dump(access_token, f)
        except Exception as e:
            print(traceback.format_exc())

        with open(f"./{USER}_access_token.pickle", "rb") as f:
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
            self.data = self.client.get_activities(limit=10000)

    def organize_data(self):
        if not self.do_download:
            self.data = pd.read_csv(f"{USER}_data.csv")
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
        df["week_of_year"] = df["start_date_local"].dt.week
        df["day_of_year"] = df["start_date_local"].dt.day_of_year
        df["time_of_day"] = df["start_date_local"].dt.time
        df["day_of_week"] = df["start_date_local"].dt.day_name()
        df["month_of_year"] = df["start_date_local"].dt.month
        self.data = df
        self.data.to_csv(f"{USER}_data.csv")

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
            splits_df = pd.read_csv(f"{USER}_splits.csv")
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

                splits_df.to_csv(f"{USER}_splits.csv")


def date_distance(runs):
    rd = {}
    for idx in range(len(runs)):
        date = runs.iloc[idx]["start_date_local"].split(" ")[0]
        distance = runs.iloc[idx]["distance"] / 1000
        if date in rd.keys():
            rd[date] += distance
        else:
            rd[date] = distance

    new_runs = pd.Series(rd)

    idx = pd.date_range("2018-03-15", "2022-09-03")
    # s = pd.Series({
    #     '09-02-2020': 2,
    #     '09-03-2020': 1,
    #     '09-06-2020': 5,
    #     '09-07-2020': 1
    # })
    new_runs.index = pd.DatetimeIndex(new_runs.index)
    new_runs = new_runs.reindex(idx, fill_value=0)
    new_runs.to_csv("date_distance.csv")


######
###### Standard Workflow
######
# data = Strava(payload, strava_code)
# data.get_access_token()
# data.download_data()
# data.organize_data()
######
###### Download Splits Data
######
# data = Strava(payload, strava_code)
# data.get_access_token()
# data.download_data()
# data.organize_data()
# runs = data.get_activity_type("Run")
# runs["pace"] = runs["moving_time"] / (runs["distance_mile"]) / 60
# try:
#     data.get_splits()
# except Exception as e:
#     print(traceback.format_exc())
#     print(6 * "\n")
#     print("continuing")
######
###### Make Histogram From Splits Data
######
data = Strava(payload, strava_code)
data.data = pd.read_csv(f"{USER}_splits.csv")
runs = data.data.copy()
runs = runs.iloc[1:]
runs["pace"] = runs["s_moving_time"] / (runs["s_distance"] * 0.0006214) / 60
data.year_histogram(
    runs,
    min=4,
    max=12,
    bins_per_div=6,
    activity_type="Split",
    metric="pace",
    weight_unit="None",
    use_weights=False,
)
######
###### Make Plots From Standard Run Data
######
# data = Strava(payload, strava_code)
# data.download_data()
# data.organize_data()
# runs = data.get_activity_type("Run")
# runs["pace"] = runs["moving_time"] / (runs["distance_mile"]) / 60
# data.year_histogram(runs, min=4, max=12, bins_per_div=6,
#                     activity_type='Run', metric='pace', weight_unit='distance_mile', use_weights=True)
# data.year_histogram(runs, min=0, max=27, bins_per_div=2,
#                     activity_type='Run', metric='distance_mile', weight_unit='distance_mile', use_weights=True)
######
###### Make Histogram of Hour Run Started
######
# plt.close()
# df = pd.read_csv('data.csv')
# hour = df['time_of_day']
# hour = [int(_.split(':')[0]) for _ in hour]
# plt.hist(hour, range(0,25), edgecolor="w",color='k')
# plt.savefig('HannahHourOfRunHistogram.png')
######
###### Make Histograms Of Splits Over Months To Visualize Change
######
# data = Strava(payload, strava_code)
# data.data = pd.read_csv(f"{USER}_splits.csv")
# runs = data.data.copy()
# runs = runs.iloc[1:]
# runs["pace"] = runs["s_moving_time"] / (runs["s_distance"] * 0.0006214) / 60
# r = list(zip([[2019]*12+[2020]*12+[2021]*12+[2022]*12][0],[list(range(1,13))*4][0]))
# r_ = r[6:-3]
# min = 4
# max = 12
# bins_per_div = 6
# activity_type = "Split"
# metric = "pace"
# weight_unit = "None"
# use_weights = False
# alpha = np.linspace(0.5, 1, 4)[::-1]
# # colors = ["g", "r", "b", "k"]
# colors = ['k', 'k','k', 'k']
# runs[metric] = runs[metric][runs[metric] >= min]
# runs[metric] = runs[metric][runs[metric] >= min]
# runs_og = runs.copy()
# bins = np.linspace(min, max, (max - min) * bins_per_div + 1)
# for r in range(3, len(r_)):
#     # import pdb; pdb.set_trace()
#     for idx, loc in enumerate(range(r - 3, r + 1)):
#         try:
#             print(r_[loc])
#             # import pdb; pdb.set_trace()
#             runs = runs_og[runs_og["year"] == r_[loc][0]]
#             runs = runs[runs["month_of_year"] == r_[loc][1]]
#             runs = runs[metric]
#             n, _ = np.histogram(runs, bins=bins)
#             if metric == "pace":
#                 try:
#                     runs.replace([np.inf, -np.inf], np.nan, inplace=True)
#                     runs.dropna(inplace=True)
#                     label_mean = data.get_km_per_min(np.mean(runs))
#                 except Exception:
#                     pass
#             else:
#                 label_mean = str(np.round(np.mean(runs), 2))

#             label_mean = " avg: " + label_mean

#             if use_weights:
#                 weights = runs[weight_unit]
#                 y_label = weight_unit
#             else:
#                 weights = np.ones(len(runs))
#                 y_label = activity_type + "s"
#             plt.hist(
#                 runs,
#                 bins=bins,
#                 label=f"{r_[loc][0]}-{r_[loc][1]}:   {label_mean}",
#                 color=colors[idx],
#                 alpha=alpha[idx],
#                 edgecolor="w",
#                 weights=weights,
#             )
#             if idx == 0:
#                 start_date = f"{r_[loc][0]}-{r_[loc][1]}"
#         except Exception as e:
#             print(traceback.format_exc())
#             print(6 * "\n")
#             print("continuing")
#     colors = colors[1:] + colors[:1]
#     end_date = f"{r_[loc][0]}-{r_[loc][1]}"
#     plt.legend()
#     plt.title(f"{start_date} to {end_date}")
#     plt.xlim([min, max])
#     plt.ylim([0, 40])
#     plt.ylabel("# {}".format(y_label))
#     plt.xlabel("{} {}".format("split", metric))
#     plt.savefig(data.fig_dir + f"{start_date} to {end_date}".format("split"))
#     plt.close()
##################################################
##################################################
