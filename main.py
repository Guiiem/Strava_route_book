import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polyline
import folium
import subprocess
import pickle
import io
import os
import sys
from pathlib import Path
from PIL import Image
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

import config


def get_access_token():
    payload = {
        'client_id': config.CLIENT_ID,
        'client_secret': config.CLIENT_SECRET,
        'refresh_token': config.REFRESH_TOKEN,
        'grant_type': "refresh_token",
        'f': 'json'
    }
    res = requests.post("https://www.strava.com/oauth/token", data=payload, verify=False)
    access_token = res.json()['access_token']
    return access_token


def get_activities_dataframe(access_token: str) -> pd.DataFrame():
    headers = {'Authorization': f'Bearer {access_token}'}
    request_page = 1
    all_activities = []
    while True:
        print(f'Reading activities {200 * (request_page - 1)}-{200 * request_page}')
        params = {'per_page': 200, 'page': request_page}
        response = requests.get("https://www.strava.com/api/v3/athlete/activities", headers=headers, params=params)
        response.raise_for_status()
        activity_data = response.json()
        if len(activity_data) == 0:
            print("Reading complete!")
            break
        all_activities.extend(activity_data)
        request_page += 1

    return pd.DataFrame(all_activities)


def get_activities_within_dates(all_activites, initial_date, end_date):
    filtered_df = all_activites[(all_activites['start_date'] > initial_date) & (all_activites['start_date'] < end_date)]
    return filtered_df.sort_values(by='start_date')


def seconds2hh_mm(seconds: float) -> tuple[float, float]:
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return hh, mm


class Activity:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.activity_id = df['id']
        self.coordinates = polyline.decode(df['map']['summary_polyline'])
        self.longitude_array = np.array([x[0] for x in self.coordinates])
        self.latitude_array = np.array([x[1] for x in self.coordinates])

        self.request_header = None
        self.activity_details = None
        self.photos = None
        self.profiles = None

    def save_map_as_png(self, filename: str | Path):
        centroid = [np.mean(self.longitude_array), np.mean(self.latitude_array)]
        long_min = np.min(self.longitude_array)
        long_max = np.max(self.longitude_array)
        lat_min = np.min(self.latitude_array)
        lat_max = np.max(self.latitude_array)

        figsize = 2048 * 0.75
        aspect_ratio = 0.4

        m = folium.Map(location=centroid, width=figsize, height=figsize * aspect_ratio, zoom_control=False)
        folium.PolyLine(act.coordinates, color='red').add_to(m)
        m.fit_bounds([[long_min, lat_min], [long_max, lat_max]])
        img_data = m._to_png(3)
        img = Image.open(io.BytesIO(img_data))
        img.save(filename)

    def get_request_header(self):
        self.request_header = {'Authorization': f'Bearer {get_access_token()}'}

    def get_details(self):
        response = requests.get(f"https://www.strava.com/api/v3/activities/{self.activity_id}?include_all_efforts=",
                                headers=self.request_header)
        response.raise_for_status()
        self.activity_details = response.json()

    def get_photos(self):
        response = requests.get(f"https://www.strava.com/api/v3/activities/{self.activity_id}/photos?size=5000",
                                headers=self.request_header)
        response.raise_for_status()
        self.photos = response.json()

    def get_profiles(self):
        params = {'keys': ','.join(['altitude', 'temp', 'velocity_smooth']), 'key_by_type': "True"}
        response = requests.get(f"https://www.strava.com/api/v3/activities/{self.activity_id}/streams",
                                headers=self.request_header, params=params)
        response.raise_for_status()
        self.profiles = response.json()

    def plot_profiles(self, filename: str | Path):
        plt.ioff()  # Don't display figures

        distance = np.array(self.profiles['distance']['data']) / 1e3
        altitude = np.array(self.profiles['altitude']['data'])
        velocity = np.array(self.profiles['velocity_smooth']['data']) * 3.6
        if 'temp' in self.profiles.keys():
            temperature = np.array(self.profiles['temp']['data'])
        else:
            temperature = np.array([0] * len(distance))

        palette = sns.color_palette("Set2")

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 4))

        axs[0].plot(distance, altitude, color=palette[0])
        axs[0].fill_between(distance, [0] * len(distance), altitude, color=palette[0], alpha=0.5)
        axs[0].set_ylabel('Altitud [m]')
        axs[0].set_ylim([0, np.max(altitude) + 20])

        axs[1].plot(distance, velocity, color=palette[2])
        axs[1].fill_between(distance, [0] * len(distance), velocity, color=palette[2], alpha=0.5)
        axs[1].set_ylabel('Velocitat [km/h]')
        axs[1].set_ylim([0, np.max(velocity) + 1])

        axs[2].plot(distance, temperature, color=palette[1])
        axs[2].fill_between(distance, [0] * len(temperature), temperature, color=palette[1], alpha=0.5)
        axs[2].set_ylabel('T [degC]')
        axs[2].set_ylim([0, np.max(temperature) + 1])

        axs[2].set_xlabel('Distància [km]')
        axs[2].set_xlim(distance[0], distance[-1])
        plt.tight_layout()

        plt.savefig(filename)

    def download_photos(self, folder: str | Path):
        filenames, captions = [], []
        for i, photo in enumerate(self.photos):
            filename = Path(folder) / f'{self.activity_id}_{i}.jpg'

            if filename.exists():
                filenames.append(filename)
                captions.append(photo['caption'])
                continue

            url = photo['urls']['5000']
            res = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(res.content)

            filenames.append(filename)
            captions.append(photo['caption'])
        return filenames, captions


class LaTex:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.figures_folder = self.folder / 'figures'
        self.filename = self.folder / 'main.tex'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.figures_folder, exist_ok=True)

    def __enter__(self):
        self.start_document()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_document()
        self.compile_document()

    def start_document(self):
        preamble = ['\\documentclass[]{article} \n',
                    '\\usepackage[margin=20mm]{geometry} \n',
                    '\\usepackage{graphicx} % For figures\n'
                    '\\usepackage{placeins} % Floatbarrier\n',
                    '\\setlength{\\parindent}{0pt} % Indentation is clearly not cool \n \n',
                    '\\begin{document} \n']
        with open(self.filename, 'w') as file:
            for line in preamble:
                file.write(line)

    def add_activity_section(self, act: Activity):
        map_filename = f'map_{act.activity_id}.png'
        map_path = self.figures_folder / map_filename
        if not map_path.exists():
            act.save_map_as_png(map_path)

        profiles_filename = f'stats_{act.activity_id}.pdf'
        profiles_path = self.figures_folder / profiles_filename
        if not profiles_path.exists():
            act.plot_profiles(profiles_path)

        sec_text = ['\\section{' + act.activity_details['name'] + '} \n',
                    '\\begin{minipage}{0.35\\textwidth} \n']

        # Maybe we don't have temperature if the GPS does not record
        if 'average_temp' in act.activity_details.keys():
            average_temperature = act.activity_details['average_temp']
        else:
            average_temperature = "-"

        summary_table = ['\\begin{center} \n',
                         '\\begin{tabular}{c|c} \n',
                         f'Data & {act.activity_details["start_date_local"][:10]} \\\\ \n',
                         f'Hora inici & {act.activity_details["start_date_local"][11:-1]} \\\\ \n',
                         f'Distància [km] & {act.activity_details["distance"] / 1000:.1f} \\\\ \n',
                         f'Desnivell acumulat [m] & {act.activity_details["total_elevation_gain"]:.0f} \\\\ \n',
                         'Temps en moviment [h] & {}:{:02d} \\\\ \n'.format(
                             *seconds2hh_mm(act.activity_details['moving_time'])),
                         'Temps total [h] & {}:{:02d} \\\\ \n'.format(
                             *seconds2hh_mm(act.activity_details['elapsed_time'])),
                         f'Velocitat mitjana [km/h] & {act.activity_details["average_speed"] * 3.6:.1f} \\\\ \n',
                         f'Temperatura mitjana [degC] & {average_temperature} \\\\ \n',
                         '\\end{tabular} \n',
                         '\\end{center} \n']

        sec_text.extend(summary_table)
        sec_text.extend([
            '\\end{minipage} \n',
            '\\hspace{0.02\\textwidth} \n',
            '\\begin{minipage}{0.6\\textwidth}\n',
            '\\includegraphics[width=\\textwidth]{figures/' + profiles_filename + '} \n',
            '\\end{minipage} \\\\ \n'])

        description = act.activity_details['description'].replace("%", "\%")
        sec_text.extend(
            ['\\includegraphics[width=\\textwidth]{figures/' + map_filename + '} \\\\ \\\\ \n',
             description,
             '\\newpage \n'])

        photos_path, captions = act.download_photos(self.figures_folder)
        for i in range(len(photos_path)):
            filename = photos_path[i].name
            pictures = ['\\begin{minipage}{0.5\\textwidth} \n',
                        '\\includegraphics[width=\\textwidth]{figures/' + filename + '} \n',
                        '\\caption{' + captions[i] + '} \n',
                        '\\end{minipage} \n']
            sec_text.extend(pictures)

        sec_text.append('\\newpage \n \n')

        with open(self.filename, 'a', encoding="utf-8") as file:
            for line in sec_text:
                file.write(line)

    def end_document(self):
        with open(self.filename, 'a', encoding="utf-8") as file:
            file.write('\\end{document} \n')

    def compile_document(self):
        command = f'latexmk -pdf {self.filename} -quiet -time -output-directory={self.folder} -auxdir={self.folder}'
        subprocess.run(command, shell=True)


def main(initial_date: str, end_date: str, book_folder: str) -> int:
    token = get_access_token()
    data = get_activities_dataframe(token)
    trimmed_data = get_activities_within_dates(data, initial_date, end_date)

    # Get the data of all activities, read it from the data folder or make the requests
    activities = []
    os.makedirs(Path(book_folder) / 'data', exist_ok=True)
    for i, (idx, row) in enumerate(trimmed_data.iterrows()):
        print(f'Reading activity {i + 1} out of {len(trimmed_data)}...')
        activity_id = row['id']
        activity_filename = Path(f'{book_folder}/data/{activity_id}.pickle')
        if activity_filename.exists():
            with open(activity_filename, 'rb') as file:
                activities.append(pickle.load(file))
        else:
            act = Activity(row)
            act.get_request_header()
            act.get_details()
            act.get_profiles()
            act.get_photos()
            activities.append(act)
            with open(activity_filename, 'wb') as file:
                pickle.dump(act, file)

    # Generate the document
    with LaTex(folder=book_folder) as tex:
        for i, act in enumerate(activities):
            print(f'Writing activity {i + 1} out of {len(activities)}...')
            tex.add_activity_section(act)

    return 0


if __name__ == '__main__':
    initial_date = '2022-09-10'
    end_date = '2022-09-12'
    book_folder = 'testing'

    sys.exit(main(initial_date, end_date, book_folder))
