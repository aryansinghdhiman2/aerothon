import requests
import urllib3
import subprocess
import json
import time


def get_location_requests():
    a = time.time()
    response = requests.get("http://localhost:5000/location/")
    b = time.time()
    if response.status_code == 200:
        location = response.json()
        print("Location using requests:", location)
        print("Time taken === ", (b-a))
    else:
        print(f"Error: {response.status_code}")


get_location_requests()


def get_location_urllib3():
    http = urllib3.PoolManager()
    a = time.time()

    response = http.request("GET", "http://localhost:5000/location/")
    b = time.time()

    if response.status == 200:
        location = response.data.decode('utf-8')
        print("Location using urllib3:", location)
        print("Time taken === ", (b-a))

    else:
        print(f"Error: {response.status}")


get_location_urllib3()


def get_location_curl():
    a = time.time()

    result = subprocess.run(
        ["curl", "-s", "http://localhost:5000/location/"],
        capture_output=True,
        text=True
    )
    b = time.time()

    if result.returncode == 0:
        location = json.loads(result.stdout)
        print("Location using curl:", location)
        print("Time taken === ", (b-a))

    else:
        print(f"Error: {result.returncode}")


get_location_curl()
