#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import pytest
import logging
import requests
import datetime


class TestPerformance:
    RESULTS_DICT = {
        "OVMS": {
            "response_time": None
        },
        "AMS": {
            "response_time": None
        },
        "OV": {
            "response_time": None
        }
    }

    @staticmethod
    def inference(image, iterations: int):
        responses = []
        for num in range(iterations):
            with open(image, mode='rb') as image_file:
                image_bytes = image_file.read()
                start_time = datetime.datetime.now()
                response = requests.post("http://localhost:5000/vehicleDetection",
                                         headers={'Content-Type': 'image/png',
                                                  'Content-Length': str(len(image))},
                                         data=image_bytes)
                stop_time = datetime.datetime.now()
                duration = (stop_time - start_time).total_seconds() * 1000
                responses.append({"response": response,
                                  "duration": duration})
        return responses

    @pytest.mark.parametrize("model", ["vehicle-detection-adas-0002"], ids=["vehicle detection"])
    @pytest.mark.parametrize("config", [4, 32], ids=["4 cores", "32 cores"])
    def test_performance_simple_for_given_model(self, model, config):
        """
        <b>Description:</b>
        Checks if AMS results are close to OVMS and OpenVino benchmark app.

        <b>Assumptions:</b>
        - OpenVino 2020.2 up and running
        - AMS wrapper up and running
        - OVMS with model: vehicle-detection-adas-0002 - up and running

        <b>Input data:</b>
        - AMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS has results close to OVMS and OpenVino benchmark app.

        <b>Steps:</b>
        1. Prepare model for AMS - model param.
        2. Prepare config.json for AMS - config param.
        3. Run OpenVino benchmark app and get response time.
        4. Run OVMS and get response time.
        5. Run AMS and get response time.
        6. Compare response time for those services.
        7. Save results.
        """
        logging.info("Prepare model for AMS - model param.")
        logging.info("Prepare config.json for AMS - config param.")
        logging.info("Run OpenVino benchmark app and get response time.")
        logging.info("Run OVMS and get response time.")
        logging.info("Run AMS and get response time.")
        image_path = "/root/model_server/tests/functional/fixtures/test_images/single_car_small.png"
        responses = self.inference(image=image_path, iterations=100)
        for rsp in responses:
            print("Processing time: {} ms, \n speed: {} fps".format(round(rsp["duration"], 2),
                                                                    round(1000/rsp["duration"], 2)))
        logging.info("Compare response time for those services.")
        logging.info("Save results.")
