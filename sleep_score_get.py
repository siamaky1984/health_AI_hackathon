import os
from typing import Any
from typing import List

from openCHA.tasks.affect import Affect
from openCHA.utils import get_from_env

import sys
import json


# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir) ) )))
sys.path.append(project_root+'/health_AI_hackathon/')
print('>>>', sys.path)
from sleep_predictor import *  #sleepQualityPredictor, build_sleep_predictor

class SleepScoreGet(Affect):
    """
    **Description:**

        This tasks calculates sleep score for a specific patient.
    """

    name: str = "affect_sleep_score_get"
    chat_name: str = "AffectSleepScoreGet"
    description: str = (
        "Returns the sleep score for a specific patient over a date or a period (if two dates are provided). "
        "This will return the detailed raw data and stores it in the datapipe."
    )

    dependencies: List[str] = []
    inputs: List[str] = [
        "user ID in string. It can be refered as user, patient, individual, etc. Start with 'par_' following with a number (e.g., 'par_1').",
        "start date of the sleep data in string with the following format: `%Y-%m-%d`",
        (
            "end date of the sleep data in string with the following format: `%Y-%m-%d`. "
            "If there is no end date, the value should be an empty string (i.e., '')"
        ),
    ]

    outputs: List[str] = ["array of sleep score" ]

    output_type: bool = True


    # dataset_source = "../../../../../health_AI_hackathon/ifh_affect"


    # folder_dict : dict= {"samsung":['awake_times','hrv_1min','pressure','pedometer'] ,
    #                     "oura":['sleep', 'activity', 'readiness', 'heart_rate']}

    # print(folder_dict)
    
    device_list: list = ['samsung', 'oura']

    file_list_samsung : list = ['awake_times','hrv_1min','pedometer']

    file_list_oura: list =['sleep', 'activity', 'readiness', 'heart_rate']


    local_dir: str = get_from_env(
    # "DATA_DIR", "DATA_DIR", "../../../../../health_AI_hackathon/ifh_affect"
    "DATA_DIR", "DATA_DIR", "../health_AI_hackathon/ifh_affect"
    )

    print('>>>>>>', local_dir)


    def _execute(
        self,
        inputs: List[Any],
    ) -> str:

        user_id = inputs[0].strip()

        df_list =[]

        ### TODO 
        ### combine the dfs of multiple sensors

        print('>>', self.device_list)

        for dev_i in self.device_list:
            # print('dev_i', dev_i)
            full_dir = os.path.join(
                self.local_dir, user_id, dev_i
            )
            print( dev_i )
            if 'samsung' in dev_i :
                file_list_sensor = self.file_list_samsung
            elif 'oura' in dev_i :
                file_list_sensor = self.file_list_oura

            for file_j in file_list_sensor:
                # print('file_j', file_j )

                # df_temp = self._get_data(
                # local_dir=full_dir,
                # file_name=file_j,
                # start_date=inputs[1].strip(),
                # end_date=inputs[2].strip(),
                # usecols=self.columns_to_keep,
                # )

                import pandas as pd
                df_temp = pd.read_csv(os.path.join(full_dir, file_j+'.csv'))

                df_list.append(df_temp)


        ### pass all df as a list to the predictor   
        # predictor, results, combined_features = build_sleep_predictor(df_list)

        predictor, results, combined_features =  build_sleep_predictor(df_list)
        
        print('results r2', results['r2'])
        print('results mse', results['mse'])

        results_summary ={}
        results_summary['r2'] = results['r2']
        results_summary['mse'] = results['mse']



        json_out = json.dumps( results_summary )
        return json_out
    


if __name__ == '__main__':
    
    sleep_score_get = SleepScoreGet()

    json_out = sleep_score_get._execute(['par_1', '01-02-2020', '09-08-2020'])

    print(json_out)
