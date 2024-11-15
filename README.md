This is a set of scripts to analyze the data collected from multiple participants and sensors such as Samsung watch, Oura, imu, ppg, and etc.:
https://datadryad.org/stash/dataset/doi:10.7280/D1WH6T

We try to predict the sleep score of a participant in a given day based on his user activity. 
You should put the dataset inside the main folder and then run sleep_score_get.py.

To see openCHA you can run the test_cha.py and in the task list select affect_sleep_score_get.
This will then show the sleeping score and activity on the GRADIO GUI.

The visualizer also is available by running the run_sleep_predictor.py.


We use python 3.10 for this.


