This is a set of scripts to analyze the data collected from multiple participants and sensors such as Samsung watch, Oura, imu, ppg, and etc.:
https://datadryad.org/stash/dataset/doi:10.7280/D1WH6T

We use python 3.12. 

It's good to use conda. Once a new environemnt is created install the requirements.

pip3 install -r requirements.txt

We try to predict the sleep score of a participant in a given day based on his user activity. 
You should put the dataset inside the main folder and then run sleep_score_get.py.

To see openCHA you can run the test_cha.py and in the task list select affect_sleep_score_get.
This will then show the sleeping score and activity on the GRADIO GUI. You can ask questions about r2 score and mse from the Gradio chatbot.
It will load the sleep_score_get function to run a random forest and predict the sleeping quality of the following night.

The visualizer also is available by running the following:

streamlit run sleep_predictor.py -- --data_path [PATH to DATA folder IFH] --start_date [YYY-MM-DD] --end_date [YYYY-MM-DD]
 
This will use a sample data in this repo to show some stats.



