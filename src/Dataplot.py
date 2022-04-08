from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class Dataplot():
    """
    Class for representing data in a plot
    """
    
    def __init__(
        self, 
        years_true_data: list,
        months_true_data: list,
        days_true_data: list,
        predicted_field,
        true_data: list,
        years_prediction: list,
        months_prediction: list,
        days_prediction: list,
        predictions,
        data_len
    ) -> None:
        plt.style.use('fivethirtyeight')
        # List and then convert to datetime object
        self.dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years_true_data, months_true_data, days_true_data)]
        self.dates = [datetime.strptime(date, '%Y-%m-%d') for date in self.dates]
        self.dates_predicted_fileds = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years_prediction, months_prediction, days_prediction)]
        self.dates_predicted_fileds = [datetime.strptime(date, '%Y-%m-%d') for date in self.dates_predicted_fileds]

        self.predicted_field = predicted_field
        self.true_data = true_data
        self.df_true_data = pd.DataFrame(data = {'date': self.dates, str(predicted_field): true_data})
        self.df_prediction_data = pd.DataFrame(data = {'date': self.dates_predicted_fileds, 'prediction': predictions})
        self.data_len = data_len
    
    def create_plot(self):
        """
        Create a new Plot based on given date in constructor
        """
        # Draw true data into graph
        list_of_dfs = [self.df_true_data.loc[i:i+self.data_len-1,:] for i in range(0, len(self.df_true_data), self.data_len)]
        for index, df in enumerate(list_of_dfs):
            if index == 0:
                plt.plot(df['date'], df[self.predicted_field], 'b-', label=self.predicted_field)
            else:
                plt.plot(df['date'], df[self.predicted_field], 'b-')
        # Draw the predicted values
        plt.plot(self.df_prediction_data['date'], self.df_prediction_data['prediction'], 'ro', label = 'prediction')
        plt.xticks(rotation = '60')
        plt.legend()
    
    def show(self, labelx, labely, title):
        """
        Show created dataplot
        Thread blocking
        Before showing plot, please call Dataplot.create_plot
        """
        plt.xlabel(labelx)
        plt.ylabel(labely) 
        plt.title(title)
        plt.show()
