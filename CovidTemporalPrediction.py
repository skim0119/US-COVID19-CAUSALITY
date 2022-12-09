from CovidRawDataManager import CovidRawDataManager
import pandas as pd
import pycaret.regression as pcr
import matplotlib.pyplot as plt


class CovidTemporalPredictor:
    """
    Class for time series prediction of covid case date
    for a given state

    """

    def __init__(self, state_name: str, data: CovidRawDataManager = None) -> None:
        data = data or CovidRawDataManager()
        self.statewise_cases_df = data.generate_statewise_history_data(
            states_df_label="cases"
        )
        self.daily_statewise_cases_df = data.generate_daily_statewise_data_from_history(
            statewise_history_df=self.statewise_cases_df
        )
        self.state_name = state_name
        self.ymd_statewise_cases_df = self.extract_year_month_day_state_df(
            self.statewise_cases_df
        )
        self.ymd_daily_statewise_cases_df = self.extract_year_month_day_state_df(
            self.daily_statewise_cases_df
        )
        self.input_df = self.ymd_daily_statewise_cases_df
        plt.style.use("seaborn")

    def extract_year_month_day_state_df(
        self, case_df: pd.DataFrame, min_case_tol: int = 10
    ) -> pd.DataFrame:
        """Extract dataframe with YMD format date with cases data"""
        ymd_df = case_df.copy()
        ymd_df["date"] = ymd_df.index
        ymd_df["year"] = [i.year for i in ymd_df["date"]]
        ymd_df["month"] = [i.month for i in ymd_df["date"]]
        ymd_df["day"] = [i.day for i in ymd_df["date"]]
        ymd_df.reset_index(drop=True, inplace=True)
        ymd_df = ymd_df[["date", "year", "month", "day"] + [self.state_name]]
        # not interested in days when less than 10 cases detected
        ymd_df = ymd_df.loc[ymd_df[self.state_name] > min_case_tol]
        return ymd_df

    def plot_input_data(self, ax: plt.Axes = None) -> plt.Axes:
        """Plot input data trends"""
        return self.input_df.plot(
            x="date",
            y=[self.state_name],
            ylabel="daily cases",
            ax=ax,
        )

    def setup_model(self, train_fraction: float = 0.8) -> None:
        """Setup the model on data using pycaret"""
        print(self.state_name + " model setup")
        _ = pcr.setup(
            self.input_df,
            target=self.state_name,
            train_size=train_fraction,
            data_split_shuffle=False,
            fold_strategy="timeseries",
            fold=3,
            ignore_features=["state"],
            numeric_features=["year", "month", "day"],
            session_id=123,
        )

    def train_model(self, error_metric: str = "MAE", verbose=True) -> None:
        """Train the model on data using pycaret"""
        if verbose:
            print(self.state_name + ": Training on different models")
        self.best_model = pcr.compare_models(sort=error_metric)
        if verbose:
            print("Evaluate error metrics for test data")
        _ = pcr.predict_model(self.best_model)
        return self.best_model

    def plot_model_fit(self, ax: plt.Axes = None) -> plt.Axes:
        """Plot model fit vs data comparison"""
        self.predictions = pcr.predict_model(self.best_model, data=self.input_df)
        self.predictions["date"] = self.input_df["date"]
        self.predictions.rename(
            columns={"prediction_label": self.state_name + " model fit"}, inplace=True
        )
        return self.predictions.plot(
            x="date",
            y=[self.state_name, self.state_name + " model fit"],
            ylabel="daily cases",
            ax=ax,
        )

    def finalise_model(self) -> None:
        """Finalise model for future prediction"""
        self.final_best_model = pcr.finalize_model(self.best_model)

    def generate_future_prediction_input_df(self, future_months: int) -> pd.DataFrame:
        """generate input features for future prediction"""
        last_date = self.input_df["date"].iloc[-1].date()
        future_dates = pd.date_range(
            start=last_date, end=last_date + pd.offsets.DateOffset(months=future_months)
        )
        future_df = pd.DataFrame()
        future_df["date"] = future_dates
        future_df["year"] = [i.year for i in future_dates]
        future_df["month"] = [i.month for i in future_dates]
        future_df["day"] = [i.day for i in future_dates]
        return future_df

    def predict_and_plot_future(
        self,
        future_months: int = 12,
        ax: plt.Axes = None,
        return_data_only: bool = False,
    ) -> plt.Axes:
        """Predict future based on best model and plot with input data"""
        self.finalise_model()
        future_df = self.generate_future_prediction_input_df(
            future_months=future_months
        )
        self.predictions_future = pcr.predict_model(
            self.final_best_model, data=future_df
        )
        self.predictions_future["date"] = future_df["date"]
        self.predictions_future.rename(
            columns={"prediction_label": self.state_name + " future prediction"},
            inplace=True,
        )
        self.input_and_future_df = pd.concat(
            [self.input_df, self.predictions_future], axis=0
        )
        if return_data_only:
            return self.input_and_future_df
        return self.input_and_future_df.plot(
            x="date",
            y=[self.state_name, self.state_name + " future prediction"],
            ylabel="daily cases",
            ax=ax,
        )
