import pandas as pd


class CovidRawDataManager:
    """
    Class for managing raw data cor COVID cases.
    Will hold both country vs date and state vs date
    COVID case data.

    Notes
    -----
    @skim0119 feel free to refactor

    """

    covid_us_df: pd.DataFrame
    covid_us_states_df: pd.DataFrame

    EXCLUDE_STATES = [
        "Alaska",
        "Hawaii",
        "Guam",
        "Puerto Rico",
        "Virgin Islands",
        "Northern Mariana Islands",
        "American Samoa",
    ]

    def __init__(self):
        self.load_data_from_url()
        self.drop_na_values()

    def load_data_from_url(self):
        """Loads dataframes from online csv"""
        repo_url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/"
        self.covid_us_df = pd.read_csv(repo_url + "us.csv")
        self.covid_us_states_df = pd.read_csv(repo_url + "us-states.csv")

    def drop_na_values(self):
        """Remove missing values"""
        self.covid_us_df.dropna(inplace=True)
        self.covid_us_states_df.dropna(inplace=True)

    def generate_statewise_history_data(self, states_df_label):
        """Generates a table with columns as states and rows as dates for given label"""
        if states_df_label not in ["cases", "deaths"]:
            raise ValueError("Invalid states data label given")
        table = pd.pivot_table(
            self.covid_us_states_df,
            index="date",
            columns="state",
            values=states_df_label,
            fill_value=0.0,
        )
        table.index = pd.to_datetime(table.index)

        table = table.drop(self.EXCLUDE_STATES, axis=1)
        return table

    def generate_daily_statewise_data_from_history(
        self, statewise_history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generates daily statewise data from cumulative history data"""
        daily_statewise_df = statewise_history_df.diff()
        # filter false trends in data (where < 0)
        daily_statewise_df = daily_statewise_df[daily_statewise_df >= 0]
        daily_statewise_df.dropna(inplace=True)
        return daily_statewise_df
