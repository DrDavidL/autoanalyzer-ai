import numpy as np

import pandas as pd
from scipy import stats
import io
import missingno as msno
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.imputation import mice
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from langchain_openai import ChatOpenAI, AzureChatOpenAI, AzureOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import json
import base64
from PIL import Image
from lifelines import KaplanMeierFitter, CoxPHFitter
from explanations.explanations import (
    shapley_explanation,
    mult_linear_reg_explanation,
    cox,
    kaplan_meier,
)
import openai
from tableone import TableOne
from random import randint
import os
from sklearn import linear_model
import statsmodels.api as sm
import category_encoders as ce
import shap
import time
import tempfile
from prompts import (
    csv_prefix_gpt4,
    data_analysis_prompt,
    plot_generation_prompt,
    quick_analysis_prompt,
    tool_explanations,
)
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any
# Import functions directly from markdown_to_docx.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# We'll use the html2docx library directly instead
import html2docx
import docx
import markdown as md
import tempfile
from bs4 import BeautifulSoup
import sweetviz as sv
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
# Import importlib.resources to replace pkg_resources
import importlib.resources
import importlib.metadata
import warnings # Import the warnings module

# Suppress specific DeprecationWarnings from seaborn
warnings.filterwarnings(
    "ignore",
    message="is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, pd.CategoricalDtype) instead",
    category=DeprecationWarning,
    module="seaborn._oldcore"
)


st.set_page_config(
    page_title="AutoAnalyzer",
    layout="centered",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="auto",
)

password_key = os.environ.get("PASSWORD")
openai_api_key = os.environ.get("OPENAI_API_KEY")
hu_key = os.environ.get("HEALTH_UNIVERSE")
openai_base_url = os.environ.get("OPENAI_BASE_URL")

if password_key is None:
    password_key = st.secrets["password"]
    openai_api_key = st.secrets["azure-openai-api-key"]
    hu_key = st.secrets["health-universe"]
    openai_base_url = st.secrets["openai-base-url"]

if "full_gpt_response" not in st.session_state:
    st.session_state.full_gpt_response = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "modified_df" not in st.session_state:
    st.session_state.modified_df = pd.DataFrame()
if "gen_csv" not in st.session_state:
    st.session_state.gen_csv = None
if "df_to_download" not in st.session_state:
    st.session_state.df_to_download = None


@st.cache_resource
def make_sweet_report(df):
    # Suppress deprecation warnings from sweetviz's internal use of pkg_resources
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return sv.analyze(df)


@st.cache_resource
def make_pandas_report(df, title):
    # Suppress deprecation warnings from ydata-profiling's internal use of pkg_resources
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return ProfileReport(df, title=title)


def get_output_path():
    tmpdirname = tempfile.mkdtemp(prefix="output_")
    return tmpdirname


from markdown_to_docx import generate_gpt_analysis_docx


if "outputs_path" not in st.session_state:
    st.session_state.outputs_path = get_output_path()


# Removed pandasai Agent-based function, as only langchain-experimental is now used.


def is_valid_api_key(api_key):
    openai.api_key = api_key
    try:
        # Send a test request to the OpenAI API
        openai.Completion.create(
            model="text-davinci-003", prompt="Hello world"
        )["choices"][0]["text"]
        return True
    except Exception:
        return False


def is_bytes_like(obj):
    return isinstance(obj, (bytes, bytearray, memoryview))


def save_image(plot, filename):
    try:
        if is_bytes_like(plot):
            img = io.BytesIO(plot)
        else:
            img = io.BytesIO()
            plot.savefig(img, format="png")
        img.seek(0)
        btn = st.download_button(
            label="Download your plot.",
            data=img,
            file_name=filename,
            mime="image/png",
        )
    except Exception as e:
        st.warning(f"Could not save or download plot: {e}")


def generate_regression_equation(intercept, coef, x_col):
    equation = f"y = {round(intercept, 4)}"

    for c, feature in zip(coef, x_col):
        equation += f" + {round(c, 4)} * {feature}"

    return equation


def df_download_options(df, report_type):
    a = randint(0, 10000000000)
    format = st.radio(
        "Select the format for your report:",
        (
            "csv",
            "json",
            "html",
        ),
        key=f"report_format_{a}",
        horizontal=True,
    )
    file_name = f"{report_type}.{format}"

    if format == "csv":
        data = df.to_csv(index=True)
        mime = "text/csv"
    if format == "json":
        data = df.to_json(orient="records")
        mime = "application/json"
    if format == "html":
        data = df.to_html()
        mime = "text/html"
    if True:
        st.download_button(
            label="Download your report.",
            data=data,
            # data=df.to_csv(index=True),
            file_name=file_name,
            mime=mime,
        )


def plot_mult_linear_reg(df, x, y):
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    # st.write('Intercept: \n', regr.intercept_)
    # st.write('Coefficients: \n', regr.coef_)

    # with statsmodels
    x = sm.add_constant(x)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    print_model = model.summary2()
    st.write(print_model)
    try:
        df_mlr_output = print_model.tables[1]
    except:
        st.write("couldn't generate dataframe version")
    return print_model, df_mlr_output, regr.intercept_, regr.coef_


def all_categorical(df):
    categ_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [
        col
        for col in df.columns
        if df[col].nunique() == 2 and df[col].dtype != "object"
    ]
    filtered_categorical_cols = [col for col in categ_cols if df[col].nunique() <= 15]
    all_categ = filtered_categorical_cols + numeric_cols
    return all_categ


def all_numerical(df):
    numerical_cols = df.select_dtypes(include="number").columns.tolist()

    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() == 2:
            unique_values = df[col].unique()
            if 0 in unique_values and 1 in unique_values:
                continue

            value_counts = df[col].value_counts()
            most_frequent_value = value_counts.idxmax()
            least_frequent_value = value_counts.idxmin()

            if most_frequent_value != 0 and least_frequent_value != 1:
                df[col] = np.where(df[col] == most_frequent_value, 0, 1)
                st.write(
                    f"Replaced most frequent value '{most_frequent_value}' with 0 and least frequent value '{least_frequent_value}' with 1 in column '{col}'."
                )
                numerical_cols.append(col)  # Update numerical_cols

    return numerical_cols


def filter_dataframe(df):
    # Get the column names and data types of the dataframe
    columns = df.columns
    dtypes = df.dtypes

    # Create a sidebar for selecting columns to exclude
    excluded_columns = st.multiselect("Exclude Columns", columns)

    # Create a copy of the dataframe to apply the filters
    filtered_df = df.copy()

    # Exclude the selected columns from the dataframe
    filtered_df = filtered_df.drop(excluded_columns, axis=1)

    # Get the column names and data types of the filtered dataframe
    filtered_columns = filtered_df.columns
    filtered_dtypes = filtered_df.dtypes

    # Create a sidebar for selecting numerical variables and their range
    numerical_columns = [
        col
        for col, dtype in zip(filtered_columns, filtered_dtypes)
        if dtype in ["int64", "float64"]
    ]
    for col in numerical_columns:
        min_val = filtered_df[col].min()
        max_val = filtered_df[col].max()
        st.write(f"**{col}**")
        min_range, max_range = st.slider(
            "", min_val, max_val, (min_val, max_val), key=col
        )

        # Filter the dataframe based on the selected range
        if min_range > min_val or max_range < max_val:
            filtered_df = filtered_df[
                (filtered_df[col] >= min_range) & (filtered_df[col] <= max_range)
            ]

    # Create a sidebar for selecting categorical variables and their values
    categorical_columns = [
        col
        for col, dtype in zip(filtered_columns, filtered_dtypes)
        if dtype == "object"
    ]
    for col in categorical_columns:
        unique_values = filtered_df[col].unique()
        selected_values = st.multiselect(col, unique_values, unique_values)

        # Filter the dataframe based on the selected values
        if len(selected_values) < len(unique_values):
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    return filtered_df


# Function to generate a download link
def get_download_link(file_path, file_type):
    try:
        with open(file_path, "rb") as file:
            contents = file.read()
        base64_data = base64.b64encode(contents).decode("utf-8")
        download_link = f'<a href="data:application/octet-stream;base64,{base64_data}" download="tableone_results.{file_type}">Click here to download the TableOne results in {file_type} format.</a>'
        return download_link
    except Exception as e:
        st.warning(f"Could not create download link: {e}")
        return ""


def find_binary_categorical_variables(df):
    binary_categorical_vars = []
    for col in df.columns:
        unique_values = df[col].unique()
        if len(unique_values) == 2:
            binary_categorical_vars.append(col)
    return binary_categorical_vars


def calculate_odds_older(table):
    odds_cases = table.iloc[1, 1] / table.iloc[1, 0]
    odds_controls = table.iloc[0, 1] / table.iloc[0, 0]
    odds_ratio = odds_cases / odds_controls
    return odds_cases, odds_controls, odds_ratio


def calculate_odds(table):
    odds_cases = table.iloc[1, 1] / table.iloc[1, 0]
    odds_controls = table.iloc[0, 1] / table.iloc[0, 0]
    odds_ratio = odds_cases / odds_controls
    return odds_cases, odds_controls, odds_ratio


def generate_2x2_table(df, var1, var2):
    table = pd.crosstab(df[var1], df[var2], margins=True)
    table.columns = ["No " + var2, "Yes " + var2, "Total"]
    table.index = ["No " + var1, "Yes " + var1, "Total"]
    return table


def plot_survival_curve(df, time_col, event_col):
    # Create a Kaplan-Meier fitter object
    try:
        if time_col not in df.columns or event_col not in df.columns:
            st.warning("Selected columns not found in dataframe.")
            return None
        if df[time_col].isnull().any() or df[event_col].isnull().any():
            st.warning("Time or event column contains missing values.")
            return None
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], event_observed=df[event_col])
        fig, ax = plt.subplots()
        kmf.plot_survival_function(ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Survival Curve")
        st.pyplot(fig)
        return fig
    except Exception as e:
        st.warning(f"Could not plot survival curve: {e}")
        return None


def calculate_rr_arr_nnt(tn, fp, fn, tp):
    rr = (tp / (tp + fn)) / (fp / (fp + tn)) if fp + tn > 0 and tp + fn > 0 else np.inf
    arr = (fn / (fn + tp)) - (fp / (fp + tn)) if fn + tp > 0 and fp + tn > 0 else np.inf
    nnt = 1 / arr if arr > 0 else np.inf
    return rr, arr, nnt


# def fetch_api_key():
#     api_key = None

#     try:
#         # Attempt to retrieve the API key as a secret
#         api_key = st.secrets["openai-api-key"]
#         # os.environ["openai-api-key"] = api_key
#         st.session_state.openai-api-key = api_key
#         os.environ['openai-api-key'] = api_key
#         # st.write(f'Here is what we think the key is step 1: {api_key}')
#     except KeyError:

#         if st.session_state.openai-api-key != '':
#             api_key = st.session_state.openai-api-key
#             os.environ['openai-api-key'] = api_key
#             # If the API key is already set, don't prompt for it again
#             # st.write(f'Here is what we think the key is step 2: {api_key}')
#             return
#         else:
#             # If the secret is not found, prompt the user for their API key
#             st.sidebar.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
#             api_key = st.sidebar.text_input("Please, whisper your API key into my ears: ",)

#             st.session_state.openai-api-key = api_key
#             os.environ['openai-api-key'] = api_key
#             # Save the API key as a secret
#             # st.secrets["my_api_key"] = api_key
#             # st.write(f'Here is what we think the key is step 3: {api_key}')
#             return

#     return


def check_password() -> bool:
    """
    Check if the entered password is correct and manage login state.
    Also resets the app when a user successfully logs in.
    """
    # Early return if st.secrets["docker"] == "docker"
    # if st.secrets["docker"] == "docker":
    #     st.session_state.password_correct = True
    #     return True
    # Initialize session state variables
    if "password" not in st.session_state:
        st.session_state.password = ""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    def password_entered() -> None:
        """Callback function when password is entered."""
        if st.session_state["password"] == password_key:
            st.session_state["password_correct"] = True
            st.session_state.login_attempts = 0
            # Reset the app
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            st.session_state.login_attempts += 1

    # Check if password is correct
    if not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )

        if st.session_state.login_attempts > 0:
            st.error(
                f"😕 Password incorrect. Attempts: {st.session_state.login_attempts}"
            )

        st.write(
            "*Please contact David Liebovitz, MD if you need an updated password for access.*"
        )
        return False

    return True


csv_prefix = """You are an agent optimally designed for answering questions about a dataframe. 
If anwering a query requires drawing a table, chart, or generating any other figure, never attempt 
to draw the figure. Instead, return the Python code as a string. Do not return JSON. The following are already imported so 
do not include any import statements in your code:
- plotly.figure_factory as ff
- matplotlib.pyplot as plt
- seaborn as sns

Please format the string response (not JSON) such that it includes:

1. Code to interpret the user's question and select the appropriate visualization.
2. Code to generate the visualization using plotly as ff, matplotlib.pyplot as plt, or seaborn as sns. 
3. Do not return JSON. The response should include the Python code as a string.

Remember to structure the code such that it is properly indented and formatted according to PEP8 guidelines.

            """


def assess_data_readiness(df):
    readiness_summary = {}
    st.write("White horizontal lines (if present) show missing data")
    try:
        missing_matrix = msno.matrix(df)
        # missing_matrix = visualimiss.matrix(df, color=(43, 102, 189), sort="asc")
        # missing_matrix = msno.matrix(df)
        st.pyplot(missing_matrix.figure)

    except:
        st.warning(
            'Dataframe not yet amenable to missing for "missingno" library analysis.'
        )

    # Check if the DataFrame is empty

    try:
        if df.empty:
            readiness_summary["data_empty"] = True
            readiness_summary["columns"] = {}
            readiness_summary["missing_columns"] = []
            readiness_summary["inconsistent_data_types"] = []
            readiness_summary["missing_values"] = {}
            readiness_summary["data_ready"] = False
            return readiness_summary
    except:
        st.warning("Dataframe not yet amenable to empty analysis.")

    try:
        columns = {col: str(df[col].dtype) for col in df.columns}
        readiness_summary["columns"] = columns
    except:
        st.warning("Dataframe not yet amenable to column analysis.")

    try:
        missing_columns = df.columns[df.isnull().all()].tolist()
        readiness_summary["missing_columns"] = missing_columns
    except:
        st.warning("Dataframe not yet amenable to missing column analysis.")

    try:
        inconsistent_data_types = []
        for col in df.columns:
            unique_data_types = df[col].apply(type).drop_duplicates().tolist()
            if len(unique_data_types) > 1:
                inconsistent_data_types.append(col)
        readiness_summary["inconsistent_data_types"] = inconsistent_data_types

    except:
        st.warning("Dataframe not yet amenable to data type analysis.")

    try:
        missing_values = df.isnull().sum().to_dict()
        readiness_summary["missing_values"] = missing_values
    except:
        st.warning("Dataframe not yet amenable to specific missing value analysis.")

    try:
        readiness_summary["data_empty"] = False
        if missing_columns or inconsistent_data_types or any(missing_values.values()):
            readiness_summary["data_ready"] = False
        else:
            readiness_summary["data_ready"] = True

        return readiness_summary
    except:
        st.warning("Dataframe not yet amenable to overall data readiness analysis.")


def process_model_output(output):
    # Check if the output is already a dictionary
    if isinstance(output, dict):
        processed_output = output
    else:
        # Remove any leading/trailing whitespace
        output = output.strip()

        # Check if the output is wrapped in ```json ... ``` markers
        if output.startswith("```json") and output.endswith("```"):
            # Remove the markers
            json_str = output[7:-3].strip()
        else:
            # Assume the entire output is JSON
            json_str = output

        try:
            processed_output = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string - {str(e)}")
            return None

    # Ensure the processed output has the expected structure
    if (
        not isinstance(processed_output, dict)
        or "code_snippets" not in processed_output
    ):
        print("Error: Unexpected output structure")
        return None

    # Process each code snippet
    for snippet in processed_output["code_snippets"]:
        if "code" in snippet:
            snippet["code"] = snippet["code"].strip()

    return processed_output


# Removed unused process_model_output_old function


def safety_check(code):
    # Basic check for dangerous keywords in code
    dangerous_keywords = [
        " exec", " eval", " open", " sys", " subprocess", " del", " delete", " remove", " os",
        " shutil", " pip", " conda", " st.write", " exit", " quit", " globals", " locals", " dir",
        " reload", " lambda", " setattr", " getattr", " delattr", " yield", " assert", " break",
        " continue", " raise", " try", "compile", "__import__",
    ]
    for keyword in dangerous_keywords:
        if keyword in code:
            return False, "Concerning code detected."
    return True, "Safe to execute."


def replace_show_with_save(code_string, filename="output.png"):
    # Replace plt.show() and fig.show() with save commands
    save_cmd1 = f"plt.savefig('{st.session_state.outputs_path}/{filename}')"
    save_cmd2 = f"pio.write_image(fig, '{st.session_state.outputs_path}/{filename}')"
    code_string = code_string.replace("plt.show()", save_cmd1)
    code_string = code_string.replace("fig.show()", save_cmd2)
    return code_string


@st.cache_data
def start_chatbot2(df, question, max_retries=5, delay=2):
    llm = AzureChatOpenAI(
        endpoint=openai_base_url,
        api_key=openai_api_key,
        model=st.secrets["azure_deployment"],
        temperature=0.3,
    )
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        max_iterations=10,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    question_updated = f"""Without invoking plots, and through step by step analysis of the dataframe repeated as needed, accurately answer the user question 
    anticipating what the user really wants to know. Ensure your terminal outputs show all columns so no data is missing from analysis. Unless specifically 
    requested to limit rows or filter criteria, always include all rows in the analysis. To ensure all analysis output columns are included in your analysis, 
    use pandas commands to show all output columns. With correct formatting use the following code snippet:
        # Adjust display options
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being split across lines
    Academic careers are at risk if there is a mistake in your analysis: {question}"""

    response = None
    for attempt in range(max_retries):
        try:
            response = agent.invoke(question_updated)
            break  # Exit the loop if the call is successful
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(
                    f"OpenAI servers are busy. Retrying {attempt + 1}/{max_retries}."
                )
                time.sleep(delay)  # Wait for a specified delay before retrying
            else:
                st.error("OpenAI servers remain busy. Please try again in 5 min.")
                raise e  # Optionally, re-raise the exception if needed
    return response


def start_chatbot3(df, model):
    # fetch_api_key()
    # openai.api_key = st.session_state.openai-api-key
    agent = create_pandas_dataframe_agent(
        # ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        ChatOpenAI(api_key=openai_api_key, temperature=0, model=model),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
        st.session_state["messages_df"] = []

    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                  
        """)

    #     # Check if the API key exists as an environmental variable
    # api_key = os.environ.get("openai-api-key")

    # if api_key:
    #     # st.write("*API key active - ready to respond!*")
    #     pass
    # else:
    #     st.warning("API key not found as an environmental variable.")
    #     api_key = st.text_input("Enter your OpenAI API key:")

    #     if st.button("Save"):
    #         if is_valid_api_key(api_key):
    #             os.environ["openai-api-key"] = api_key
    #             st.success("API key saved as an environmental variable!")
    #         else:
    #             st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_input(
        "Your question, e.g., 'Create a scatterplot for age and BMI.' *This option only generates plots.* ",
        "",
    )
    if st.button("Send"):
        try:
            st.session_state.messages_df.append(
                {"role": "user", "content": csv_question}
            )
            csv_input = csv_prefix + csv_question
            output = agent.run(csv_input)
            # st.write(output)
            code_string = process_model_output(str(output))
            # st.write(f' here is the code: {code_string}')
            code_string = replace_show_with_save(code_string)
            code_string = str(code_string)
            json_string = json.dumps(code_string)
            decoded_string = json.loads(json_string)
            with st.expander("What is the code?"):
                st.write(
                    "Here is the custom code for your request and the image below:"
                )
                st.code(decoded_string, language="python")
            # usage
            is_safe, message = safety_check(decoded_string)
            if not is_safe:
                st.write("Code safety concern. Try again.", message)
            if is_safe:
                try:
                    exec(decoded_string)
                    image = Image.open(f"{st.session_state.outputs_path}/output.png")
                    st.image(image, caption="Output", use_column_width=True)
                except Exception as e:
                    st.write("Error - we noted this was fragile! Try again.", e)
        except Exception:
            st.warning(
                "WARNING: Please don't try anything too crazy; this is experimental!"
            )
            # sys.exit(1)
            # return None, None


def start_plot_gpt4(df, question, max_retries=5, delay=3):
    llm = AzureChatOpenAI(
        azure_deployment=st.secrets["azure_deployment"],
        api_version=st.secrets["api_version"],
        # api_version= "2024-05-01-preview",
        azure_endpoint=openai_base_url,
        api_key=openai_api_key,
        temperature=0.3,
        max_tokens=4000,
        timeout=None,
        max_retries=2,
        model_kwargs={
            "seed": 42,
        },
    )
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        max_iterations=10,
        agent_type="tool-calling",
        verbose=True,
        return_intermediate_steps=True,
        number_of_head_rows=-1,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    # st.write(f'Question: {question_updated}')

    model_output = None
    for attempt in range(max_retries):
        try:
            model_output = agent.invoke(question)
            break  # Exit the loop if the call is successful
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(
                    f"OpenAI servers are busy. Retrying {attempt + 1}/{max_retries}."
                )
                time.sleep(delay)  # Wait for a specified delay before retrying
            else:
                st.error("OpenAI servers remain busy. Please try again in 5 min.")
                raise e  # Optionally, re-raise the exception if needed
        # model_output = agent.run(csv_input)
        # Display raw output
    return model_output


class StreamlitAsyncCallbackHandler(AsyncCallbackHandler):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.token_count = 0
        self.total_tokens = 100  # Estimate, adjust as needed

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_count += 1
        progress = min(self.token_count / self.total_tokens, 1.0)
        self.progress_bar.progress(progress)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.progress_bar.progress(1.0)


async def start_plot_gpt4_async(df, question, progress_bar, max_retries=5, delay=2):
    callback_handler = StreamlitAsyncCallbackHandler(progress_bar)

    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4",
        temperature=0.3,
        streaming=True,
        callbacks=[callback_handler],
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        max_iterations=10,
        agent_type="tool-calling",
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    for attempt in range(max_retries):
        try:
            model_output = await agent.ainvoke(question)
            return model_output
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(
                    f"OpenAI servers are busy. Retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(delay)
            else:
                st.error("OpenAI servers remain busy. Please try again in 5 min.")
                raise e


async def run_analysis(df, question1, question2):
    progress_bar1 = st.progress(0)
    progress_bar2 = st.progress(0)

    task1 = asyncio.create_task(start_plot_gpt4_async(df, question1, progress_bar1))
    task2 = asyncio.create_task(start_plot_gpt4_async(df, question2, progress_bar2))

    result1, result2 = await asyncio.gather(task1, task2)

    return result1, result2


def start_plot_gpt4_old2(df):
    # fetch_api_key()
    # openai.api_key = st.session_state.openai-api-key
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4o"),
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
        st.session_state["messages_df"] = []

    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                
        """)

    #     # Check if the API key exists as an environmental variable
    # api_key = os.environ.get("openai-api-key")

    # if api_key:
    #     # st.write("*API key active - ready to respond!*")
    #     pass
    # else:
    #     st.warning("API key not found as an environmental variable.")
    #     api_key = st.text_input("Enter your OpenAI API key:")

    #     if st.button("Save"):
    #         if is_valid_api_key(api_key):
    #             os.environ["openai-api-key"] = api_key
    #             st.success("API key saved as an environmental variable!")
    #         else:
    #             st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_area(
        "Your question, e.g., 'Create a heatmap. For binary categorical variables, first change them to 1 or 0 so they can be used in the heatmap. Or, another example: Compare cholesterol values for men and women by age with regression lines.",
        "",
    )
    if st.button("Send"):
        try:
            st.session_state.messages_df.append(
                {"role": "user", "content": csv_question}
            )
            csv_input = csv_prefix_gpt4 + csv_question
            model_output = agent.run(csv_input)
            # Display raw output
            st.subheader("Raw Output:")
            st.code(model_output, language="json")

            # Process the model output
            processed_output = process_model_output(model_output)

            if processed_output:
                st.subheader("Processed Output:")
                st.json(processed_output)

                # Display text response
                st.subheader("Text Response:")
                st.write(processed_output["text_response"])

                # Execute and display each code snippet
                for i, snippet in enumerate(processed_output["code_snippets"], 1):
                    st.subheader(f"Plot {i}: {snippet['description']}")

                    # Display the code
                    st.code(snippet["code"], language="python")

                    # Execute the code
                    try:
                        exec(snippet["code"])
                    except Exception as e:
                        st.error(f"Error executing code: {str(e)}")

            else:
                st.error("Failed to process the model output.")
            # st.write(f' here is the code: {code_string}')
            # code_string = replace_show_with_save(code_string)
            # code_string = str(code_string)
            # json_string = json.dumps(code_string)
            # decoded_string = json.loads(json_string)
            # with st.expander("What is the code?"):
            #     st.write('Here is the custom code for your request and the image below:')
            #     st.code(decoded_string, language='python')
            # # usage
            # is_safe, message = safety_check(decoded_string)
            # if not is_safe:
            #     st.write("Code safety concern. Try again.", message)
            # if is_safe:
            #     try:
            #         exec(decoded_string)
            #         image = Image.open(f'./{st.session_state.outputs_path}/output.png')
            #         st.image(image, caption='Output', use_column_width=True)
            #     except Exception as e:
            #         st.write('Error - we noted this was fragile! Try again.', e)
        except Exception:
            st.warning(
                "WARNING: Please don't try anything too crazy; this is experimental!"
            )
            # sys.exit(1)
            # return None, None


def generate_df(columns, n_rows, selected_model):
    """
    Generates a synthetic dataframe based on column names and number of rows using an LLM.
    """
    from prompts import dataframe_generation_system_prompt
    system_prompt = dataframe_generation_system_prompt

    user_prompt = f"columns : {columns}, number : {n_rows}"

    # Instantiate the LLM using the same configuration as the GPT analysis tab
    llm = AzureChatOpenAI(
        azure_deployment=st.secrets["azure_deployment"],
        api_version=st.secrets["api_version"],
        azure_endpoint=openai_base_url,
        api_key=openai_api_key,
        temperature=0.5, # Use a slightly higher temperature for data generation creativity
        max_tokens=4000,
        timeout=None,
        max_retries=2,
        model_kwargs={
            "seed": 42,
        },
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Invoke the LLM
        response = llm.invoke(messages)
        
        # Extract the content from the response
        generated_content = response.content if hasattr(response, "content") else str(response)

        # Use StringIO to convert the string data into file-like object
        data = io.StringIO(generated_content)

        # Read the data into a DataFrame, skipping the first row (assuming the LLM output includes headers)
        # If the LLM output does not include headers, remove skiprows=1 and names=columns
        try:
            # Attempt to read assuming headers are present and match requested columns
            df = pd.read_csv(data, sep=",")
            # Basic check if columns match, otherwise try reading without headers
            if not all(col in df.columns for col in columns):
                 data.seek(0) # Reset StringIO position
                 df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)

        except Exception as e:
            st.warning(f"Failed to parse generated data: {e}")
            return pd.DataFrame(), None

        # Convert DataFrame to CSV and create download link
        gen_csv = df.to_csv(index=False)

        return df, gen_csv

    except Exception as e:
        st.warning(
            f"WARNING: Could not generate data. Please double check your proposed column names for duplicates or invalid characters, or try again later. Error: {e}"
        )
        return pd.DataFrame(), None


def generate_table(df, categorical_variable, nonnormal_variables):
    # Generate the table using TableOne
    mytable = TableOne(
        df,
        columns=df.columns.tolist(),
        categorical=categorical,
        groupby=categorical_variable,
        nonnormal=nonnormal_variables,
        pval=True,
    )
    return mytable


def preprocess_for_pca(df):
    included_cols = []
    excluded_cols = []
    binary_mapping = {}  # initialize empty dict for binary mapping
    binary_encoded_vars = []  # initialize empty list for binary encoded vars

    # Create a binary encoder
    bin_encoder = ce.BinaryEncoder()

    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == "object":
            unique = df[col].nunique()

            # For binary categorical columns
            if unique == 2:
                most_freq = df[col].value_counts().idxmax()
                least_freq = df[col].value_counts().idxmin()
                df[col] = df[col].map({most_freq: 0, least_freq: 1})
                binary_mapping[col] = {
                    most_freq: 0,
                    least_freq: 1,
                }  # add mapping to dict
                included_cols.append(col)

            # For categorical columns with less than 15 unique values
            elif 2 < unique <= 15:
                try:
                    # Perform binary encoding
                    df_transformed = bin_encoder.fit_transform(df[col])
                    # Drop the original column from df
                    df.drop(columns=[col], inplace=True)
                    # Join the transformed data to df
                    df = pd.concat([df, df_transformed], axis=1)
                    # Add transformed columns to binary encoded vars list and included_cols
                    transformed_cols = df_transformed.columns.tolist()
                    binary_encoded_vars.extend(transformed_cols)
                    included_cols.extend(transformed_cols)
                except Exception as e:
                    st.write(f"Failure in encoding {col} due to {str(e)}")
                    excluded_cols.append(col)
            else:
                excluded_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            included_cols.append(col)
        else:
            excluded_cols.append(col)

    # Display binary mappings and binary encoded variables in streamlit
    if binary_mapping:
        st.write("Binary Mappings: ", binary_mapping)
    if binary_encoded_vars:
        st.write("Binary Encoded Variables: ", binary_encoded_vars)

    return df[included_cols], included_cols, excluded_cols


def create_scree_plot(df):
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)

    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)

    # Create a PCA instance: n_components should be None so variance is preserved from all initial features
    pca = PCA(n_components=None)
    pca.fit_transform(x)

    # Scree plot
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(pca.explained_variance_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
    )
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    st.pyplot(fig)
    return fig


def perform_pca_plot(df):
    st.write(
        "Note: For this PCA analysis, categorical columns with 2 values are mapped to 1 and 0. Categories with more than 2 values have been binary encoded."
    )
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)

    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)

    # Select the target column for PCA
    cols_2_15_unique_vals = [
        col for col in included_cols if 2 <= df[col].nunique() <= 15
    ]
    target_col_pca = st.selectbox(
        "Select the target column for PCA", cols_2_15_unique_vals
    )

    num_unique_targets = df[
        target_col_pca
    ].nunique()  # Calculate the number of unique targets

    # Ask the user to request either 2 or 3 component PCA
    n_components = st.selectbox("Select the number of PCA components (2 or 3)", [2, 3])

    # Create a PCA instance
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)

    # Depending on user choice, plot the appropriate PCA
    if n_components == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
    else:
        principalDf = pd.DataFrame(
            data=principalComponents, columns=["PC1", "PC2", "PC3"]
        )

    finalDf = pd.concat([principalDf, df[[target_col_pca]]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    if n_components == 2:
        ax = fig.add_subplot(111)
    else:
        # ax = Axes3D(fig)
        ax = plt.axes(projection="3d")
        ax.set_zlabel("Principal Component 3", fontsize=15)

    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)

    ax.set_title(f"{n_components} component PCA", fontsize=20)

    targets = finalDf[target_col_pca].unique().tolist()
    colors = sns.color_palette("husl", n_colors=num_unique_targets)
    # finalDf

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[target_col_pca] == target
        if n_components == 2:
            ax.scatter(
                finalDf.loc[indicesToKeep, "PC1"],
                finalDf.loc[indicesToKeep, "PC2"],
                c=[color],
                s=50,
            )
        else:
            ax.scatter(
                finalDf.loc[indicesToKeep, "PC1"],
                finalDf.loc[indicesToKeep, "PC2"],
                finalDf.loc[indicesToKeep, "PC3"],
                c=[color],
                s=50,
            )

    ax.legend(targets)

    # Make a scree plot

    # Display the plot using Streamlit
    st.pyplot(fig)
    st.subheader("Use the PCA Updated Dataset for Machine Learning")
    st.write(
        "Download the current plot if you'd like to save it! Then, follow steps to apply machine learning to your PCA modified dataset."
    )
    st.info(
        "Step 1. Click Button to use the PCA Dataset for ML. Step 2. Select Modified Dataframe on left sidebar and switch to the Machine Learning tab. (You'll overfit if you click below again!)"
    )
    if st.button("Use PCA Updated dataset on Machine Learning Tab"):
        st.session_state.modified_df = finalDf

    return fig


def display_metrics(y_true, y_pred, y_scores, set_name="Test"):
    # Check if this is a regression model (continuous outputs) or classification model
    is_regression = False
    try:
        # Try to compute classification metrics
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Display classification metrics
        st.info(
            f"**Your Model Metrics ({set_name} Set):** F1 score: {f1:.2f}, Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}"
        )
    except ValueError as e:
        # If we get a ValueError, it's likely because we're using a regression model
        is_regression = True
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Compute regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Display regression metrics
        st.info(
            f"**Your Model Metrics ({set_name} Set):** RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}"
        )
    
    with st.expander("Explanations for the Metrics"):
        if is_regression:
            st.write(
                """
### Explanation of Regression Metrics
- **RMSE** (Root Mean Squared Error) measures the average magnitude of the errors. It gives higher weight to larger errors.
- **MAE** (Mean Absolute Error) measures the average magnitude of the errors without considering their direction.
- **R²** (R-squared) represents the proportion of variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better fit.
"""
            )
        else:
            st.write(
                """
### Explanation of Classification Metrics
- **F1 score** is the harmonic mean of precision and recall, and it tries to balance the two. It is a good metric when you have imbalanced classes.
- **Accuracy** is the ratio of correct predictions to the total number of predictions. It can be misleading if the classes are imbalanced.
- **ROC AUC** (Receiver Operating Characteristic Area Under Curve) represents the likelihood of the classifier distinguishing between a positive sample and a negative sample. It's equal to 0.5 for random predictions and 1.0 for perfect predictions.
- **PR AUC** (Precision-Recall Area Under Curve) is another way of summarizing the trade-off between precision and recall, and it gives more weight to precision. It's useful when the classes are imbalanced.
"""
            )
    # Confusion matrix, ROC, and PR curves are now shown in the main ML tab for clarity.
    # ROC curve is now shown only once per set, outside this function.
    
    return is_regression


def plot_pr_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower right")
    return fig


def get_categorical_and_numerical_cols(df):
    # Initialize empty lists for categorical and numerical columns
    categorical_cols = []
    numeric_cols = []

    # Go through each column in the dataframe
    for col in df.columns:
        # If the column data type is numerical and has more than two unique values, add it to the numeric list
        if np.issubdtype(df[col].dtype, np.number) and len(df[col].unique()) > 2:
            numeric_cols.append(col)
        # Otherwise, add it to the categorical list
        else:
            categorical_cols.append(col)

    # Sort the lists
    numeric_cols.sort()
    categorical_cols.sort()

    return numeric_cols, categorical_cols


# Removed unused plot_confusion_matrix_old function


def plot_confusion_matrix(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the ConfusionMatrixDisplay object
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Class 0", "Class 1"]
    )

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(dpi=100)

    # Plot the confusion matrix using the `plot` method
    cmd.plot(ax=ax, cmap="Blues", values_format="d")

    # Customize the plot if needed
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")

    return fig


def preprocess(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == "object":
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()

                    # Update the mapping to include 'F' as 0 and 'M' as 1
                    df[col] = df[col].map({most_freq: 0, least_freq: 1, "F": 0})

                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")

                included_cols.append(col)

    return df[included_cols], included_cols, excluded_cols


def preprocess_old(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == "object":
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()
                    df[col] = df[col].map({most_freq: 0, least_freq: 1})
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")
                included_cols.append(col)

    # st.write(f"Included Columns: {included_cols}")
    # st.write(f"Excluded Columns: {excluded_cols}")

    return df[included_cols], included_cols, excluded_cols


def create_boxplot(df, numeric_col, categorical_col, show_points=False):
    try:
        if numeric_col and categorical_col:
            fig, ax = plt.subplots()

            # Plot the notched box plot
            sns.boxplot(x=categorical_col, y=numeric_col, data=df, notch=True, ax=ax)

            if show_points:
                # Add the actual data points on the plot
                sns.swarmplot(x=categorical_col, y=numeric_col, data=df, color=".25", ax=ax)

            # Add a title to the plot
            ax.set_title(f"Box Plot of {numeric_col} by {categorical_col}")

            st.pyplot(fig)
            return fig
        else:
            st.warning("Please select both a numerical and a categorical column for the box plot.")
            return None
    except Exception as e:
        st.warning(f"Could not create box plot: {e}")
        return None


def create_violinplot(df, numeric_col, categorical_col):
    try:
        if numeric_col and categorical_col:
            fig, ax = plt.subplots()

            # Plot the violin plot
            sns.violinplot(x=categorical_col, y=numeric_col, data=df, ax=ax)

            # Add a title to the plot
            ax.set_title(f"Violin Plot of {numeric_col} by {categorical_col}")

            st.pyplot(fig)
            return fig
        else:
            st.warning("Please select both a numerical and a categorical column for the violin plot.")
            return None
    except Exception as e:
        st.warning(f"Could not create violin plot: {e}")
        return None


def create_scatterplot(df, scatter_x, scatter_y):
    try:
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()

            # Plot the scatter plot
            sns.regplot(x=scatter_x, y=scatter_y, data=df, ax=ax)

            # Calculate the slope and intercept of the regression line
            try:
                slope, intercept = np.polyfit(df[scatter_x], df[scatter_y], 1)
                # Add the slope and intercept as a text annotation on the plot
                ax.text(0.05, 0.95, f"y={slope:.2f}x+{intercept:.2f}", transform=ax.transAxes)
            except Exception:
                pass

            ax.set_title("Scatter Plot for " + scatter_y + " vs " + scatter_x)

            st.pyplot(fig)
            with st.expander("What is a scatter plot?"):
                st.write("""
A scatterplot is a type of plot that displays values for typically two variables for a set of data. It's used to visualize the relationship between two numerical variables, where one variable is on the x-axis and the other variable is on the y-axis. Each point on the plot represents an observation in your dataset.

**Which types of variables are appropriate for the x and y axes?**

Both the x and y axes of a scatterplot are typically numerical variables. For example, one might be "Patient Age" (on the x-axis) and the other might be "Blood Pressure" (on the y-axis). Each dot on the scatterplot then represents a patient's age and corresponding blood pressure. 

However, the variables used do not have to be numerical. They could be ordinal categories, such as stages of a disease, which have a meaningful order. 

The choice of which variable to place on each axis doesn't usually matter much for exploring relationships, but traditionally the independent variable (the one you control or think is influencing the other) is placed on the x-axis, and the dependent variable (the one you think is being influenced) is placed on the y-axis.

**What does a regression line mean when added to a scatterplot?**

A regression line (or line of best fit) is a straight line that best represents the data on a scatter plot. This line may pass through some of the points, none of the points, or all of the points. It's a way of modeling the relationship between the x and y variables. 

In the context of a scatterplot, the regression line is used to identify trends and patterns between the two variables. If the data points and the line are close, it suggests a strong correlation between the variables.

The slope of the regression line also tells you something important: for every unit increase in the variable on the x-axis, the variable on the y-axis changes by the amount of the slope. For example, if we have patient age on the x-axis and blood pressure on the y-axis, and the slope of the line is 2, it would suggest that for each year increase in age, we expect blood pressure to increase by 2 units, on average.

However, keep in mind that correlation does not imply causation. Just because two variables move together, it doesn't mean that one is causing the other to change.

For medical students, think of scatterplots as a way to visually inspect the correlation between two numerical variables. It's a way to quickly identify patterns, trends, and outliers, and to formulate hypotheses for further testing.""")
            return fig
        else:
            st.warning("Please select both x and y columns for the scatterplot.")
            return None
    except Exception as e:
        st.warning(f"Could not create scatterplot: {e}")
        return None


# Function to replace missing values


def replace_missing_values(df, method):
    # Differentiate numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if method == "drop":
        df = df.dropna()
    elif method == "zero":
        df[num_cols] = df[num_cols].fillna(0)
    elif method == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif method == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif method == "mode":
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    elif method == "mice":
        imp = mice.MICEData(df[num_cols])  # only apply to numerical columns
        df[num_cols] = imp.data
    st.session_state.df = df
    return df


# This function will be cached
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            st.warning("Loaded CSV is empty.")
        return data
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or not a valid CSV.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        st.error("The uploaded file could not be parsed. Please check the file format.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def analyze_dataframe(df):
    # Analyzing missing values
    missing_values = df.isnull().sum()

    # Analyzing outliers using the Z-score
    # (you might want to use a different method for identifying outliers)
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > 3).sum()

    # Analyzing data types
    data_types = df.dtypes

    # Analyzing skewness for numeric columns
    skewness = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew())

    # Analyzing cardinality in categorical columns
    cardinality = df.select_dtypes(include=["object", "category"]).nunique()

    return missing_values, outliers, data_types, skewness, cardinality


# Function to plot pie chart


def plot_pie(df, col_name):
    try:
        plt.figure(figsize=(10, 8))  # set the size of the plot
        df[col_name].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title(f"Distribution for {col_name}")
        return plt
    except Exception as e:
        st.warning(f"Could not plot pie chart for {col_name}: {e}")
        return None


# Function to summarize categorical data

import pandas as pd


def summarize_categorical(df):
    try:
        # Select only categorical columns
        cat_df = df.select_dtypes(include=["object", "category"])

        # If there are no categorical columns, return None
        if cat_df.empty:
            st.write("The DataFrame does not contain any categorical columns.")
            return None

        # Create a list to store dictionaries for each column's summary
        summary_data = []

        for col in cat_df.columns:
            # Number of unique values
            unique_count = df[col].nunique()

            # Most frequent category and its frequency
            try:
                most_frequent = df[col].mode()[0]
                freq_most_frequent = df[col].value_counts().iloc[0]
            except Exception:
                most_frequent = None
                freq_most_frequent = 0

            # Append the column summary as a dictionary to the list
            summary_data.append(
                {
                    "column": col,
                    "unique_count": unique_count,
                    "most_frequent": most_frequent,
                    "frequency_most_frequent": freq_most_frequent,
                }
            )

        # Create the summary DataFrame from the list of dictionaries
        summary = pd.DataFrame(summary_data)
        summary.set_index("column", inplace=True)

        return summary
    except Exception as e:
        st.warning(f"Could not summarize categorical columns: {e}")
        return None


# Function to plot correlation heatmap


def plot_corr(df):
    try:
        df_copy = df.copy()

        for col in df_copy.columns:
            if df_copy[col].dtype == "object":  # Check if the column is categorical
                unique_vals = df_copy[col].unique()
                if (
                    len(unique_vals) == 2
                ):  # If the categorical variable has exactly 2 unique values
                    value_counts = df_copy[col].value_counts()
                    df_copy[col] = df_copy[col].map(
                        {value_counts.idxmax(): 0, value_counts.idxmin(): 1}
                    )

        # Keep only numerical and binary categorical columns
        df_copy = df_copy.select_dtypes(include=[np.number])

        if df_copy.empty:
            st.warning("No numeric columns available for correlation heatmap.")
            return None

        corr = df_copy.corr()  # Compute pairwise correlation of columns
        plt.figure(figsize=(12, 10))  # Set the size of the plot
        sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap")
        return plt
    except Exception as e:
        st.warning(f"Could not plot correlation heatmap: {e}")
        return None

# --- New Statistical Test Functions ---

def run_ttest(df, numeric_col, group_col):
    try:
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            st.warning("T-test requires exactly 2 groups.")
            return None
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        t_stat, p_val = stats.ttest_ind(group1, group2)
        st.write(f"T-test between {groups[0]} and {groups[1]} for {numeric_col}:")
        st.write(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.3g}")
        return t_stat, p_val
    except Exception as e:
        st.warning(f"Could not run t-test: {e}")
        return None

def run_anova(df, numeric_col, group_col):
    try:
        groups = [df[df[group_col] == g][numeric_col].dropna() for g in df[group_col].dropna().unique()]
        if len(groups) < 2:
            st.warning("ANOVA requires at least 2 groups.")
            return None
        f_stat, p_val = stats.f_oneway(*groups)
        st.write(f"ANOVA for {numeric_col} by {group_col}:")
        st.write(f"F-statistic = {f_stat:.3f}, p-value = {p_val:.3g}")
        return f_stat, p_val
    except Exception as e:
        st.warning(f"Could not run ANOVA: {e}")
        return None

def run_mannwhitney(df, numeric_col, group_col):
    try:
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            st.warning("Mann-Whitney U test requires exactly 2 groups.")
            return None
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        st.write(f"Mann-Whitney U test between {groups[0]} and {groups[1]} for {numeric_col}:")
        st.write(f"U-statistic = {u_stat:.3f}, p-value = {p_val:.3g}")
        return u_stat, p_val
    except Exception as e:
        st.warning(f"Could not run Mann-Whitney U test: {e}")
        return None

def run_kruskal(df, numeric_col, group_col):
    try:
        groups = [df[df[group_col] == g][numeric_col].dropna() for g in df[group_col].dropna().unique()]
        if len(groups) < 2:
            st.warning("Kruskal-Wallis test requires at least 2 groups.")
            return None
        h_stat, p_val = stats.kruskal(*groups)
        st.write(f"Kruskal-Wallis test for {numeric_col} by {group_col}:")
        st.write(f"H-statistic = {h_stat:.3f}, p-value = {p_val:.3g}")
        return h_stat, p_val
    except Exception as e:
        st.warning(f"Could not run Kruskal-Wallis test: {e}")
        return None

def run_chi2(df, col1, col2):
    try:
        table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        st.write(f"Chi-square test for {col1} vs {col2}:")
        st.write(f"Chi2 = {chi2:.3f}, p-value = {p:.3g}, dof = {dof}")
        st.write("Contingency Table:")
        st.write(table)
        return chi2, p, dof, expected
    except Exception as e:
        st.warning(f"Could not run Chi-square test: {e}")
        return None

def run_simple_linear_regression(df, x_col, y_col):
    try:
        x = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        model = linear_model.LinearRegression()
        model.fit(x, y)
        
        # Get the equation components
        intercept = model.intercept_
        slope = model.coef_[0]
        equation = f"{y_col} = {intercept:.3f} + {slope:.3f} × {x_col}"
        
        st.write(f"Simple linear regression: {y_col} ~ {x_col}")
        st.write(f"Intercept: {intercept:.3f}")
        st.write(f"Slope: {slope:.3f}")
        st.info(f"**Equation**: {equation}")
        
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], label="Data")
        ax.plot(df[x_col], model.predict(x), color="red", label="Fit")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.legend()
        st.pyplot(fig)
        return model
    except Exception as e:
        st.warning(f"Could not run simple linear regression: {e}")
        return None

def run_crosstab(df, col1, col2):
    try:
        table = pd.crosstab(df[col1], df[col2])
        st.write(f"Crosstabulation of {col1} and {col2}:")
        st.write(table)
        return table
    except Exception as e:
        st.warning(f"Could not create crosstab: {e}")
        return None

def plot_time_series(df, time_col, value_col):
    try:
        fig, ax = plt.subplots()
        ax.plot(df[time_col], df[value_col], marker="o")
        ax.set_xlabel(time_col)
        ax.set_ylabel(value_col)
        ax.set_title(f"Time Series: {value_col} over {time_col}")
        st.pyplot(fig)
        return fig
    except Exception as e:
        st.warning(f"Could not plot time series: {e}")
        return None

def plot_missing_data(df):
    try:
        fig = msno.matrix(df)
        st.pyplot(fig.figure)
        return fig
    except Exception as e:
        st.warning(f"Could not plot missing data: {e}")
        return None



def plot_enhanced_association_heatmap(df, title="Enhanced Association Heatmap"):
    # Import associations from dython if available
    try:
        from dython.nominal import associations
    except ImportError:
        st.warning("dython is not installed. Please install it to use enhanced association heatmap.")
        return None

    assoc = associations(
        df,
        theil_u=True,         # asymmetric measure for categorical vars
        plot=True,
        return_results=True,
        nominal_columns='auto',
        figsize=(14, 12),
        mark_columns=True,
        title=title
    )
    return assoc



# @st.cache_resource
# def make_profile(df):
#     sv.analyze(df)
#     return ProfileReport(df, title="Profiling Report")


# Function to plot bar chart
def plot_categorical(df, col_name):
    try:
        # Get frequency of categories
        freq = df[col_name].value_counts()

        # Create bar chart
        plt.figure(figsize=(10, 6))  # set the size of the plot
        plt.bar(freq.index, freq.values)

        # Add title and labels
        plt.title(f"Frequency of Categories for {col_name}")
        plt.xlabel("Category")
        plt.ylabel("Frequency")

        return plt
    except Exception as e:
        st.warning(f"Could not plot bar chart for {col_name}: {e}")
        return None


def plot_numeric(df, col_name):
    try:
        plt.figure(figsize=(10, 6))  # set the size of the plot
        plt.hist(df[col_name], bins=30, alpha=0.5, color="blue", edgecolor="black")

        # Add title and labels
        plt.title(f"Distribution for {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Frequency")

        return plt
    except Exception as e:
        st.warning(f"Could not plot histogram for {col_name}: {e}")
        return None


# Removed unused process_dataframe function


# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .tool-category {
        font-weight: bold;
        color: #1976D2;
        margin-top: 0.8rem;
    }
    .highlight-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("<h1 class='main-title'>📊 AutoAnalyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Interactive data exploration and machine learning for healthcare data</p>", unsafe_allow_html=True)

if "model_output1" not in st.session_state:
    st.session_state.model_output1 = ""

if "model_output2" not in st.session_state:
    st.session_state.model_output2 = ""

if "full_gpt_response" not in st.session_state:
    st.session_state.full_gpt_response = ""

st.markdown(
    "<div class='highlight-box'>Welcome to the AutoAnalyzer! Use the left sidebar to upload your data or select a demo dataset. Then, follow the steps to explore your data.</div>",
    unsafe_allow_html=True
)

with st.expander("📚 Getting Started: Using AutoAnalyzer"):
    st.info("""Be sure your data is first in a 'tidy' format. Use the demo datasets for examples. (*See https://tidyr.tidyverse.org/ for more information.*)
Follow the steps listed in the sidebar on the left. After your exploratory analysis is complete, try the machine learning tab to see if you can predict a target variable.""")
    st.warning(
        "This is not intended to be a comprehensive tool for data analysis. It is meant to be a starting point for data exploration and machine learning. Do not upload PHI. Clone the Github repository and run locally without the chatbot if you have PHI."
    )
    st.markdown("[Github Repository](https://github.com/DrDavidL/auto_analyze)")

    # """)
    st.write(
        "Author: David Liebovitz, MD, Northwestern University, davidl at northwestern dot edu"
    )
    st.write("Last updated 5/4/25")

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🧠 Machine Learning", "🤖 Analyze with GPT"])
# fetch_api_key()
# gpt_version = st.sidebar.radio("Select GPT model:", ("GPT-3.5 ($)", "GPT-4 ($$$$)"), index=0)
# if gpt_version == "GPT-3.5 ($)":
#     selected_model ="gpt-3.5-turbo"
# if gpt_version == "GPT-4 ($$$$)":
#     selected_model = "gpt-4-turbo"
# if gpt_version == "GPT-3.5 16k ($$)":
#     selected_model  = "gpt-3.5-turbo-16k"

# if openai.api_key is None:
#     os.environ["openai-api-key"] = fetch_api_key()
#     openai.api_key = os.getenv("openai-api-key")

with tab1:
    # st.sidebar.subheader("Upload your data")

    st.sidebar.markdown("<div class='step-header'>Step 1: Upload your data or view a demo dataset</div>", unsafe_allow_html=True)
    
    demo_or_custom = st.sidebar.selectbox(
        "Upload a CSV or Excel file. NO PHI - use only anonymized data",
        (
            "🩸 Demo 1 (diabetes)",
            "🔬 Demo 2 (cancer)",
            "❓ Demo 3 (missing data example)",
            "📈 Demo 4 (time series -CHF deaths)",
            "🧠 Demo 5 (stroke)",
            "✨ Generate Data",
            "📁 CSV or Excel Upload",
            "🔄 Modified Dataframe",
        ),
        index=0,
    )
    
    # Add a message in the main area if CSV/Excel upload is selected
    if demo_or_custom == "📁 CSV or Excel Upload":
        st.info("Please use the file uploader that appeared in the sidebar on the left.")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = load_data(uploaded_file)
            else:
                try:
                    st.session_state.df = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.warning(f"Failed to load Excel file: {e}")

    if demo_or_custom == "🩸 Demo 1 (diabetes)":
        file_path = os.path.join("data", "predictdm.csv")
        st.sidebar.markdown(
            "[About Demo 1 dataset](https://hbiostat.org/data/repo/diabetes)"
        )
        st.session_state.df = load_data(file_path)

    if demo_or_custom == "🔬 Demo 2 (cancer)":
        file_path = os.path.join("data", "breastcancernew.csv")
        st.sidebar.write(
            "[About Demo 2 dataset](https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra)"
        )
        st.session_state.df = load_data(file_path)

    if demo_or_custom == "❓ Demo 3 (missing data example)":
        file_path = os.path.join("data", "missing_data.csv")
        st.sidebar.markdown(
            "[About Demo 3 dataset](https://www.lshtm.ac.uk/research/centres-projects-groups/missing-data#dia-missing-data)"
        )
        st.session_state.df = load_data(file_path)

    if demo_or_custom == "Modified Dataframe":
        # st.sidebar.markdown("Using the dataframe from the previous step.")
        if len(st.session_state.modified_df) == 0:
            st.sidebar.warning("No saved dataframe; using demo dataset 1.")
            file_path = "data/predictdm.csv"
            st.sidebar.markdown(
                "[About Demo 1 dataset](https://hbiostat.org/data/repo/diabetes))"
            )
            st.session_state.df = load_data(file_path)

        else:
            st.session_state.df = st.session_state.modified_df
            # st.sidebar.write("Download the modified dataframe as a CSV file.")
        modified_csv = st.session_state.modified_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Modified Dataset!",
            data=modified_csv,
            file_name="modified_data.csv",
            mime="text/csv",
        )

    if demo_or_custom == "✨ Generate Data":
        # Removed authentication info message as requested
        st.sidebar.markdown("Enter column names on the main page ➡️")
        # Move input fields and button to main area, but keep the password check
        if hu_key == "True" or check_password(): # Keep the check
            user_input = st.text_area( # Use st.text_area for main area
                "Enter comma or space separated names for columns, e.g., Na, Cr, WBC, A1c, SPB, Diabetes:"
            )

            if "," in user_input:
                user_list = user_input.split(",")
            elif " " in user_input:
                user_list = user_input.split()
            else:
                user_list = [user_input]

            # Remove leading/trailing whitespace from each item in the list
            user_columns = [item.strip() for item in user_list]
            user_rows = st.number_input( # Use st.number_input for main area
                "Enter approx number of rows (max 100).",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
            )
            if st.button("Generate Data"): # Use st.button for main area
                # Use a default model if not otherwise set
                selected_model = "gpt-4o-mini"
                st.session_state.df, st.session_state.gen_csv = generate_df(
                    user_columns, user_rows, selected_model
                )
                st.info(
                    "Here are the first 5 rows of your generated data. Use the tools in the sidebar to explore your new dataset! And, download and save your new CSV file from the sidebar!"
                )
                st.write(st.session_state.df.head())
        # The check_password() function handles displaying the password input if needed.

    if demo_or_custom == "📈 Demo 4 (time series -CHF deaths)":
        file_path = os.path.join("data", "S1Data.csv")
        st.sidebar.markdown(
            "[About Demo 4 dataset](https://plos.figshare.com/articles/dataset/Survival_analysis_of_heart_failure_patients_A_case_study/5227684/1)"
        )
        st.session_state.df = load_data(file_path)

    if demo_or_custom == "🧠 Demo 5 (stroke)":
        file_path = os.path.join("data", "healthcare-dataset-stroke-data.csv")
        st.sidebar.markdown(
            "[About Demo 5 dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)"
        )
        st.session_state.df = load_data(file_path)

    with st.sidebar:
        if st.session_state.gen_csv is not None:
            # st.warning("Save your generated data!")
            st.download_button(
                label="💾 Download Generated Data!",
                data=st.session_state.gen_csv,
                file_name="patient_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("<div class='step-header'>Step 2: Assess Data Readiness</div>", unsafe_allow_html=True)

        check_preprocess = st.checkbox(
            "🔍 Assess dataset readiness", key="Preprocess now needed"
        )
        needs_preprocess = st.checkbox(
            "🛠️ Select if dataset fails readiness", key="Open Preprocess"
        )
        filter_data = st.checkbox(
            "🔎 Filter data if needed (Switch to Modified Dataframe after filtering)",
            key="Filter data",
        )

        st.markdown("<div class='step-header'>Step 3: Tools for Analysis</div>", unsafe_allow_html=True)
        # Balanced tool options for sidebar columns
        col1, col2 = st.columns(2)
        
        # Add category headers
        st.markdown("<div class='tool-category'>Basic Analysis Tools</div>", unsafe_allow_html=True)
        
        # Re-balance: move three tools from col2_tools to col1_tools for better balance
        col1_tools = [
            "📋 Show header (top 5 rows of data)",
            "📊 Summary (numerical data)",
            "📑 Summary (categorical data)",
            "📝 Create a Table 1",
            "📈 Scatterplot",
            "🔍 View Dataset",
            "📊 Bar chart (categorical data)",
            "📊 Histogram (numerical data)",
            "🥧 Pie chart (categorical data)",
            "🔥 Correlation heatmap",
            "📦 Box plot",
            "🎻 Violin plot",
            "🧪 T-test (2 groups)",
            "🧪 ANOVA (3+ groups)",
        ]
        col2_tools = [
            "🧪 Mann-Whitney U test (2 groups, nonparametric)",
            "🧪 Kruskal-Wallis test (3+ groups, nonparametric)",
            "🧪 Chi-square test (categorical)",
            "📊 Crosstab/Frequency Table",
            "📈 Simple linear regression",
            "📈 Multiple linear regression",
            "🧮 Perform PCA",
            "📈 Time series plot",
            "❓ Visualize missing data",
            "🔬 Categorical outcome analysis (Cohort or case-control datasets)",
            "📉 Survival curve (need duration column)",
            "📉 Cox Proportional Hazards (need duration column)",
            "📑 *(Takes 1-2 minutes*) **Download a Full Analysis** (*Check **Alerts** with key findings.*)",
        ]

        # Use explicit keys for each checkbox to ensure correct mapping
        with col1:
            header = st.checkbox(
                col1_tools[0], key="show_header", help=tool_explanations.get(col1_tools[0], "")
            )
            summary = st.checkbox(
                col1_tools[1], key="show_data", help=tool_explanations.get(col1_tools[1], "")
            )
            summary_cat = st.checkbox(
                col1_tools[2], key="show_summary_cat", help=tool_explanations.get(col1_tools[2], "")
            )
            show_table = st.checkbox(
                col1_tools[3], key="show_table", help=tool_explanations.get(col1_tools[3], "")
            )
            show_scatter = st.checkbox(
                col1_tools[4], key="show_scatter", help=tool_explanations.get(col1_tools[4], "")
            )
            view_full_df = st.checkbox(
                col1_tools[5], key="view_full_df", help=tool_explanations.get(col1_tools[5], "")
            )
            barchart = st.checkbox(
                col1_tools[6], key="show_barchart", help=tool_explanations.get(col1_tools[6], "")
            )
            histogram = st.checkbox(
                col1_tools[7], key="show_histogram", help=tool_explanations.get(col1_tools[7], "")
            )
            piechart = st.checkbox(
                col1_tools[8], key="show_piechart", help=tool_explanations.get(col1_tools[8], "")
            )
            show_corr = st.checkbox(
                col1_tools[9], key="show_corr", help=tool_explanations.get(col1_tools[9], "")
            )
            box_plot = st.checkbox(
                col1_tools[10], key="show_box", help=tool_explanations.get(col1_tools[10], "")
            )
            violin_plot = st.checkbox(
                col1_tools[11], key="show_violin", help=tool_explanations.get(col1_tools[11], "")
            )
            ttest = st.checkbox(
                col1_tools[12], key="ttest", help=tool_explanations.get(col1_tools[12], "")
            )
            anova = st.checkbox(
                col1_tools[13], key="anova", help=tool_explanations.get(col1_tools[13], "")
            )

        with col2:
            mannwhitney = st.checkbox(
                col2_tools[0], key="mannwhitney", help=tool_explanations.get(col2_tools[0], "")
            )
            kruskal = st.checkbox(
                col2_tools[1], key="kruskal", help=tool_explanations.get(col2_tools[1], "")
            )
            chi2 = st.checkbox(
                col2_tools[2], key="chi2", help=tool_explanations.get(col2_tools[2], "")
            )
            crosstab = st.checkbox(
                col2_tools[3], key="crosstab", help=tool_explanations.get(col2_tools[3], "")
            )
            simple_linreg = st.checkbox(
                col2_tools[4], key="simple_linreg", help=tool_explanations.get(col2_tools[4], "")
            )
            mult_linear_reg = st.checkbox(
                col2_tools[5], key="show_mult_linear_reg", help=tool_explanations.get(col2_tools[5], "")
            )
            perform_pca = st.checkbox(
                col2_tools[6], key="show_pca", help=tool_explanations.get(col2_tools[6], "")
            )
            time_series = st.checkbox(
                col2_tools[7], key="time_series", help=tool_explanations.get(col2_tools[7], "")
            )
            missing_data_vis = st.checkbox(
                col2_tools[8], key="missing_data_vis", help=tool_explanations.get(col2_tools[8], "")
            )
            binary_categ_analysis = st.checkbox(
                col2_tools[9], key="binary_categ_analysis", help=tool_explanations.get(col2_tools[9], "")
            )
            survival_curve = st.checkbox(
                col2_tools[10], key="show_survival", help=tool_explanations.get(col2_tools[10], "")
            )
            cox_ph = st.checkbox(
                col2_tools[11], key="show_cox_ph", help=tool_explanations.get(col2_tools[11], "")
            )
            full_analysis = st.checkbox(
                col2_tools[12], key="show_analysis", help=tool_explanations.get("Download a Full Analysis", "")
            )

    if filter_data:
        current_df = st.session_state.df
        st.session_state.modified_df = filter_dataframe(current_df)
        st.write(
            "Switch to Modified Dataframe (top left) to see the filtered data below and use in analysis tools."
        )
        st.session_state.modified_df

    if ttest:
        st.subheader("T-test (2 groups)")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        ttest_num = st.selectbox("Select a numerical column:", numeric_cols, key="ttest_num")
        ttest_cat = st.selectbox("Select a grouping (categorical) column:", categorical_cols, key="ttest_cat")
        if st.button("Run T-test"):
            run_ttest(st.session_state.df, ttest_num, ttest_cat)
        with st.expander("Show code for t-test"):
            st.code(
                f'''from scipy import stats

group1 = df[df["{ttest_cat}"] == df["{ttest_cat}"].unique()[0]]["{ttest_num}"].dropna()
group2 = df[df["{ttest_cat}"] == df["{ttest_cat}"].unique()[1]]["{ttest_num}"].dropna()
t_stat, p_val = stats.ttest_ind(group1, group2)
print("t-statistic:", t_stat, "p-value:", p_val)
''', language="python")

    if anova:
        st.subheader("ANOVA (3+ groups)")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        anova_num = st.selectbox("Select a numerical column:", numeric_cols, key="anova_num")
        anova_cat = st.selectbox("Select a grouping (categorical) column:", categorical_cols, key="anova_cat")
        if st.button("Run ANOVA"):
            run_anova(st.session_state.df, anova_num, anova_cat)
        with st.expander("Show code for ANOVA"):
            st.code(
                f'''from scipy import stats

groups = [df[df["{anova_cat}"] == g]["{anova_num}"].dropna() for g in df["{anova_cat}"].unique()]
f_stat, p_val = stats.f_oneway(*groups)
print("F-statistic:", f_stat, "p-value:", p_val)
''', language="python")

    if mannwhitney:
        st.subheader("Mann-Whitney U test (2 groups, nonparametric)")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        mw_num = st.selectbox("Select a numerical column:", numeric_cols, key="mw_num")
        mw_cat = st.selectbox("Select a grouping (categorical) column:", categorical_cols, key="mw_cat")
        if st.button("Run Mann-Whitney U test"):
            run_mannwhitney(st.session_state.df, mw_num, mw_cat)
        with st.expander("Show code for Mann-Whitney U test"):
            st.code(
                f'''from scipy import stats

group1 = df[df["{mw_cat}"] == df["{mw_cat}"].unique()[0]]["{mw_num}"].dropna()
group2 = df[df["{mw_cat}"] == df["{mw_cat}"].unique()[1]]["{mw_num}"].dropna()
u_stat, p_val = stats.mannwhitneyu(group1, group2)
print("U-statistic:", u_stat, "p-value:", p_val)
''', language="python")

    if kruskal:
        st.subheader("Kruskal-Wallis test (3+ groups, nonparametric)")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        kruskal_num = st.selectbox("Select a numerical column:", numeric_cols, key="kruskal_num")
        kruskal_cat = st.selectbox("Select a grouping (categorical) column:", categorical_cols, key="kruskal_cat")
        if st.button("Run Kruskal-Wallis test"):
            run_kruskal(st.session_state.df, kruskal_num, kruskal_cat)
        with st.expander("Show code for Kruskal-Wallis test"):
            st.code(
                f'''from scipy import stats

groups = [df[df["{kruskal_cat}"] == g]["{kruskal_num}"].dropna() for g in df["{kruskal_cat}"].unique()]
h_stat, p_val = stats.kruskal(*groups)
print("H-statistic:", h_stat, "p-value:", p_val)
''', language="python")

    if chi2:
        st.subheader("Chi-square test (categorical)")
        categorical_cols = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()
        chi2_col1 = st.selectbox("Select first categorical column:", categorical_cols, key="chi2_col1")
        chi2_col2 = st.selectbox("Select second categorical column:", categorical_cols, key="chi2_col2")
        if st.button("Run Chi-square test"):
            run_chi2(st.session_state.df, chi2_col1, chi2_col2)
        with st.expander("Show code for Chi-square test"):
            st.code(
                f'''from scipy import stats

table = pd.crosstab(df["{chi2_col1}"], df["{chi2_col2}"])
chi2, p, dof, expected = stats.chi2_contingency(table)
print("Chi2:", chi2, "p-value:", p, "dof:", dof)
print(table)
''', language="python")

    if crosstab:
        st.subheader("Crosstab/Frequency Table")
        categorical_cols = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()
        ct_col1 = st.selectbox("Select first categorical column:", categorical_cols, key="ct_col1")
        ct_col2 = st.selectbox("Select second categorical column:", categorical_cols, key="ct_col2")
        if st.button("Show Crosstab"):
            run_crosstab(st.session_state.df, ct_col1, ct_col2)
        with st.expander("Show code for crosstab"):
            st.code(
                f'''table = pd.crosstab(df["{ct_col1}"], df["{ct_col2}"])
print(table)
''', language="python")

    if simple_linreg:
        st.subheader("Simple linear regression")
        numeric_cols, _ = get_categorical_and_numerical_cols(st.session_state.df)
        slr_x = st.selectbox("Select X (predictor):", numeric_cols, key="slr_x")
        slr_y = st.selectbox("Select Y (outcome):", numeric_cols, key="slr_y")
        if st.button("Run Simple Linear Regression"):
            run_simple_linear_regression(st.session_state.df, slr_x, slr_y)
        with st.expander("Show code for simple linear regression"):
            st.code(
                f'''from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = df["{slr_x}"].values.reshape(-1, 1)
y = df["{slr_y}"].values
model = LinearRegression()
model.fit(x, y)
plt.scatter(df["{slr_x}"], df["{slr_y}"])
plt.plot(df["{slr_x}"], model.predict(x), color="red")
plt.xlabel("{slr_x}")
plt.ylabel("{slr_y}")
plt.title("{slr_y} vs {slr_x}")
plt.show()
''', language="python")

    if time_series:
        st.subheader("Time series plot")
        cols = st.session_state.df.columns.tolist()
        time_col = st.selectbox("Select time column:", cols, key="ts_time")
        value_col = st.selectbox("Select value column:", cols, key="ts_value")
        if st.button("Plot Time Series"):
            plot_time_series(st.session_state.df, time_col, value_col)
        with st.expander("Show code for time series plot"):
            st.code(
                f'''import matplotlib.pyplot as plt

plt.plot(df["{time_col}"], df["{value_col}"], marker="o")
plt.xlabel("{time_col}")
plt.ylabel("{value_col}")
plt.title("Time Series: {value_col} over {time_col}")
plt.show()
''', language="python")

    if missing_data_vis:
        st.subheader("Visualize missing data")
        plot_missing_data(st.session_state.df)
        with st.expander("Show code for missing data visualization"):
            st.code(
                '''import missingno as msno

msno.matrix(df)
plt.show()
''', language="python")

    if mult_linear_reg:
        st.subheader("Multiple Linear Regression")
        st.warning(
            "This tool is for use with numerical data only; binary categorical variables are updated to 1 and 0 and explained below if needed."
        )
        # Get column names for time and event from the user
        temp_df_mlr = st.session_state.df.copy()
        numeric_columns_mlr = all_numerical(temp_df_mlr)

        x_col = st.multiselect(
            "Select the columns for x", numeric_columns_mlr, numeric_columns_mlr[1]
        )
        y_col = st.selectbox("Select the column for y", numeric_columns_mlr)
        # Convert the columns to numeric values
        # temp_df_mlr[x_col] = temp_df_mlr[x_col].astype(float)
        # temp_df_mlr[y_col] = temp_df_mlr[y_col].astype(float)
        # x_col_array = np.array(x_col)
        # y_col_array = np.array(y_col)

        # x_col_reshaped = x_col_array.reshape(-1, 1)
        # y_col_reshaped = y_col_array.reshape(-1, 1)
        # Plot the survival curve
        try:
            mult_linear_reg, mlr_report, intercept, coef = plot_mult_linear_reg(
                temp_df_mlr, temp_df_mlr[x_col], temp_df_mlr[y_col]
            )
            mlr_equation = generate_regression_equation(intercept, coef, x_col)
            show_equation = st.checkbox("Show regression equation")
            # mlr_report
            if show_equation:
                st.write(mlr_equation)
            st.write("Download your cooefficients and intercept below.")
            df_download_options(mlr_report, "Your Multiple Linear Regression")
            with st.expander("Show code for multiple linear regression"):
                st.code(
                    f'''from sklearn.linear_model import LinearRegression

# Fit multiple linear regression
X = df[{x_col}]
y = df["{y_col}"]
regr = LinearRegression()
regr.fit(X, y)
print("Intercept:", regr.intercept_)
print("Coefficients:", regr.coef_)
''', language="python")
        except:
            st.error("Please select at least one column for x and one column for y.")
        # save_image(mult_linear_reg, 'mult_linear_reg.png')
        # df_download_options(mult_linear_reg, 'csv')
        with st.expander("What is a Multiple Linear Regression?"):
            from prompts import mult_linear_reg_text
            st.write(mult_linear_reg_text)

    if cox_ph:
        df = st.session_state.df

        st.markdown("## Cox Analysis: Select Columns")

        categ_columns_cox = all_categorical(df)
        numeric_columns_cox = all_numerical(df)

        event_col = st.selectbox(
            "Select the event column", categ_columns_cox, key="event_col"
        )
        selected_columns_cox = st.multiselect(
            "Choose your feature columns", numeric_columns_cox
        )
        duration_col = st.selectbox("Select the duration column", numeric_columns_cox)

        if st.button("Analyze", key="analyze"):
            if len(selected_columns_cox) < 1:
                st.error("Select at least one column!")
            else:
                cph_data = df[selected_columns_cox + [event_col] + [duration_col]]
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cph_data, duration_col=duration_col, event_col=event_col)
                summary_cox = cph.summary
                st.session_state.df_to_download = summary_cox
                st.subheader("Summary of the Cox PH Analysis")
                st.info(
                    "Note, the exp(coef) column is the hazard ratio for each variable."
                )
                st.dataframe(summary_cox)
                with st.expander("Show code for Cox Proportional Hazards model"):
                    st.code(
                        f'''from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df[{selected_columns_cox + [event_col] + [duration_col]}], duration_col="{duration_col}", event_col="{event_col}")
print(cph.summary)
''', language="python")
        else:
            st.text("Select columns & hit 'Analyze'.")
        if st.session_state.df_to_download is not None:
            df_download_options(st.session_state.df_to_download, "cox_ph_summary")
        with st.expander("What is a Cox Proportional Hazards Analysis?"):
            from prompts import cox_text
            st.write(cox_text)

    if survival_curve:
        st.subheader("Survival Curve")
        st.warning(
            "This tool is for use with survival analysis data. Any depiction will not make sense if 'time' isn't a column for your dataset"
        )
        time_col = st.selectbox(
            "Select the column for time", st.session_state.df.columns
        )
        event_col = st.selectbox(
            "Select the column for event", st.session_state.df.columns
        )

        surv_curve = plot_survival_curve(st.session_state.df, time_col, event_col)
        with st.expander("Show code for Kaplan-Meier survival curve"):
            st.code(
                f'''from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

kmf = KaplanMeierFitter()
kmf.fit(df["{time_col}"], event_observed=df["{event_col}"])
fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("Survival Probability")
ax.set_title("Survival Curve")
plt.show()
''', language="python")
        save_image(surv_curve, "survival_curve.png")
        with st.expander("What is a Kaplan-Meier Curve?"):
            from prompts import kaplan_meier_text
            st.write(kaplan_meier_text)

    if binary_categ_analysis:
        st.subheader("""
        Choose your exposures and outcomes.
        """)
        st.info("Note - categories with more than 15 unique values will not be used.")
        var1, var2 = st.columns(2)
        s_categorical_cols = st.session_state.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        numeric_cols = [
            col
            for col in st.session_state.df.columns
            if st.session_state.df[col].nunique() == 2
            and st.session_state.df[col].dtype != "object"
        ]
        filtered_categorical_cols = [
            col
            for col in s_categorical_cols
            if st.session_state.df[col].nunique() <= 15
        ]
        sd_categorical_cols = filtered_categorical_cols + numeric_cols
        if len(sd_categorical_cols) > 1:
            sd_exposure = var1.selectbox(
                "Select a categorical column as the exposure:",
                sd_categorical_cols,
                index=0,
            )
            sd_outcome = var2.selectbox(
                "Select a categorical column as the outcome:",
                sd_categorical_cols,
                index=1,
            )
            sd_exposure_values = var1.multiselect(
                "Select one or more values for the exposure:",
                st.session_state.df[sd_exposure].unique().tolist(),
                [st.session_state.df[sd_exposure].unique().tolist()[1]],
            )
            sd_outcome_values = var2.multiselect(
                "Select one or more values for the outcome:",
                st.session_state.df[sd_outcome].unique().tolist(),
                [st.session_state.df[sd_outcome].unique().tolist()[1]],
            )

            # Create a temporary dataframe to store the modified values
            temp_df = st.session_state.df.copy()

            # Replace the selected exposure values with 1 and others with 0
            temp_df[sd_exposure] = temp_df[sd_exposure].apply(
                lambda x: 1 if x in sd_exposure_values else 0
            )

            # Replace the selected outcome values with 1 and others with 0
            temp_df[sd_outcome] = temp_df[sd_outcome].apply(
                lambda x: 1 if x in sd_outcome_values else 0
            )

            cohort_or_case = st.radio(
                "Choose an approach", ("Cohort Study", "Case Control Study")
            )

            # Generate the 2x2 table
            table = generate_2x2_table(temp_df, sd_exposure, sd_outcome)
            if cohort_or_case == "Cohort Study":
                st.write("For use with cohort study data.")

                # Calculate relative risk, ARR, and NNT
                tn = table.iloc[0, 0]
                fp = table.iloc[0, 1]
                fn = table.iloc[1, 0]
                tp = table.iloc[1, 1]
                rr, arr, nnt = calculate_rr_arr_nnt(tn, fp, fn, tp)

                # Display the 2x2 table and analysis results
                st.subheader("2x2 Table")
                st.write(table)
                st.subheader("Results")
                st.write("Relative Risk (RR):", round(rr, 2))
                st.write("Absolute Risk Reduction (ARR):", round(arr, 2))
                st.write("Number Needed to Treat (NNT):", round(nnt, 2))

            if cohort_or_case == "Case Control Study":
                st.write("For use with case-control data.")
                # Calculate odds and odds ratio
                odds_cases, odds_controls, odds_ratio = calculate_odds(table)

                # Display the 2x2 table and analysis results
                st.subheader("2x2 Table")
                st.write(table)
                st.subheader("Results")
                st.write("Odds in cases:", round(odds_cases, 2))
                st.write("Odds in controls:", round(odds_controls, 2))
                st.write("Odds Ratio:", round(odds_ratio, 2))
        else:
            st.subheader("Insufficient categorical variables found in the data.")

    if needs_preprocess:
        st.info(
            "Data Preprocessing Tools - *Assess Data Readiness **first**. Use only if needed.*"
        )
        st.write(
            "Step 1: Make a copy of your dataset to modify by clicking the button below."
        )
        if st.button("Copy dataset"):
            st.session_state.modified_df = st.session_state.df
        st.write(
            "Step 2: Select 'Modified Dataframe' in Step 1 of the sidebar to use the dataframe you just copied."
        )
        st.write(
            "Step 3: Select a method to impute missing values in your dataset. Built in checks to apply only to applicable data types."
        )
        method = st.selectbox(
            "Choose a method to replace missing values",
            ("Select here!", "drop", "zero", "mean", "median", "mode", "mice"),
        )
        if st.button("Apply the Method to Replace Missing Values"):
            st.session_state.modified_df = replace_missing_values(
                st.session_state.modified_df, method
            )
        st.write(
            "Recheck data readiness to see if you are ready to proceed with analysis."
        )

    # if activate_chatbot:

    if summary:
        st.info("Summary of numerical data")
        sum_num_data = st.session_state.df.describe()
        st.write(sum_num_data)
        with st.expander("Show code for summary (describe)"):
            st.code(
                '''# Show summary statistics for numerical columns
summary = df.describe()
print(summary)
''', language="python")
        st.session_state.df_to_download = sum_num_data
        if st.session_state.df_to_download is not None:
            df_download_options(
                st.session_state.df_to_download, "numerical_data_summary"
            )

    if header:
        st.info("First 5 Rows of Data")
        st.write(st.session_state.df.head())
        with st.expander("Show code for displaying first 5 rows"):
            st.code(
                '''# Show the first 5 rows of the dataframe
print(df.head())
''', language="python")

    if full_analysis:
        # Suppress all deprecation warnings during full analysis
        import warnings
        
        full_analysis_method = st.radio(
            "Choose a method for full analysis", ("Pandas Profiling", "Sweetviz")
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            if full_analysis_method == "Sweetviz":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                    st.info(
                        "Full analysis of data using [*Sweetviz*](https://github.com/fbdesignpro/sweetviz)"
                    )
                    report_path = temp_file.name
                    report = make_sweet_report(st.session_state.df)
                    report.show_html(
                        filepath=report_path,
                        open_browser=False,
                        layout="vertical",
                        scale=1.0,
                    )
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download Sweetviz Report",
                            data=file,
                            file_name="SWEETVIZ_REPORT.html",
                            mime="text/html",
                        )
                    with open(report_path, "r", encoding="utf-8") as display:
                        source_code = display.read()
                    components.html(source_code, height=1200, scrolling=True)

            if full_analysis_method == "Pandas Profiling":
                st.info(
                    "Full analysis of data using [*Pandas Profiling*](https://github.com/ydataai/ydata-profiling). Check out *alerts*!"
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                    report_path = temp_file.name
                    with st.spinner("Generating the report..."):
                        report = make_pandas_report(
                            st.session_state.df, title="Pandas Profiling Report"
                        )
                        report.to_file(report_path)
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Pandas Profiling Report",
                            data=file,
                            file_name="Pandas_Profile.html",
                            mime="text/html",
                        )
                    with open(report_path, "r", encoding="utf-8") as display:
                        source_code = display.read()
                    components.html(source_code, height=1200, scrolling=True)

    if histogram:
        st.info("Histogram of data")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        selected_col = st.selectbox("Choose a column", numeric_cols, key="histogram")
        if selected_col:
            plt = plot_numeric(st.session_state.df, selected_col)
            st.pyplot(plt)
            with st.expander("Show code for histogram"):
                st.code(
                    f'''import matplotlib.pyplot as plt

# Histogram for numeric column
plt.figure(figsize=(10, 6))
plt.hist(df["{selected_col}"], bins=30, alpha=0.5, color="blue", edgecolor="black")
plt.title("Distribution for {selected_col}")
plt.xlabel("{selected_col}")
plt.ylabel("Frequency")
plt.show()
''', language="python")
        save_image(plt, "histogram.png")

    if barchart:
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        cat_selected_col = st.selectbox(
            "Choose a column", categorical_cols, key="bar_category"
        )
        if cat_selected_col:
            plt = plot_categorical(st.session_state.df, cat_selected_col)
            st.pyplot(plt)
            with st.expander("Show code for bar chart"):
                st.code(
                    f'''import matplotlib.pyplot as plt

# Bar chart for categorical column
df["{cat_selected_col}"].value_counts().plot(kind="bar")
plt.xlabel("{cat_selected_col}")
plt.ylabel("Frequency")
plt.title("Frequency of Categories for {cat_selected_col}")
plt.show()
''', language="python")
        save_image(plt, "bar_chart.png")
        with st.expander("Expand for Python|Streamlit Code"):
            st.code("""
import matplotlib.pyplot as plt
import pandas as pd

# Function to get categorical and numerical columns from the dataframe
def get_categorical_and_numerical_cols(df):
numeric_cols = []
categorical_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)
return numeric_cols, categorical_cols

# Function to plot the categorical data
def plot_categorical(df, column):
plt.figure(figsize=(10, 6))
df[column].value_counts().plot(kind='bar')
plt.xlabel(column)
plt.ylabel('Count')
plt.title(f'Bar Chart for {column}')
plt.xticks(rotation=45)
plt.tight_layout()
return plt

# Get the numeric and categorical columns from the dataframe
numeric_cols, categorical_cols = get_categorical_and_numerical_cols(df)

# Select a column from the categorical columns
cat_selected_col = input("Choose a column: ")

# Check if a column is selected
if cat_selected_col in categorical_cols:
# Plot the categorical data
plt = plot_categorical(df, cat_selected_col)
plt.show()
            """)

    if show_corr:
        st.info("Correlation heatmap")
        plt = plot_corr(st.session_state.df)
        st.pyplot(plt)
        with st.expander("Show code for correlation heatmap"):
            st.code(
                '''import seaborn as sns
import matplotlib.pyplot as plt

# Convert binary categorical columns to numeric if needed
df_copy = df.copy()
for col in df_copy.columns:
    if df_copy[col].dtype == "object" and len(df_copy[col].unique()) == 2:
        value_counts = df_copy[col].value_counts()
        df_copy[col] = df_copy[col].map({value_counts.idxmax(): 0, value_counts.idxmin(): 1})

# Compute correlation matrix and plot
corr = df_copy.select_dtypes(include=[float, int]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()
''', language="python")
        save_image(plt, "heatmap.png")
        with st.expander("What is a correlation heatmap?"):
            from prompts import correlation_heatmap_text
            st.write(correlation_heatmap_text)

    if summary_cat:
        st.info("Summary of categorical data")
        summary = summarize_categorical(st.session_state.df)
        st.write(summary)
        with st.expander("Show code for summary of categorical data"):
            st.code(
                '''# Summarize categorical columns
cat_df = df.select_dtypes(include=["object", "category"])
summary_data = []
for col in cat_df.columns:
    unique_count = df[col].nunique()
    most_frequent = df[col].mode()[0]
    freq_most_frequent = df[col].value_counts().iloc[0]
    summary_data.append({
        "column": col,
        "unique_count": unique_count,
        "most_frequent": most_frequent,
        "frequency_most_frequent": freq_most_frequent,
    })
summary = pd.DataFrame(summary_data).set_index("column")
print(summary)
''', language="python")
        st.session_state.df_to_download = summary
        if st.session_state.df_to_download is not None:
            df_download_options(st.session_state.df_to_download, "categorical_summary")

    if piechart:
        st.info("Pie chart for categorical data")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        cat_selected_col = st.selectbox(
            "Choose a column", categorical_cols, key="pie_category"
        )
        if cat_selected_col:
            plt = plot_pie(st.session_state.df, cat_selected_col)
            st.pyplot(plt)
            with st.expander("Show code for pie chart"):
                st.code(
                    f'''import matplotlib.pyplot as plt

# Pie chart for categorical column
df["{cat_selected_col}"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribution for {cat_selected_col}")
plt.show()
''', language="python")
        save_image(plt, "pie_chart.png")

    if check_preprocess:
        # st.write("Running readiness assessment...")
        readiness_summary = assess_data_readiness(st.session_state.df)
        # st.write("Readiness assessment complete.")
        # Display the readiness summary using Streamlit
        # Display the readiness summary using Streamlit
        st.subheader("Data Readiness Summary")
        st.info("Original Column Sequence")

        try:
            if readiness_summary["data_empty"]:
                st.write("The DataFrame is empty.")
            else:
                # Combine column information and readiness summary into a single DataFrame
                column_info_df = pd.DataFrame.from_dict(
                    readiness_summary["columns"], orient="index", columns=["Data Type"]
                )
                summary_df = pd.DataFrame.from_dict(
                    readiness_summary["missing_values"],
                    orient="index",
                    columns=["Missing Values"],
                )
                summary_df["Data Type"] = column_info_df["Data Type"]

                # Display the combined table
                st.write(summary_df)

                if readiness_summary["missing_columns"]:
                    st.write("Missing Columns:")
                    st.write(readiness_summary["missing_columns"])

                if readiness_summary["inconsistent_data_types"]:
                    st.write("Inconsistent Data Types:")
                    st.write(readiness_summary["inconsistent_data_types"])

                if readiness_summary["data_ready"]:
                    st.success("The data is ready for analysis!")
                else:
                    st.warning("The data is not fully ready for analysis.")
        except:
            st.write("The DataFrame is isn't yet ready for readiness assessment. :)  ")
            # st.info("Check if you need to preprocess data")
            # missing_values, outliers, data_types, skewness, cardinality = analyze_dataframe(df)
            # st.write("Missing values")
            # st.write(missing_values)
            # st.write("Outliers")
            # st.write(outliers)
            # st.write("Data types")
            # st.write(data_types)
            # st.write("Skewness")
            # st.write(skewness)
            # st.write("Cardinality")
            # st.write(cardinality)

    if show_scatter:
        st.info("Scatterplot")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        numeric_cols.sort()
        categorical_cols.sort()
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("Select column for x axis:", numeric_cols)
        with col2:
            scatter_y = st.selectbox("Select column for y axis:", numeric_cols, index=1)

        with st.expander("Filter Options"):
            remaining_cols = [
                col for col in numeric_cols if col != scatter_x and col != scatter_y
            ]
            if remaining_cols:
                filter_col = st.selectbox(
                    "Select a numerical column to filter data:", remaining_cols
                )
                if filter_col:
                    min_val, max_val = (
                        float(st.session_state.df[filter_col].min()),
                        float(st.session_state.df[filter_col].max()),
                    )
                    if np.isnan(min_val) or np.isnan(max_val):
                        st.write(
                            f"Cannot filter by {filter_col} because it contains NaN values."
                        )
                    else:
                        filter_range = st.slider(
                            "Select a range to filter data:",
                            min_val,
                            max_val,
                            (min_val, max_val),
                        )
                        st.session_state.df = st.session_state.df[
                            (st.session_state.df[filter_col] >= filter_range[0])
                            & (st.session_state.df[filter_col] <= filter_range[1])
                        ]

            if categorical_cols:
                filter_cat_col = st.selectbox(
                    "Select a categorical column to filter data:", categorical_cols
                )
                if filter_cat_col:
                    categories = st.session_state.df[filter_cat_col].unique().tolist()
                    selected_categories = st.multiselect(
                        "Select categories to include in the data:",
                        categories,
                        default=categories,
                    )
                    st.session_state.df = st.session_state.df[
                        st.session_state.df[filter_cat_col].isin(selected_categories)
                    ]
        if st.session_state.df.empty:
            st.write(
                "The current filter settings result in an empty dataset. Please adjust the filter settings."
            )
        else:
            scatterplot = create_scatterplot(st.session_state.df, scatter_x, scatter_y)
            with st.expander("Show code for scatterplot"):
                st.code(
                    f'''import seaborn as sns
import matplotlib.pyplot as plt

# Scatterplot with regression line
sns.regplot(x="{scatter_x}", y="{scatter_y}", data=df)
plt.title("Scatter Plot for {scatter_y} vs {scatter_x}")
plt.show()
''', language="python")
            save_image(scatterplot, "custom_scatterplot.png")

    if box_plot:
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        numeric_cols.sort()
        categorical_cols.sort()
        numeric_col = st.selectbox(
            "Select a numerical column:", numeric_cols, key="box_numeric"
        )
        categorical_col = st.selectbox(
            "Select a categorical column:", categorical_cols, key="box_category"
        )
        mybox = create_boxplot(
            st.session_state.df, numeric_col, categorical_col, show_points=False
        )
        with st.expander("Show code for box plot"):
            st.code(
                f'''import seaborn as sns
import matplotlib.pyplot as plt

# Notched box plot
sns.boxplot(x="{categorical_col}", y="{numeric_col}", data=df, notch=True)
plt.title("Box Plot of {numeric_col} by {categorical_col}")
plt.show()
''', language="python")
        save_image(mybox, "box_plot.png")
        with st.expander("What is a box plot?"):
            from prompts import box_plot_text
            st.write(box_plot_text)

    if violin_plot:
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(
            st.session_state.df
        )
        numeric_cols.sort()
        categorical_cols.sort()
        numeric_col = st.selectbox(
            "Select a numerical column:", numeric_cols, key="violin_numeric"
        )
        categorical_col = st.selectbox(
            "Select a categorical column:", categorical_cols, key="violin_category"
        )

        violin = create_violinplot(st.session_state.df, numeric_col, categorical_col)
        with st.expander("Show code for violin plot"):
            st.code(
                f'''import seaborn as sns
import matplotlib.pyplot as plt

# Violin plot
sns.violinplot(x="{categorical_col}", y="{numeric_col}", data=df)
plt.title("Violin Plot of {numeric_col} by {categorical_col}")
plt.show()
''', language="python")
        save_image(violin, "violin_plot.png")
        with st.expander("What is a violin plot?"):
            from prompts import violin_plot_text
            st.write(violin_plot_text)

    if view_full_df:
        st.dataframe(st.session_state.df)
        st.download_button(
            label="Download current (filtered/cleaned) data as CSV",
            data=st.session_state.df.to_csv(index=False),
            file_name="filtered_data.csv",
            mime="text/csv",
        )

    if show_table:
        if st.session_state.df.shape[1] > 99:
            st.warning(
                f"You have {st.session_state.df.shape[1]} columns. This would not look good in a publication. Less than 50 would be much better."
            )
        else:
            nunique = st.session_state.df.select_dtypes(
                include=["object", "category"]
            ).nunique()
            to_drop = nunique[nunique > 15].index
            df_filtered = st.session_state.df.drop(to_drop, axis=1)
            numerical_columns = df_filtered.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            for col in numerical_columns:
                if df_filtered[col].nunique() == 2:
                    df_filtered[col] = df_filtered[col].astype(str)

            categorical = df_filtered.select_dtypes(include=[object]).columns.tolist()

            st.header("Table 1")
            categorical_variable = st.selectbox(
                "Select the categorical variable for grouping:", options=categorical
            )
            nonnormal_variables = st.multiselect(
                "Select any non-normally distributed variables for rank-based analysis",
                df_filtered.columns.tolist(),
            )

            table = generate_table(
                df_filtered, categorical_variable, nonnormal_variables
            )
            with st.expander("Show code for Table 1 (TableOne)"):
                st.code(
                    f'''from tableone import TableOne

# Prepare your dataframe (df_filtered), categorical variable, and nonnormal variables
mytable = TableOne(
    df_filtered,
    columns=df_filtered.columns.tolist(),
    categorical={categorical},
    groupby="{categorical_variable}",
    nonnormal={nonnormal_variables},
    pval=True,
)
print(mytable.tabulate(tablefmt="github"))
''', language="python")
            st.write(table.tabulate(tablefmt="github"))
            st.write("-------")
            st.info("""Courtesy of TableOne: Tom J Pollard, Alistair E W Johnson, Jesse D Raffa, Roger G Mark;
tableone: An open source Python package for producing summary statistics
for research papers, JAMIA Open, Volume 1, Issue 1, 1 July 2018, Pages 26–31,
https://doi.org/10.1093/jamiaopen/ooy012""")
            st.write("-------")
            if st.checkbox("Click to Download Your Table 1"):
                table_format = st.selectbox(
                    "Select a file format:", ["csv", "excel", "html", "latex"]
                )

                if table_format == "excel":
                    output_path = (
                        f"{st.session_state.outputs_path}/tableone_results.xlsx"
                    )
                    table.to_excel(output_path)
                    st.markdown(
                        get_download_link(output_path, "xlsx"), unsafe_allow_html=True
                    )

                if table_format == "csv":
                    output_path = (
                        f"{st.session_state.outputs_path}/tableone_results.csv"
                    )
                    table.to_csv(output_path)
                    st.markdown(
                        get_download_link(output_path, "csv"), unsafe_allow_html=True
                    )

                if table_format == "html":
                    output_path = (
                        f"{st.session_state.outputs_path}/tableone_results.html"
                    )
                    table.to_html(output_path)
                    st.markdown(
                        get_download_link(output_path, "html"), unsafe_allow_html=True
                    )

                if table_format == "latex":
                    output_path = (
                        f"{st.session_state.outputs_path}/tableone_results.tex"
                    )
                    table.to_latex(output_path)
                    st.markdown(
                        get_download_link(output_path, "tex"), unsafe_allow_html=True
                    )

    if perform_pca:
        # Create PCA plot

        pca_fig2 = perform_pca_plot(st.session_state.df)
        with st.expander("Show code for PCA plot"):
            st.code(
                '''from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Standardize features
x = StandardScaler().fit_transform(df.select_dtypes(include=[float, int]))

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("2 component PCA")
ax.scatter(principalDf["PC1"], principalDf["PC2"])
plt.show()
''', language="python")
        save_image(pca_fig2, f"./{st.session_state.outputs_path}/pca_plot.png")
        scree_plot = create_scree_plot(st.session_state.df)
        with st.expander("Show code for scree plot"):
            st.code(
                '''from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

x = StandardScaler().fit_transform(df.select_dtypes(include=[float, int]))
pca = PCA(n_components=None)
pca.fit_transform(x)
plt.plot(
    range(1, len(pca.explained_variance_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
)
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.show()
''', language="python")
        save_image(scree_plot, f"./{st.session_state.outputs_path}/scree_plot.png")

        with st.expander("What is PCA?"):
            st.write("""Principal Component Analysis, or PCA, is a method used to highlight important information in datasets that have many variables and to bring out strong patterns in a dataset. It's a way of identifying underlying structure in data.

Here's an analogy that might make it more understandable: Imagine a swarm of bees flying around in a three-dimensional space: up/down, left/right, and forward/backward. These are our original variables. Now, imagine you want to take a picture of this swarm that captures as much information as possible, but your camera can only take pictures in two dimensions. You can rotate your camera in any direction, but once you take a picture, you'll lose the third dimension. 

PCA helps us choose the best angle to take this picture. The first principal component (PC1) represents the best angle that captures the most variation in the swarm. The second principal component (PC2) is the best angle perpendicular to the first that captures the remaining variation, and so on. The idea is to minimize the information (variance) lost when we reduce dimensionality (like going from a 3D swarm to a 2D picture).

In a medical context, you might have data from thousands of genes or hundreds of physical and behavioral characteristics. Not all of these variables are independent, and many of them tend to change together. PCA allows us to represent the data in fewer dimensions that capture the most important variability in the dataset. 

Each Principal Component represents a combination of original features (like genes or patient characteristics) and can often be interpreted in terms of those features. For example, a PC might represent a combination of patient's age, blood pressure, and cholesterol level. The coefficients of the features in the PC (the "loadings") tell us how much each feature contributes to that PC.

Finally, PCA can be particularly useful in visualizing high-dimensional data. By focusing on the first two or three principal components, we can create a scatterplot of our data, potentially highlighting clusters or outliers. However, remember that this visualization doesn't capture all the variability in the data—only the variability best captured by the first few principal components.""")

with tab2:
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 15px; border-radius: 5px; border-left: 5px solid #1E88E5;">
        <h3 style="margin-top: 0; color: #1976D2;">Machine Learning Playground</h3>
        <p>This section shows a glimpse of what's possible with machine learning on your data. Any model shown is not yet optimized and requires ML and domain expertise. This is a good starting point to explore predictive modeling with your dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        x = st.session_state.df
    except NameError:
        st.warning(
            "First upload a CSV file or choose a demo dataset from the **Data Exploration** tab"
        )
    else:
        # Filter categorical columns and numerical bivariate columns
        categorical_cols = st.session_state.df.select_dtypes(
            include=[object]
        ).columns.tolist()

        # Add bivariate numerical columns
        numerical_bivariate_cols = [
            col
            for col in st.session_state.df.select_dtypes(
                include=["int64", "float64"]
            ).columns
            if st.session_state.df[col].nunique() == 2
        ]

        # Combine the two lists and sort them
        categorical_cols = categorical_cols + numerical_bivariate_cols
        categorical_cols.sort()  # sort the list of columns

        with st.expander("Click to see your current dataset"):
            st.info("The first 5 rows:")
            st.write(st.session_state.df.head())

        st.subheader("""
        Choose the Target Column
        """)
        target_col = st.selectbox(
            "Select a categorical column as the target:", categorical_cols
        )

        st.subheader("""
        Set the Target Class Value to Predict
        """)
        try:
            categories_to_predict = st.multiselect(
                "Select one or more categories but not all. You need 2 options to predict a group, i.e, your target versus the rest.:",
                st.session_state.df[target_col].unique().tolist(),
                key="target_categories-ml",
            )

            # Preprocess the data and exclude the target column from preprocessing
            df_processed, included_cols, excluded_cols = preprocess(
                st.session_state.df.drop(columns=[target_col]), target_col
            )
            df_processed[target_col] = st.session_state.df[
                target_col
            ]  # Include the target column back into the dataframe

            st.subheader("""
            Select Features to Include in the Model
            """)
            st.info(f"Available Features for your Model: {included_cols}")
            st.warning(
                f"Your Selected Target for Prediction: {target_col} = {categories_to_predict}"
            )
            all_features = st.checkbox(
                "Select all features", value=False, key="select_all_features-10"
            )
            if all_features:
                final_columns = included_cols
            else:
                final_columns = st.multiselect(
                    "Select features to include in your model:",
                    included_cols,
                    key="columns_to_include-10",
                )
            if len(excluded_cols) > 0:
                st.write(f"Unavailable columns for modeling: {excluded_cols}")

            # Create binary target variable based on the selected categories
            df_processed[target_col] = df_processed[target_col].apply(
                lambda x: 1 if x in categories_to_predict else 0
            )
            X = df_processed[final_columns]
            # st.write(X.head())

            # Split the dataframe into data and labels
            # List of available scaling options
            scaling_options = {
                "No Scaling": None,
                "Standard Scaling": StandardScaler(),
                "Min-Max Scaling": MinMaxScaler(),
            }

            # List of available normalization options
            normalization_options = {
                "No Normalization": None,
                "L1 Normalization": "l1",
                "L2 Normalization": "l2",
            }
            scaling_or_norm = st.checkbox(
                "Scaling or Normalization?", value=False, key="scaling_or_norm-10"
            )
            # User selection for scaling option
            if scaling_or_norm == True:
                scaling_option = st.selectbox(
                    "Select Scaling Option", list(scaling_options.keys())
                )
                # User selection for normalization option
                normalization_option = st.selectbox(
                    "Select Normalization Option", list(normalization_options.keys())
                )

                # Apply selected scaling and normalization options to the features

                if scaling_option != "No Scaling":
                    scaler = scaling_options[scaling_option]
                    X = scaler.fit_transform(X)

                if normalization_option != "No Normalization":
                    normalization_type = normalization_options[normalization_option]
                    X = normalize(X, norm=normalization_type)
            # X = df_processed.drop(columns=[target_col])
            # X = df_processed[final_columns]
            y = df_processed[target_col]

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # pca_check = st.checkbox("PCA?", value=False, key="pca_check-10")
            # if pca_check == True:
            #     n_neighbors = 3
            #     random_state = 0
            #     dim = len(X[0])
            #     n_classes = len(np.unique(y))

            #     # Reduce dimension to 2 with PCA
            #     pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

            #     # Reduce dimension to 2 with LinearDiscriminantAnalysis
            #     lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

            #     # Reduce dimension to 2 with NeighborhoodComponentAnalysis
            #     nca = make_pipeline(
            #         StandardScaler(),
            #         NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
            #     )

            #     # Use a nearest neighbor classifier to evaluate the methods
            #     knn = KNeighborsClassifier(n_neighbors=n_neighbors)

            #     # Make a list of the methods to be compared
            #     dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

            #     # plt.figure()
            #     for i, (name, model) in enumerate(dim_reduction_methods):
            #         plt.figure()
            #         # plt.subplot(1, 3, i + 1, aspect=1)

            #         # Fit the method's model
            #         model.fit(X_train, y_train)

            #         # Fit a nearest neighbor classifier on the embedded training set
            #         knn.fit(model.transform(X_train), y_train)

            #         # Compute the nearest neighbor accuracy on the embedded test set
            #         acc_knn = knn.score(model.transform(X_test), y_test)

            #         # Embed the data set in 2 dimensions using the fitted model
            #         X_embedded = model.transform(X)

            #         # Plot the projected points and show the evaluation score
            #         plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
            #         plt.title(
            #             "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
            #         )
            #     fig = plt.show()
            #     st.pyplot(fig)
        except:
            st.warning(
                "Please select a target column first or pick a dataset with a target column avaialble."
            )

        st.subheader("""
        Choose the Machine Learning Model
        """)
        model_option = st.selectbox(
            "Which machine learning model would you like to use?",
            (
                "Logistic Regression",
                "Ridge Classifier",
                "Lasso Regression",
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting Machines (GBMs)",
                "XGBoost",
                "Linear Discriminant Analysis (LDA)",
                "Support Vector Machines (SVMs)",
                "Neural Network",
            ),
            index=6,
        )
        perform_shapley = st.checkbox(
            "Attempt to Explain Model", value=False, key="perform_shapley-10"
        )
        if perform_shapley == True:
            st.warning(
                "Model explanation is computationally expensive and may not work well with all model types (like Ridge Classifier or KNN). Please be patient."
            )
        if st.button("Predict"):
            model = None
            model_explanation = ""
            shap_explainer = None
            shap_values = None
            shap_summary_plot = None
            y_scores = None
            # Model selection and fitting
            if model_option == "Logistic Regression":
                model = LogisticRegression()
                model_explanation = "Logistic regression is a statistical model commonly used in the field of medicine to predict binary outcomes. It models the probability that a given input belongs to a particular category."
                with st.expander("What is logistic regression?"):
                    from prompts import logistic_regression_text
                    st.write(logistic_regression_text)
            elif model_option == "Ridge Classifier":
                model = RidgeClassifier()
                model_explanation = "Ridge Classifier is a linear model for classification that applies L2 regularization, helping to prevent overfitting."
                with st.expander("What is Ridge Classifier?"):
                    st.write("Ridge Classifier is a linear model for classification that applies L2 regularization, helping to prevent overfitting.")
            elif model_option == "Lasso Regression":
                model = Lasso()
                model_explanation = "Lasso Regression is a linear model that uses L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection. Note: This is a regression model, not a classification model."
                with st.expander("What is Lasso Regression?"):
                    st.write("Lasso Regression is a linear model that uses L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection. It produces continuous outputs rather than binary classifications.")
            elif model_option == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier()
                model_explanation = "K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies data points based on the majority class among their k nearest neighbors."
                with st.expander("What is K-Nearest Neighbors (KNN)?"):
                    st.write("K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies data points based on the majority class among their k nearest neighbors.")
            elif model_option == "Naive Bayes":
                model = GaussianNB()
                model_explanation = "Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features. It's especially useful for categorical data."
                with st.expander("What is Naive Bayes?"):
                    st.write("Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features. It's especially useful for categorical data.")
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
                model_explanation = "A decision tree is a type of predictive model that you can think of as similar to the flowcharts sometimes used in medical decision making."
                with st.expander("What is a decision tree?"):
                    from prompts import decision_tree_text
                    st.write(decision_tree_text)
            elif model_option == "Random Forest":
                model = RandomForestClassifier()
                model_explanation = "Random Forest is a type of machine learning model that is excellent for making predictions (both binary and multiclass) and for understanding which features are most important."
                with st.expander("What is a random forest?"):
                    from prompts import random_forest_text
                    st.write(random_forest_text)
            elif model_option == "Gradient Boosting Machines (GBMs)":
                model = GradientBoostingClassifier()
                model_explanation = "Gradient Boosting Machines (GBMs) are ensemble models that build trees sequentially, each one correcting the errors of the previous."
                with st.expander("What is a gradient boosting machine?"):
                    from prompts import gbm_text
                    st.write(gbm_text)
            elif model_option == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                model_explanation = "XGBoost is a high-performance, scalable gradient boosting library that is widely used in data science competitions and industry."
                with st.expander("What is XGBoost?"):
                    st.write("XGBoost is a high-performance, scalable gradient boosting library that is widely used in data science competitions and industry.")
            elif model_option == "Linear Discriminant Analysis (LDA)":
                model = LinearDiscriminantAnalysis()
                model_explanation = "Linear Discriminant Analysis (LDA) is a classification method that projects data onto a lower-dimensional space to maximize class separability."
                with st.expander("What is Linear Discriminant Analysis (LDA)?"):
                    st.write("Linear Discriminant Analysis (LDA) is a classification method that projects data onto a lower-dimensional space to maximize class separability.")
            elif model_option == "Support Vector Machines (SVMs)":
                model = svm.SVC(probability=True)
                model_explanation = "Support Vector Machines (SVMs) are powerful classifiers that find the optimal hyperplane to separate classes in the feature space."
                with st.expander("What is a support vector machine?"):
                    from prompts import svm_text
                    st.write(svm_text)
            elif model_option == "Neural Network":
                model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu")
                model_explanation = "A neural network is a type of machine learning model inspired by the structure and function of the human brain."
                with st.expander("What is a neural network?"):
                    from prompts import neural_network_text
                    st.write(neural_network_text)

            if model is not None:
                model.fit(X_train, y_train)
                # Test set metrics only
                predictions = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_scores = model.decision_function(X_test)
                else:
                    y_scores = predictions

                st.subheader("Test Set Performance (most important)")
                is_regression = display_metrics(y_test, predictions, y_scores, set_name="Test")

                # Only show classification visualizations for classification models
                if not is_regression:
                    # Show confusion matrix
                    st.subheader("Confusion Matrix (Test Set)")
                    st.pyplot(plot_confusion_matrix(y_test, predictions))
                    with st.expander("What is a confusion matrix?"):
                        st.write("""A confusion matrix is a tool that helps visualize the performance of a predictive model in terms of classification. It's a table with four different combinations of predicted and actual values, specifically for binary classification.

The four combinations are:

1. **True Positives (TP)**: These are the cases in which we predicted yes (patients have the condition), and they do have the condition.

2. **True Negatives (TN)**: We predicted no (patients do not have the condition), and they don't have the condition.

3. **False Positives (FP)**: We predicted yes (patients have the condition), but they don't actually have the condition. Also known as "Type I error" or "False Alarm".

4. **False Negatives (FN)**: We predicted no (patients do not have the condition), and they actually do have the condition. Also known as "Type II error" or "Miss".

In the context of medicine, a false positive might mean that a test indicated a patient had a disease (like cancer), but in reality, the patient did not have the disease. This might lead to unnecessary stress and further testing for the patient. 

On the other hand, a false negative might mean that a test indicated a patient was disease-free, but in reality, the patient did have the disease. This could delay treatment and potentially worsen the patient's outcome.

A perfect test would have only true positives and true negatives (all outcomes appear in the top left and bottom right), meaning that it correctly identified all patients with and without the disease. Of course, in practice, no test is perfect, and there is often a trade-off between false positives and false negatives.

It's worth noting that a good machine learning model not only has a high accuracy (total correct predictions / total predictions) but also maintains a balance between precision (TP / (TP + FP)) and recall (TP / (TP + FN)). This is particularly important in a medical context, where both false positives and false negatives can have serious consequences. 

Lastly, when interpreting the confusion matrix, it's crucial to consider the cost associated with each type of error (false positives and false negatives) within the specific medical context. Sometimes, it's more crucial to minimize one type of error over the other. For example, with a serious disease like cancer, you might want to minimize false negatives to ensure that as few cases as possible are missed, even if it means having more false positives.
""")

                    # Show ROC curve
                    st.subheader("ROC Curve (Test Set)")
                    st.pyplot(plot_roc_curve(y_test, y_scores))
                    with st.expander("What is an ROC curve?"):
                        st.write("""
An ROC (Receiver Operating Characteristic) curve is a graph that shows the performance of a classification model at all possible thresholds, which are the points at which the model decides to classify an observation as positive or negative. 

In medical terms, you could think of this as the point at which a diagnostic test decides to classify a patient as sick or healthy.

The curve is created by plotting the True Positive Rate (TPR), also known as Sensitivity or Recall, on the y-axis and the False Positive Rate (FPR), or 1-Specificity, on the x-axis at different thresholds.

In simpler terms:

- **True Positive Rate (TPR)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify?

- **False Positive Rate (FPR)**: Out of all the actual negative cases (for example, all the patients who are really disease-free), how many did our model incorrectly identify as positive?

The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test. In other words, the bigger the area under the curve, the better the model is at distinguishing between patients with the disease and no disease.

The area under the ROC curve (AUC) is a single number summary of the overall model performance. The value can range from 0 to 1, where:

- **AUC = 0.5**: This is no better than a random guess, or flipping a coin. It's not an effective classifier.
- **AUC < 0.5**: This means the model is worse than a random guess. But, by reversing its decision, we can get AUC > 0.5.
- **AUC = 1**: The model has perfect accuracy. It perfectly separates the positive and negative cases, but this is rarely achieved in real life.

In clinical terms, an AUC of 0.8 for a test might be considered reasonably good, but it's essential to remember that the consequences of False Positives and False Negatives can be very different in a medical context, and the ROC curve and AUC don't account for this.

Therefore, while the ROC curve and AUC are very useful tools, they should be interpreted in the context of the costs and benefits of different types of errors in the specific medical scenario you are dealing with.""")

                    # Show PR curve
                    st.subheader("Precision-Recall (PR) Curve (Test Set)")
                    pr_fig = plot_pr_curve(y_test, y_scores)
                    st.pyplot(pr_fig)
                    with st.expander("What is a PR curve?"):
                        st.write("""
A Precision-Recall curve is a graph that depicts the performance of a classification model at different thresholds, similar to the ROC curve. However, it uses Precision and Recall as its measures instead of True Positive Rate and False Positive Rate.

In the context of medicine:

- **Recall (or Sensitivity)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify? It's the ability of the test to find all the positive cases.
 
- **Precision (or Positive Predictive Value)**: Out of all the positive cases that our model identified (for example, all the patients that our model thinks have the disease), how many did our model correctly identify? It's the ability of the classification model to identify only the relevant data points.

The Precision-Recall curve is especially useful when dealing with imbalanced datasets, a common problem in medical diagnosis where the number of negative cases (healthy individuals) often heavily outweighs the number of positive cases (sick individuals).

A model with perfect precision (1.0) and recall (1.0) will have a curve that reaches to the top right corner of the plot. A larger area under the curve represents both higher recall and higher precision, where higher precision relates to a low false-positive rate, and high recall relates to a low false-negative rate. High scores for both show that the classifier is returning accurate results (high precision), and returning a majority of all positive results (high recall).

The PR AUC score (Area Under the PR Curve) is used as a summary of the plot, and a higher PR AUC indicates a more predictive model.

In the clinical context, a high recall would ensure that the patients with the disease are correctly identified, while a high precision would ensure that only those patients who truly have the disease are classified as such, minimizing false-positive results.

However, there is usually a trade-off between precision and recall. Aiming for high precision might lower your recall and vice versa, depending on the threshold you set for classification. So, the Precision-Recall curve and PR AUC must be interpreted in the context of what is more important in your medical scenario: classifying all the positive cases correctly (high recall) or ensuring that the cases you classify as positive are truly positive (high precision).""")

                # Show model explanation
                with st.expander("About this model"):
                    st.write(model_explanation)

                # Show code for model training
                with st.expander("Show code for model training"):
                    st.code(
                        f"""# Model training code
model = {model.__class__.__name__}()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
                        language="python",
                    )

                # Show comparison table for metrics
                st.subheader("Model Performance Metrics")
                if is_regression:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    metrics_data = {
                        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                        "MAE": mean_absolute_error(y_test, predictions),
                        "R²": r2_score(y_test, predictions),
                    }
                else:
                    metrics_data = {
                        "F1 Score": f1_score(y_test, predictions),
                        "Accuracy": accuracy_score(y_test, predictions),
                        "ROC AUC": roc_auc_score(y_test, y_scores),
                    }
                metrics_df = pd.DataFrame(
                    [metrics_data], index=[type(model).__name__]
                )
                st.table(metrics_df)

                # Show equation for linear models
                if model_option in ["Logistic Regression", "Ridge Classifier", "Lasso Regression"]:
                    try:
                        coeff = model.coef_[0]
                        features = X_train.columns if hasattr(X_train, "columns") else [f"X{i}" for i in range(len(coeff))]
                        intercept = model.intercept_[0] if hasattr(model.intercept_, "__len__") else model.intercept_
                        equation = f"{model.__class__.__name__} Equation: y = {intercept:.3f}"
                        for c, feature in zip(coeff, features):
                            equation += f" + {c:.3f} * {feature}"
                        st.write("The equation of the model is:")
                        st.write(equation)
                    except Exception:
                        pass

                # --- Improved SHAP Implementation ---
                if perform_shapley:
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)
                    with st.spinner("Performing Shapley Analysis..."):
                        # Try to select the best SHAP explainer
                        try:
                            # Flag to control flow
                            continue_analysis = True
                            # Standardize features for KernelExplainer if needed
                            # Special handling for different model types
                            if model_option in [
                                "Decision Tree",
                                "Random Forest",
                                "Gradient Boosting Machines (GBMs)",
                                "XGBoost",
                            ]:
                                # Tree-based models can use TreeExplainer
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_test)
                                X_test_for_shap = X_test
                            elif model_option == "K-Nearest Neighbors (KNN)":
                                # For KNN, use a simpler approach with permutation importance
                                st.info("For KNN models, we'll show feature importance using permutation importance instead of SHAP values")
                                
                                from sklearn.inspection import permutation_importance
                                result = permutation_importance(
                                    model, X_test, y_test, 
                                    n_repeats=10, 
                                    random_state=42
                                )
                                
                                # Create a DataFrame for visualization
                                perm_importance_df = pd.DataFrame({
                                    'Feature': X_train.columns if hasattr(X_train, "columns") else [f"Feature {i}" for i in range(X_train.shape[1])],
                                    'Importance': result.importances_mean
                                }).sort_values('Importance', ascending=False)
                                
                                # Plot permutation importance instead of SHAP
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Importance', y='Feature', data=perm_importance_df, ax=ax)
                                ax.set_title("Feature Importance (Permutation Method)")
                                st.pyplot(fig)
                                plt.close()
                                
                                # Skip the regular SHAP plots for KNN
                                st.info("SHAP force plots are not shown for KNN models due to computational limitations")
                                # Set flag to skip the rest of the analysis
                                continue_analysis = False
                            else:
                                # For other models, use KernelExplainer with careful settings
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Use a fixed number of features and a small background dataset
                                # to avoid computational issues
                                explainer = shap.KernelExplainer(
                                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                                    shap.sample(X_train_scaled, min(50, len(X_train_scaled))),
                                    l1_reg='num_features(5)'  # Limit to top 5 features for speed and stability
                                )
                                
                                # Use fewer test samples for SHAP calculation to improve performance
                                max_samples = min(20, len(X_test_scaled))
                                shap_values = explainer.shap_values(X_test_scaled[:max_samples])
                                X_test_for_shap = X_test_scaled[:max_samples]

                            # Skip the rest of the analysis if the flag is False (for KNN)
                            if not continue_analysis:
                                # Skip to the end of the try block
                                st.warning("Skipping SHAP analysis for this model type")
                                raise Exception("Skipping SHAP analysis - not an error")
                                
                            # For binary classification, select the correct class
                            if isinstance(shap_values, list):
                                shap_values_for_class = shap_values[1]
                            else:
                                shap_values_for_class = shap_values

                            # Only show SHAP plots for non-KNN models
                            if model_option != "K-Nearest Neighbors (KNN)":
                                # Show summary plot (bar)
                                st.subheader("SHAP Feature Importance (Summary Plot)")
                                # Create a new figure for the SHAP summary plot
                                plt.figure(figsize=(10, 6))
                                # SHAP summary_plot does not support 'ax' in recent versions; use default behavior and display with st.pyplot
                                shap.summary_plot(
                                    shap_values_for_class,
                                    X_test_for_shap,  # Use the appropriate test data based on model type
                                    plot_type="bar",
                                    show=False,
                                    max_display=10  # Limit to top 10 features for clarity
                                )
                                st.pyplot(plt.gcf())
                                plt.close()  # Close the figure to ensure it doesn't affect subsequent plots

                            # Only show force plot for non-KNN models
                            if model_option != "K-Nearest Neighbors (KNN)":
                                # Show force plot for first instance
                                st.subheader("SHAP Force Plot (First Test Instance)")
                                # Handle different model types and their expected_value formats
                                if isinstance(explainer.expected_value, (list, np.ndarray)):
                                    # For binary classification with TreeExplainer (has two classes)
                                    if len(explainer.expected_value) == 2:
                                        expected_val = explainer.expected_value[1]  # Use positive class
                                    else:
                                        expected_val = explainer.expected_value[0]  # Fallback
                                else:
                                    # For KernelExplainer or single value
                                    expected_val = explainer.expected_value
                                    
                                try:
                                    # Get feature names for better visualization
                                    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"Feature {i}" for i in range(X_train.shape[1])]
                                    
                                    # Handle different shapes of SHAP values
                                    if len(shap_values_for_class.shape) == 3:  # For TreeExplainer with multi-class
                                        instance_shap_values = shap_values_for_class[0, :, 0]  # First instance, all features, first class
                                    elif len(shap_values_for_class.shape) == 2:  # For binary classification
                                        instance_shap_values = shap_values_for_class[0]  # First instance
                                    else:
                                        instance_shap_values = shap_values_for_class  # Fallback
                                    
                                    # Get the feature values for the first instance
                                    if hasattr(X_test_for_shap, "iloc"):
                                        instance_features = X_test_for_shap.iloc[0]
                                    else:
                                        instance_features = X_test_for_shap[0]
                                    
                                    # Create a temporary file to save the HTML
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                                        temp_path = tmp.name
                                    
                                    # Create the force plot with proper error handling
                                    force_plot = shap.force_plot(
                                        expected_val,
                                        instance_shap_values,
                                        instance_features,
                                        feature_names=feature_names,
                                        matplotlib=False,
                                        show=False
                                    )
                                    
                                    # Save the plot to the temporary file
                                    shap.save_html(temp_path, force_plot)
                                    
                                    # Read the HTML content and display it
                                    with open(temp_path, 'r') as f:
                                        html_content = f.read()
                                    st.components.v1.html(html_content, height=400)
                                    
                                    # Clean up the temporary file
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                                except Exception as e:
                                    st.error(f"Error generating force plot: {str(e)}")
                                    st.write("Debug info:")
                                    st.write(f"SHAP values shape: {shap_values_for_class.shape}")
                                    st.write(f"First instance shape: {(X_test.iloc[0].shape if hasattr(X_test, 'iloc') else X_test[0].shape)}")
                                    st.write(f"Feature names: {feature_names[:5]}...")  # Show first 5 feature names
                        except Exception as e:
                            if str(e) == "Skipping SHAP analysis - not an error":
                                pass  # This is our controlled exit for KNN models
                            else:
                                st.warning(f"Could not generate SHAP plots: {e}")
            # End of if model is not None


with tab3:
    if hu_key == "True" or check_password():
        # Add max iterations slider to sidebar
        with st.sidebar:
            st.markdown("<div class='step-header'>GPT Analysis Settings</div>", unsafe_allow_html=True)
            max_iterations = st.slider(
                "Maximum iterations for GPT analysis", 
                min_value=1, 
                max_value=10, 
                value=5, 
                help="Maximum number of times GPT will refine its answer to ensure accuracy"
            )
        
        # Use container to ensure proper scrolling
        container = st.container()
        with container:
            st.markdown("<h1 style='color: #1E88E5;'>🤖 Analyze with GPT (AutoAnalyzer AI)</h1>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background-color: #E3F2FD; padding: 15px; border-radius: 5px; border-left: 5px solid #1E88E5;">
                <h3 style="margin-top: 0; color: #1976D2;">AI-Powered Data Analysis</h3>
                <p>Ask any question about your data in plain English. The AI will generate and execute Python code using your dataframe as <code>df</code>. Results and plots will appear below.</p>
                <p>The AI will iterate up to <strong>""" + str(max_iterations) + """</strong> times to refine its answer if needed.</p>
                <p><strong>Example questions:</strong></p>
                <ul>
                    <li>Show me the relationship between age and blood pressure with a regression line</li>
                    <li>Create a heatmap of correlations between all numerical variables</li>
                    <li>What's the average BMI by gender? Show it as a bar chart</li>
                    <li>Is there a significant difference in cholesterol levels between diabetic and non-diabetic patients?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # File uploader
        data_source = st.radio(
            "Select data source",
            ["Use existing dataframe", "Upload a new file"],
            horizontal=True,
        )

        if data_source == "Upload a new file":
            uploaded_file_gpt = st.file_uploader(
                "Use existing dataframe or upload a new Excel or CSV file",
                type=["csv", "xlsx"],
            )

            if uploaded_file_gpt is None:
                st.warning("Please upload a file to continue.")
            else:
                if uploaded_file_gpt.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file_gpt)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file_gpt)

        # Display the current dataframe being used for analysis
        # This could be the initially loaded df or the gpt_working_df from a previous run
        current_analysis_df = st.session_state.get("gpt_working_df", st.session_state.df)

        n_rows, n_cols = current_analysis_df.shape
        st.write(f"Current DataFrame shape: {n_rows} rows × {n_cols} columns")

        with st.expander("View the current dataframe", expanded=True):
            st.write("### Current Data Frame")
            st.dataframe(current_analysis_df, height=200)

        import matplotlib
        matplotlib.use("Agg")  # Ensure non-GUI backend for matplotlib

        from langchain_openai import AzureChatOpenAI
        from langchain_experimental.utilities import PythonREPL

        # Set up the LLM (Azure)
        llm = AzureChatOpenAI(
            azure_deployment=st.secrets["azure_deployment"],
            api_version=st.secrets["api_version"],
            azure_endpoint=openai_base_url,
            api_key=openai_api_key,
            temperature=0.0,
            max_tokens=4000,
            timeout=None,
            max_retries=2,
            model_kwargs={
                "seed": 42,
            },
        )

        # Session state for code, images, and output
        if "gpt_analysis_code" not in st.session_state:
            st.session_state.gpt_analysis_code = ""
        if "gpt_analysis_images" not in st.session_state:
            st.session_state.gpt_analysis_images = []
        if "model_output1" not in st.session_state:
            st.session_state.model_output1 = ""

        st.markdown("<h3 style='color: #1976D2; margin-top: 20px;'>Ask a Question About Your Data!</h3>", unsafe_allow_html=True)
        agent_question = st.text_area(
            "Type your question in plain English:",
            placeholder="Example: Create a boxplot comparing blood glucose levels between diabetic and non-diabetic patients",
            height=100,
            key="agent_question_input" # Add a key to manage state
        )

        # Import necessary modules at the top level to ensure they're available
        import re
        import io
        import sys
        import glob
        import os
        import traceback
        import time
        from contextlib import redirect_stdout
        
        # Initialize persistent storage if needed
        if "persistent_gpt_code" not in st.session_state:
            st.session_state.persistent_gpt_code = {}
        if "persistent_gpt_output" not in st.session_state:
            st.session_state.persistent_gpt_output = {}
        if "persistent_gpt_images" not in st.session_state:
            st.session_state.persistent_gpt_images = {}
        if "iteration_history" not in st.session_state:
            st.session_state.iteration_history = {}
        if "last_agent_question" not in st.session_state: # Initialize session state for the question
            st.session_state.last_agent_question = ""
            
        if st.button("🚀 Analyze My Data", use_container_width=True):
            # Ensure a dataframe is loaded before proceeding
            if st.session_state.df.empty:
                st.warning("Please load a dataframe first.")
                # Stop execution if no dataframe is loaded
                # The rest of the code in this block will not run if df is empty

            # Set up the REPL for code execution *inside* the button click
            # Only proceed if the dataframe is not empty
            if not st.session_state.df.empty:
                repl = PythonREPL()
                # Initialize REPL globals with the current dataframe and common libraries
                repl.globals.update({
                    "df": st.session_state.df.copy(), # Use a copy of the loaded df as the initial working df
                    "original_df": st.session_state.df, # Keep a reference to the original df
                    "plt": plt,
                    "sns": sns,
                    "np": np,
                    "pd": pd,
                })
                # Store the initial working df in session state
                st.session_state.gpt_working_df = st.session_state.df.copy()

                # Store the current question in session state
                st.session_state.last_agent_question = agent_question
                
                # Remove any prior plot images in the output directory
                image_exts = ["png", "jpg", "jpeg", "svg", "pdf"]
                outputs_path = st.session_state.outputs_path
            for ext in image_exts:
                for img_path in glob.glob(f"{outputs_path}/*.{ext}"):
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass

            st.session_state.gpt_analysis_code = ""
            st.session_state.gpt_analysis_images = []
            st.session_state.model_output1 = ""
            
            # Reset categorical mappings and working dataframe for new analysis
            st.session_state.categorical_mappings = {}
            if "gpt_working_df" in st.session_state:
                st.session_state.gpt_working_df = st.session_state.df.copy()
            
            # Create a progress bar for iterations
            progress_bar = st.progress(0)
            iteration_status = st.empty()

            # Only allow English language questions, not direct Python code
            def get_code_from_llm(question, df, iteration=1, previous_code="", previous_output="", previous_error=None):
                col_list = list(df.columns)
                
                # Base prompt for first iteration
                if iteration == 1:
                    prompt = f"""
You are an expert Python data analyst. The user has provided a pandas dataframe called `df` and asked the following question:

{question}

The dataframe columns are: {col_list}

You have access to two dataframes:
1. `df` - A working copy that you can modify as needed for your analysis
2. `original_df` - The original unmodified dataframe (read-only reference)

If the user refers to a column name in a different case (e.g., 'glucose' instead of 'Glucose'), always match it to the correct column name in the dataframe, ignoring case. For example, if the user says 'glucose', use 'Glucose' if that is the actual column name.

Before performing any analysis that requires numeric data (such as correlation heatmaps, PCA, or regression), always check for categorical columns (object dtype or string values). 
- If a categorical column has exactly 2 unique values, convert it to numeric by mapping the most common value to 0 and the least common value to 1. Use the `safe_map_categorical()` function for this conversion and print a message indicating which columns were converted and how.
- If a categorical column has more than 2 unique values, use one-hot encoding (e.g., `pd.get_dummies(df, columns=[col])`) to create additional columns as needed, and print a message indicating which columns were one-hot encoded.
- Always check for and handle NaN values in categorical columns before mapping or encoding.
Do this as a first step in your code if needed.

**Important:** The unique values for categorical columns in the current `df` are printed in the previous output/history for your reference. Use this information to correctly identify and handle categorical values.

At the top of your code, always include:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

Write Python code to answer the question. 
- Feel free to modify the `df` dataframe as needed (filter, transform, etc.)
- If you need to reference the original unmodified data, use `original_df`
- If a plot is needed, save it to '{st.session_state.outputs_path}/gpt_plot_{iteration}.png' using plt.savefig and then call plt.close().
- Do not use plt.show().
- Do not print explanations, only print results or tables.
- Do not return any text or explanation, only the code.
- If the question is ambiguous, make reasonable assumptions and proceed.
- If the question is not answerable, raise an Exception with a helpful message.
Return only the code, nothing else.
Respond ONLY with valid Python code, not with natural language or explanations.
"""
                # Refinement prompt for subsequent iterations
                else:
                    error_info = f"\nThe previous code generated this error: {previous_error}" if previous_error else ""
                    
                    prompt = f"""
You are an expert Python data analyst. The user has provided a pandas dataframe called `df` and asked the following question:

{question}

The dataframe columns are: {col_list}

You have access to two dataframes:
1. `df` - This is your working copy of the dataframe. It reflects any modifications made in previous steps of this analysis. You can modify this dataframe as needed (filter, transform, create new columns, etc.).
2. `original_df` - This is the initial, unmodified dataframe (read-only reference). Use this if you need to start a calculation from the original data state.

This is iteration {iteration} of your analysis. You previously wrote this code:

```python
{previous_code}
```

When executed, it produced this output:
```
{previous_output}
```
{error_info}

Your task is to improve the code to better answer the user's question. Consider:
1. Is the output correct and complete?
2. Does it fully answer the user's question?
3. Are there any errors or issues to fix?
4. Could the visualization be improved?
5. Is there additional analysis that would help answer the question?

At the top of your code, always include:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

Write improved Python code to better answer the question.
- Feel free to modify the `df` dataframe as needed (filter, transform, etc.). These modifications will persist for subsequent iterations.
- If you need to reference the original unmodified data, use `original_df`.
- If a plot is needed, save it to '{st.session_state.outputs_path}/gpt_plot_{iteration}.png' using plt.savefig and then call plt.close().
- Do not use plt.show().
- Do not print explanations, only print results or tables.
- If the question is not answerable, raise an Exception with a helpful message.
- When converting categorical variables to numeric, use the `safe_map_categorical()` function and clearly document the mapping.
- Always check for and handle NaN values in categorical columns before mapping or encoding.

**Important:** The unique values for categorical columns in the current `df` are printed in the previous output/history for your reference. Use this information to correctly identify and handle categorical values.

Return only the improved code, nothing else.
Respond ONLY with valid Python code, not with natural language or explanations.
"""

                with st.spinner(f"Generating code (iteration {iteration}/{max_iterations})..."):
                    response = llm.invoke(prompt)
                code = response.content if hasattr(response, "content") else str(response)
                code = re.sub(r"^```python|^```|```$", "", code, flags=re.MULTILINE).strip()
                # If the code does not look like Python, return an empty string to avoid exec errors
                if not any(x in code for x in ("import ", "plt.", "sns.", "df.", "pd.")):
                    return ""
                return code

            # Run the code and capture output and plots
            def run_code_and_capture(code):
                f = io.StringIO()
                images_before = set(glob.glob(f"{st.session_state.outputs_path}/*.png"))
                
                # Ensure the working copy of the dataframe exists and is used
                if "gpt_working_df" not in st.session_state:
                    st.session_state.gpt_working_df = st.session_state.df.copy()

                # Ensure the REPL globals are updated with the current working df state
                # This is crucial for subsequent iterations
                repl.globals.update({
                    "df": st.session_state.gpt_working_df,
                    "original_df": st.session_state.df, # Ensure original_df is always the initial df
                    # Other globals (plt, sns, np, pd) are already set during repl initialization
                })
                
                # Initialize a dictionary to store categorical variable mappings
                if "categorical_mappings" not in st.session_state:
                    st.session_state.categorical_mappings = {}
                
                # Store the original map method at function scope
                original_map = pd.Series.map
                
                # Add code to track categorical variable mappings and print unique values
                tracking_code = """
# Track categorical variable mappings
categorical_mappings = {}

# Original DataFrame columns
original_columns = df.columns.tolist()
original_dtypes = df.dtypes.to_dict()

def track_categorical_mapping(df_col, mapping):
    if isinstance(df_col, str):
        categorical_mappings[df_col] = mapping
    
# Monkey patch pandas Series map method to capture mappings
original_map_inner = pd.Series.map
def map_with_tracking(self, arg, *args, **kwargs):
    if isinstance(arg, dict):
        track_categorical_mapping(self.name, arg)
    # Use the original map method but handle NaN values safely
    result = original_map_inner(self, arg, *args, **kwargs)
    return result
pd.Series.map = map_with_tracking

# Print unique values for categorical columns for LLM reference
print("\\n--- Unique Categorical Values ---")
for col in df.select_dtypes(include=['object', 'category']).columns:
    try:
        unique_vals = df[col].unique().tolist()
        # Check for NaN values
        has_nan = any(pd.isna(val) for val in unique_vals)
        if has_nan:
            print(f"Column '{col}': {unique_vals} (contains NaN values)")
        else:
            print(f"Column '{col}': {unique_vals}")
    except Exception as e:
        print(f"Could not get unique values for column '{col}': {e}")
print("---------------------------------")
"""
                
                # Add safe mapping helper function
                safe_mapping_code = """
# Helper function for safe categorical mapping
def safe_map_categorical(series, mapping, default=None):
    '''
    Safely map categorical values, handling NaN values and unknown categories.
    
    Args:
        series: pandas Series to map
        mapping: dictionary mapping values to new values
        default: value to use for categories not in mapping (None = keep original)
    
    Returns:
        Mapped pandas Series
    '''
    # Create a copy to avoid modifying the original
    result = series.copy()
    
    # Handle each value
    for i, val in enumerate(result):
        if pd.isna(val):
            # Keep NaN values as NaN
            continue
        elif val in mapping:
            # Apply mapping for known values
            result.iloc[i] = mapping[val]
        elif default is not None:
            # Use default for unknown values if provided
            result.iloc[i] = default
    
    # Track the mapping
    if isinstance(series.name, str):
        categorical_mappings[series.name] = mapping
        
    return result
"""

                # Prepend the tracking code and safe mapping helper to the user's code
                full_code = tracking_code + "\n" + safe_mapping_code + "\n" + code
                
                try:
                    with redirect_stdout(f):
                        exec(full_code, repl.globals)
                    output = f.getvalue()
                    error = None

                    # Update the working dataframe with the state after execution
                    # Only update if execution was successful
                    if 'df' in repl.globals:
                        st.session_state.gpt_working_df = repl.globals['df'].copy()

                    # Capture any categorical mappings that were created
                    if 'categorical_mappings' in repl.globals:
                        mappings = repl.globals['categorical_mappings']
                        if mappings:
                            # Store mappings in session state
                            if "categorical_mappings" not in st.session_state:
                                st.session_state.categorical_mappings = {}
                            st.session_state.categorical_mappings.update(mappings)

                            # Add mapping information to the output
                            mapping_output = "\n\n--- Categorical Variable Encodings ---\n"
                            for col, mapping in mappings.items():
                                mapping_output += f"\nColumn '{col}' encoded as:\n"
                                for original, encoded in mapping.items():
                                    mapping_output += f"  {original} → {encoded}\n"

                            output += mapping_output

                    # Restore original map method (using the function-scope variable)
                    pd.Series.map = original_map

                except Exception as e:
                    output = f.getvalue() + "\n" + traceback.format_exc()
                    error = str(e)
                    # Restore original map method in case of error
                    pd.Series.map = original_map
                    # Do NOT update st.session_state.gpt_working_df if there was an error
                    # This preserves the state from the last successful iteration
                
                images_after = set(glob.glob(f"{st.session_state.outputs_path}/*.png"))
                new_images = list(images_after - images_before)
                new_images = sorted(new_images, key=os.path.getmtime)
                if not new_images:
                    all_pngs = sorted(glob.glob(f"{st.session_state.outputs_path}/*.png"), key=os.path.getmtime)
                    if all_pngs and len(images_before) < len(images_after):
                        new_images = [all_pngs[-1]]
                # Limit image size to ensure scrolling works
                for img_path in new_images:
                    try:
                        img = Image.open(img_path)
                        if img.height > 800:  # Limit height of large images
                            img_ratio = img.width / img.height
                            new_height = 800
                            new_width = int(new_height * img_ratio)
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                            img.save(img_path)
                    except Exception:
                        pass
                return output, error, new_images

            # Iterative analysis process
            timestamp = str(int(time.time()))
            st.session_state.current_analysis_timestamp = timestamp
            st.session_state.iteration_history[timestamp] = []
            
            final_code = ""
            final_output = ""
            final_error = None
            final_images = []
            
            # Initialize variables for the iterative process
            code_to_run = ""
            output = ""
            error = None
            new_images = []
            
            # Make sure we have a clean working dataframe at the start
            if "gpt_working_df" not in st.session_state or st.session_state.gpt_working_df is None:
                st.session_state.gpt_working_df = st.session_state.df.copy()
            
            # Run up to max_iterations
            for iteration in range(1, max_iterations + 1):
                # Update progress bar
                progress_bar.progress(iteration / (max_iterations + 1))
                iteration_status.info(f"Running iteration {iteration}/{max_iterations}...")
                
                # Generate code based on previous results
                if iteration == 1:
                    code_to_run = get_code_from_llm(agent_question, st.session_state.gpt_working_df, iteration)
                else:
                    code_to_run = get_code_from_llm(agent_question, st.session_state.gpt_working_df, iteration, 
                                                   previous_code=final_code, 
                                                   previous_output=final_output,
                                                   previous_error=final_error)
                
                # Run the code
                output, error, new_images = run_code_and_capture(code_to_run)
                
                # Store this iteration's results
                iteration_result = {
                    "iteration": iteration,
                    "code": code_to_run,
                    "output": output,
                    "error": error,
                    "images": new_images.copy()
                }
                st.session_state.iteration_history[timestamp].append(iteration_result)
                
                # Update final results
                final_code = code_to_run
                final_output = output
                final_error = error
                final_images = new_images
                
                # If there's an error but we haven't reached max iterations, continue to next iteration
                # to let the model fix the error
                if error and iteration < max_iterations:
                    iteration_status.warning(f"Error in iteration {iteration}, attempting to fix in next iteration...")
                    continue
                
                # If there's no error and we've reached max iterations, or if we're at the last iteration
                if not error and iteration < max_iterations:
                    # Ask LLM if the answer is complete
                    completion_check_prompt = f"""
You are an expert data analyst evaluating code execution results. 
The user asked: "{agent_question}"

The code:
```python
{code_to_run}
```

Produced this output:
```
{output}
```

Does this completely and correctly answer the user's question? Answer with ONLY "YES" if the answer is complete and correct, or "NO" if further iterations could improve the answer.
"""
                    with st.spinner(f"Evaluating completeness of answer (iteration {iteration}/{max_iterations})..."):
                        completion_response = llm.invoke(completion_check_prompt)
                    completion_answer = completion_response.content if hasattr(completion_response, "content") else str(completion_response)
                    
                    # If the answer is complete, break the loop
                    if "YES" in completion_answer.upper() and not "NO" in completion_answer.upper():
                        iteration_status.success(f"Answer complete after {iteration} iterations!")
                        break
                
                # If this is the last iteration, use what we have
                if iteration == max_iterations:
                    if error:
                        iteration_status.warning(f"Completed all {iteration} iterations with errors in the final iteration")
                    else:
                        iteration_status.info(f"Completed all {iteration} iterations")
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            
            # If the final result has an error, try to find the last successful iteration
            if final_error and len(st.session_state.iteration_history[timestamp]) > 1:
                # Look for the most recent iteration without errors
                for i in range(len(st.session_state.iteration_history[timestamp])-2, -1, -1):
                    prev_iteration = st.session_state.iteration_history[timestamp][i]
                    if not prev_iteration["error"]:
                        # Use this iteration's results instead
                        final_code = prev_iteration["code"]
                        final_output = prev_iteration["output"]
                        final_images = prev_iteration["images"]
                        st.warning(f"Using results from iteration {i+1} because the final iteration had errors.")
                        
                        # Also restore the working dataframe state from that iteration if possible
                        if i > 0 and "gpt_working_df" in st.session_state:
                            # We need to re-run the code up to this point to get the correct dataframe state
                            temp_repl = PythonREPL()
                            temp_repl.globals.update({
                                "df": st.session_state.df.copy(),
                                "original_df": st.session_state.df,
                                "plt": plt,
                                "sns": sns,
                                "np": np,
                                "pd": pd,
                            })
                            
                            # Run all successful iterations up to this point
                            for j in range(i+1):
                                iter_code = st.session_state.iteration_history[timestamp][j]["code"]
                                try:
                                    exec(iter_code, temp_repl.globals)
                                    # If this was the last successful iteration, save its dataframe
                                    if j == i:
                                        st.session_state.gpt_working_df = temp_repl.globals["df"].copy()
                                except Exception:
                                    # If any error occurs, stop trying to restore the dataframe
                                    break
                        break
            
            # Generate a summary of the findings for busy researchers
            with st.spinner("Generating research summary..."):
                # Create a prompt for the summary
                summary_prompt = f"""
You are an expert data analyst summarizing findings for a busy researcher. The user asked the following question:

{agent_question}

The analysis produced this output:
```
{final_output}
```

The code that generated this analysis is:
```python
{final_code}
```

The analysis generated {len(final_images)} visualizations. Based on the code, these visualizations include:
{', '.join([f"'{os.path.basename(img)}'" for img in final_images]) if final_images else "No visualizations were generated"}

Please provide:
1. A clear, concise summary of the key findings (3-5 bullet points)
2. A brief explanation of what the visualizations show and how they can be utilized (be specific about which plots were created based on the code)
3. Any important limitations or caveats to consider

Your summary should be written in professional academic language suitable for a busy researcher.
"""
                summary_response = llm.invoke(summary_prompt)
                research_summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
                
                # Store the summary in session state
                st.session_state.research_summary = research_summary
            
            # Save the final results to session state
            st.session_state.gpt_analysis_code = final_code
            st.session_state.model_output1 = final_output
            st.session_state.gpt_analysis_images = final_images
            
            # Store in persistent storage
            st.session_state.persistent_gpt_code[timestamp] = final_code
            st.session_state.persistent_gpt_output[timestamp] = final_output
            st.session_state.persistent_gpt_images[timestamp] = final_images
            st.session_state.persistent_research_summary = research_summary

            if final_error:
                st.error(f"Error in final code: {final_error}")
                st.info("The system has attempted to recover by using the last successful iteration's results. You can view the iteration history to see all attempts.")
            
        # Display results if they exist in session state
        if st.session_state.model_output1 or st.session_state.gpt_analysis_code:
            # Get the current timestamp
            timestamp = st.session_state.get("current_analysis_timestamp", "")
            
            # Display the user's question
            if st.session_state.last_agent_question:
                st.markdown(f"## Your Question:")
                st.write(st.session_state.last_agent_question)
                st.markdown("---")

            # Display research summary if available
            if hasattr(st.session_state, 'research_summary') and st.session_state.research_summary:
                st.markdown("## 📋 Research Summary")
                st.markdown(st.session_state.research_summary)
                st.markdown("---")
                
            # Display output if available
            if st.session_state.model_output1:
                st.write("**Detailed Output:**")
                st.code(st.session_state.model_output1)
                
            # Check if the working dataframe was modified and is different from the original
            if "gpt_working_df" in st.session_state and st.session_state.gpt_working_df is not None:
                try:
                    # Check if dataframes are different (ignoring index)
                    original_df_reset = st.session_state.df.reset_index(drop=True)
                    working_df_reset = st.session_state.gpt_working_df.reset_index(drop=True)
                    
                    # Compare columns first to avoid errors with different column sets
                    columns_same = set(original_df_reset.columns) == set(working_df_reset.columns)
                    
                    if not columns_same or not working_df_reset.equals(original_df_reset):
                        with st.expander("View modified dataframe", expanded=False):
                            st.write("The analysis modified the dataframe. Here's the result:")
                            st.dataframe(st.session_state.gpt_working_df)
                                
                            # Add option to use this as the new dataframe
                            if st.button("Use this modified dataframe for future analyses"):
                                st.session_state.modified_df = st.session_state.gpt_working_df.copy()
                                st.success("Modified dataframe saved! Select 'Modified Dataframe' in the sidebar to use it.")
                except Exception as e:
                    st.warning(f"Could not compare dataframes: {e}")
                
            # Display code if available
            if st.session_state.gpt_analysis_code:
                with st.expander("Show code used for this analysis", expanded=False):
                    st.code(st.session_state.gpt_analysis_code, language="python")
            
            # Show iteration history if available
            if timestamp in st.session_state.iteration_history and len(st.session_state.iteration_history[timestamp]) > 1:
                with st.expander("Show iteration history", expanded=False):
                    for i, iteration in enumerate(st.session_state.iteration_history[timestamp]):
                        st.subheader(f"Iteration {i+1}")
                        
                        # Show if there was an error
                        if iteration["error"]:
                            st.error(f"Error: {iteration['error']}")
                        
                        # Show code
                        st.write("Code:")
                        st.code(iteration["code"], language="python")
                        
                        # Show output
                        st.write("Output:")
                        st.code(iteration["output"])
                        
                        # Show images if any
                        if iteration["images"]:
                            st.write("Generated images:")
                            for img_path in iteration["images"]:
                                if os.path.exists(img_path):
                                    try:
                                        st.image(img_path, 
                                                caption=f"Iteration {i+1}: {os.path.basename(img_path)}", 
                                                use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not display image {img_path}: {e}")
                        
                        st.markdown("---")
            
            # Display images if available
            shown = set()
            images_to_show = []
            
            # First try to get images from persistent storage
            if timestamp and timestamp in st.session_state.persistent_gpt_images:
                images_to_show = st.session_state.persistent_gpt_images[timestamp]
            # Fallback to session state images
            elif st.session_state.gpt_analysis_images:
                images_to_show = st.session_state.gpt_analysis_images
                
            for img_path in images_to_show:
                if img_path not in shown and os.path.exists(img_path):
                    try:
                        st.image(img_path, 
                                caption=f"Generated Plot: {os.path.basename(img_path)}", 
                                use_column_width=True)
                        shown.add(img_path)
                    except Exception as e:
                        st.warning(f"Could not display image {img_path}: {e}")
            # Add some spacing to improve layout
            st.write("")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Generate Word Doc from Last GPT Analysis", use_container_width=True):
                    try:
                        # Gather all relevant content for the docx
                        timestamp = st.session_state.get("current_analysis_timestamp", "")
                        # Question - Retrieve from session state
                        question_for_doc = st.session_state.get("last_agent_question", "")
                        # Research summary
                        if hasattr(st.session_state, 'persistent_research_summary'):
                            research_summary = st.session_state.persistent_research_summary
                        elif hasattr(st.session_state, 'research_summary'):
                            research_summary = st.session_state.research_summary
                        else:
                            research_summary = ""
                        # Output
                        if timestamp and timestamp in st.session_state.persistent_gpt_output:
                            output_for_doc = st.session_state.persistent_gpt_output[timestamp]
                        else:
                            output_for_doc = st.session_state.model_output1
                        # Code
                        if timestamp and timestamp in st.session_state.persistent_gpt_code:
                            code_for_doc = st.session_state.persistent_gpt_code[timestamp]
                        else:
                            code_for_doc = st.session_state.gpt_analysis_code
                        # Images
                        if timestamp and timestamp in st.session_state.persistent_gpt_images:
                            images_for_doc = st.session_state.persistent_gpt_images[timestamp]
                        else:
                            images_for_doc = st.session_state.gpt_analysis_images
                        # Categorical mappings
                        categorical_mappings = getattr(st.session_state, "categorical_mappings", None)

                        with st.spinner("Creating Word document..."):
                            docx_file = generate_gpt_analysis_docx(
                                "gpt_analysis",
                                question_for_doc, # Use the question from session state
                                research_summary,
                                code_for_doc,
                                output_for_doc,
                                images_for_doc,
                                categorical_mappings,
                            )

                        if docx_file and os.path.exists(docx_file):
                            with open(docx_file, "rb") as file:
                                btn = st.download_button(
                                    label="Download DOCX",
                                    data=file,
                                    file_name="gpt_analysis.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                )
                            st.success("Word document created successfully!")
                            try:
                                os.remove(docx_file)
                            except Exception as e:
                                st.warning(f"Could not remove temporary file: {e}")
                        else:
                            st.error("Failed to create Word document. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred while creating the DOCX file: {str(e)}")
