csv_prefix_gpt4 = """You are an AI assistant designed to analyze data to answer user questions and show your work. 

Overall process:
1. Use the full dataframe (df) provided to answer the user's question comprehensively.
2. After answering the user question, generate up to 5 python code snippets for illustrative plots to complement the answer.

Detailed process descriptions:
1. Use the full dataframe variable (df) provided for the analysis. Your goal is to accurately and comprehensively anticipate what the user likely wants to learn from the dataframe. Provide specific and informative answers, not how to get the answers. 
Ensure your terminal outputs include all columns so no data is missing from analysis by using pandas commands that explicitly display all output columns. 
Correctly apply the following code snippet in your analyses:
        # Adjust display options
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being split across lines
2. Generate up to 5 code snippets within a JSON object to allow the user to see complementary data plots for the initial answer. 
- Identify likely key output labels and visualize across groups or categories if possible to highlight trends or relationships.
- Each snippet should be fully complete, including necessary imports. Variable definitions from your analysis should be recreated if needed since they will not pass automatically.
- Add trend lines when they are potentially helpful to a plot.
- Prevent execution errors. For correlations or heatmaps identify each categorical column and convert each to numerical or drop the column if non-binary.
- Generate plots that can be displayed directly in Streamlit without saving to a file.
- Follow PEP8 guidelines for code formatting.
- Use libraries like matplotlib, seaborn, or plotly for visualization.
---
Example code snippet:
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fig, ax = plt.subplots()
sns.boxplot(x='Diabetes', y='Systolic BP', data=df, ax=ax)
ax.set_title('Systolic Blood Pressure by Diabetes Status')
ax.set_xlabel('Diabetes Status')
ax.set_ylabel('Systolic Blood Pressure (mmHg)')
st.pyplot(fig)
---
- Format the JSON object for the code snippets:
{
  "code_snippets": [
    {
      "description": "A brief description of what this plot shows",
      "code": "Python code as a string that generates a plot"
    },
    {
      "description": "Description of second plot (if applicable)",
      "code": "Python code for second plot (if applicable)"
    },
    {
      "description": "Description of third plot (if applicable)",
      "code": "Python code for third plot (if applicable)"
    }
  ]
}

"""

data_analysis_prompt = """You are an AI assistant designed to analyze data to answer user questions and show your work.

Include every row and column in the dataframe (df) provided to answer the user's question comprehensively. Your goal is to accurately and anticipate what the user likely wants to learn from the dataframe. Provide specific and informative answers, not how to get the answers.

Ensure your terminal outputs include all columns so no data is missing from analysis by using pandas commands that explicitly display all output columns. 

Correctly apply the following code snippet in your analyses:
    # Adjust display options
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being split across lines

Provide a detailed analysis of the data, including relevant statistics, trends, and insights that address the user's question. Be thorough and explanatory in your response.
"""

plot_generation_prompt = """You are an AI assistant designed to generate, display, and run code for visualizations to complement data analysis of a provided dataframe variable. 

Generate, display, and execute up to 5 code snippets to allow the user to see illustrative data plots inside the Streamlit app to answer the user's question. Follow these guidelines:

- Each snippet should be fully complete, including necessary imports. Variable definitions from your analysis should be redefined if needed since they will not pass automatically.
- Prevent correlation execution errors by converting each categorical datafram column to a float (use 1 for least frequent finding) only when needed for the specific snippet analysis or dropping if conversion is not possible. 
- Plots should help users visualize across groups or categories if possible to highlight trends or relationships.
- Add trend lines when they are helpful to a plot.
- A shown below, use code to generate plots for display in the Streamlit app.
- Follow PEP8 guidelines for code formatting.
- Use libraries like matplotlib, seaborn, or plotly for visualization.

Example code snippet to execute; no need to load a CSV file, the dataframe is already provided:
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fig, ax = plt.subplots()
sns.boxplot(x='Diabetes', y='Systolic BP', data=df, ax=ax)
ax.set_title('Systolic Blood Pressure by Diabetes Status')
ax.set_xlabel('Diabetes Status')
ax.set_ylabel('Systolic Blood Pressure (mmHg)')
st.pyplot(fig)

"""

quick_analysis_prompt = """If a text answer is required, perform a careful analysis of the data to answer the user's question. If a plot is required, 
generate and execute code for plots to display in the Streamlit app. Code should be complete, including necessary imports. Variable definitions from any prior analysis should be redefined 
if needed since they will not pass automatically. After code is executed once, no need to run it again or display the code.
Example code snippet to execute; no need to load a CSV file, the dataframe is already provided:
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fig, ax = plt.subplots()
sns.boxplot(x='Diabetes', y='Systolic BP', data=df, ax=ax)
ax.set_title('Systolic Blood Pressure by Diabetes Status')
ax.set_xlabel('Diabetes Status')
ax.set_ylabel('Systolic Blood Pressure (mmHg)')
st.pyplot(fig)

"""


prefix_teacher = """You politely decline to answer questions outside the domains of data science, statistics, and medicine. 
If the question is appropriate, you teach for students at all levels. Your response appears next to a web  
tool that can generate bar charts, violin charts, histograms, pie charts, scatterplots, and summary statistics for  sample datasets or a user supplied CSV file.         
"""
# Explanatory text blocks for use in the app

mult_linear_reg_text = """
Multiple linear regression is a statistical technique used to model the relationship between one dependent variable and two or more independent variables. The goal is to determine the best-fitting linear equation that describes how the dependent variable changes as the independent variables change.

In a medical research setting, multiple linear regression can be used in various ways. For example:
1. Predicting health outcomes: Multiple linear regression can be used to predict a patient's health outcome (e.g., blood pressure, cholesterol level) based on several risk factors (e.g., age, weight, smoking status).
2. Identifying risk factors: Multiple linear regression can be used to identify the independent variables that are most strongly associated with a particular health outcome.

The regression equation takes the form:
y = β0 + β1x1 + β2x2 + ... + βnxn + ε

Where:
- y is the dependent variable (outcome)
- x1, x2, ..., xn are the independent variables (predictors)
- β0 is the intercept
- β1, β2, ..., βn are the coefficients for each predictor
- ε is the error term

The coefficients represent the expected change in the dependent variable for a one-unit change in the predictor, holding all other predictors constant.
"""

cox_text = """
Cox Proportional Hazards analysis, also known as Cox regression, is a statistical method used to investigate the effect of several variables on the time a specified event takes to happen. It is commonly used in medical research for survival analysis.

Key points:
1. The Cox model estimates the hazard (risk) of an event occurring, such as death or relapse, at a particular time, given certain predictor variables.
2. The model assumes that the hazard ratios are constant over time (the proportional hazards assumption).
3. The output includes hazard ratios for each variable, which indicate how the risk of the event changes with a one-unit increase in the variable.

Interpretation:
- A hazard ratio greater than 1 indicates increased risk.
- A hazard ratio less than 1 indicates decreased risk.
- A hazard ratio equal to 1 indicates no effect.

Cox regression is valuable for identifying risk factors and adjusting for confounding variables in time-to-event data.
"""

kaplan_meier_text = """
The Kaplan-Meier survival curve is a graphical representation of the probability of surviving over time, often used in medical research to estimate patient survival rates.

Key points:
1. The curve shows the proportion of patients surviving at each time point after a treatment or diagnosis.
2. Each step down in the curve represents an event (e.g., death, relapse).
3. The curve can be used to compare survival between different groups (e.g., treatment vs. control).

Interpretation:
- The y-axis shows the probability of survival.
- The x-axis shows time.
- The curve provides a visual summary of survival data and can help identify differences between groups.
"""

correlation_heatmap_text = """
A correlation heatmap is a graphical representation of the correlation matrix, which is a table showing correlation coefficients between sets of variables. Each cell in the table shows the correlation between two variables. In the heatmap, correlation coefficients are color-coded, where the intensity of the color represents the magnitude of the correlation coefficient.

Key points:
- Red signifies a high positive correlation (variables move in the same direction).
- Blue represents negative correlation (variables move in opposite directions).
- The correlation values appear in each square, giving a precise numeric correlation coefficient along with the visualized color intensity.

Why are correlation heatmaps useful?
- They help determine the relationship between different variables.
- In medicine, this can help identify risk factors for diseases, where variables could be different health indicators like age, cholesterol level, blood pressure, etc.

Understanding correlation values:
- Correlation coefficients range from -1 to 1:
  - 1: perfect positive correlation
  - -1: perfect negative correlation
  - 0: no linear relationship

Note: Correlation does not imply causation. Correlation heatmaps are based on linear relationships; non-linear relationships may not be captured.
"""

box_plot_text = """
Box plots (also known as box-and-whisker plots) are a great way to visually represent the distribution of data. They're particularly useful when you want to compare distributions between several groups.

Components of a box plot:
1. Box: Represents the interquartile range (IQR), containing the middle 50% of the data.
2. Median: The line inside the box shows the median (50th percentile).
3. Whiskers: Lines extending from the box indicate variability outside the IQR.
4. Outliers: Points beyond the whiskers are considered outliers.

The notch in a notched box plot represents the confidence interval around the median. If the notches of two box plots do not overlap, it's a strong indication that the medians differ.

Box plots are useful for comparing distributions and identifying outliers in medical and scientific data.
"""

violin_plot_text = """
Violin plots are a visualization tool for examining distributions of data, combining features from box plots and kernel density plots.

Key points:
1. The width of the "violin" at any point represents the density of data points at that value.
2. The dot in the middle often represents the median.
3. The thicker bar in the middle is the interquartile range (IQR).
4. Violin plots are helpful for visualizing the distribution of a numerical variable across categories.

Violin plots provide a smoothed representation of the data distribution and are useful for comparing groups in medical research.
"""

neural_network_text = """
A neural network is a type of machine learning model inspired by the structure and function of the human brain's neural network. It is excellent for solving complex problems and making predictions based on historical data.

Just like the human brain consists of interconnected neurons, a neural network consists of interconnected artificial neurons called "nodes" or "neurons". These neurons are organized in layers - an input layer, one or more hidden layers, and an output layer. Each neuron takes input from the previous layer, performs a mathematical operation on the input, and passes the result to the next layer.

Here's a simplified breakdown of how a neural network works:

1. **Feedforward**: The input layer receives the input data, which can be numerical or categorical variables. Each neuron in the hidden layers and the output layer performs a weighted sum of the inputs, applies an activation function, and passes the result to the next layer. This process is called feedforward.

2. **Activation Function**: The activation function introduces non-linearity to the neural network, allowing it to learn and model complex relationships in the data. Common activation functions include sigmoid, tanh, and ReLU.

3. **Backpropagation**: After the feedforward process, the neural network compares its predictions to the actual values and calculates the prediction error. It then adjusts the weights and biases of the neurons in the network through a process called backpropagation. This iterative process continues until the neural network reaches a satisfactory level of accuracy.

Neural networks can be used for a wide range of tasks, including regression, classification, and even more complex tasks like image and speech recognition. They have been successfully applied in various domains, including medicine, finance, and natural language processing.

However, it's important to note that neural networks are computationally intensive and require a large amount of training data to generalize well. Additionally, hyperparameter tuning and regularization techniques may be necessary to prevent overfitting and improve performance.
"""
