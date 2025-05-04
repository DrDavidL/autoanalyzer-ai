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
logistic_regression_text = """
Logistic regression is a statistical model commonly used in the field of medicine to predict binary outcomes - such as whether a patient has a disease (yes/no), whether a patient survived or not after a treatment (survived/did not survive), etc.

Logistic regression, like linear regression, establishes a relationship between the predictor variables (such as patient's age, weight, smoking history) and the target variable (e.g., presence or absence of a disease). However, unlike linear regression which predicts a continuous outcome, logistic regression predicts the probability of an event occurring, which is perfect for binary (two-category) outcomes.

Here's a simplified step-by-step breakdown:

1. **Collect and Prepare Your Data**: This involves gathering medical data that includes both the outcome (what you want to predict) and predictor variables (information you will use to make the prediction).

2. **Build the Model**: Logistic regression uses a mathematical formula that looks somewhat similar to the formula for a line in algebra (y = mx + b), but it's modified to predict probabilities. The formula takes your predictors and calculates the "log odds" of the event occurring.

3. **Interpret the Model**: The coefficients (the values that multiply the predictors) in the logistic regression model represent the change in the log odds of the outcome for a one-unit increase in the predictor variable. For example, if age is a predictor and its coefficient is 0.05, it means that for each one year increase in age, the log odds of the disease occurring (assuming all other factors remain constant) increase by 0.05. Because these are "log odds", the relationship between the predictors and the probability of the outcome isn't a straight line, but a curve that can't go below 0 or above 1.

4. **Make Predictions**: You can input a new patient's information into the logistic regression equation, and it will output the predicted probability of the outcome. For example, it might predict a patient has a 75% chance of having a disease. You can then convert this into a binary outcome by setting a threshold, such as saying any probability above 50% will be considered a "yes."

Remember that logistic regression, while powerful, makes several assumptions. It assumes a linear relationship between the log odds of the outcome and the predictor variables, it assumes that errors are not measured and that there's no multicollinearity (a high correlation among predictor variables). As with any model, it's also only as good as the data you feed into it.

In the medical field, logistic regression can be a helpful tool to predict outcomes and identify risk factors. However, it's important to understand its assumptions and limitations and to use clinical judgment alongside the model's predictions.
"""

decision_tree_text = """
A decision tree is a type of predictive model that you can think of as similar to the flowcharts sometimes used in medical diagnosis. They're made up of nodes (decision points) and branches (choices or outcomes), and they aim to predict an outcome based on input data.

Here's how they work:

1. **Start at the root node**: This is the first decision that needs to be made and it's based on one of your input variables. For instance, in a medical context, this might be a question like, "Is the patient's temperature above 100.4 degrees Fahrenheit?"

2. **Follow the branch for your answer**: If the answer is "yes," follow the branch for "yes," and if it's "no," follow the branch for "no."

3. **Make the next decision**: Each branch leads to another node, where another decision will be made based on another variable. Maybe this time it's, "Does the patient have a cough?"

4. **Continue until you reach a leaf node**: Leaf nodes are nodes without any further branches. They represent the final decisions and are predictions of the outcome. In a binary outcome scenario, leaf nodes could represent "disease" or "no disease."

The decision tree "learns" from data by splitting the data at each node based on what would provide the most significant increase in information (i.e., the best separation of positive and negative cases). For instance, if patients with a certain disease often have a fever, the model might learn to split patients based on whether they have a fever.

While decision trees can be powerful and intuitive tools, there are a few caveats to keep in mind:

- **Overfitting**: If a tree is allowed to grow too deep (too many decision points), it may start to fit not just the underlying trends in the data, but also the random noise. This means it will perform well on the data it was trained on, but poorly on new data.

- **Instability**: Small changes in the data can result in a very different tree. This can be mitigated by using ensemble methods, which combine many trees together (like a random forest).

- **Simplicity**: Decision trees make very simple, linear cuts in the data. They can struggle with relationships in the data that are more complex.

Overall, decision trees can be an excellent tool for understanding and predicting binary outcomes from medical data. They can handle a mixture of data types, deal with missing data, and the results are interpretable and explainable. Just like with any medical test, though, the results should be interpreted with care and in the context of other information available.
"""

random_forest_text = """
Random Forest is a type of machine learning model that is excellent for making predictions (both binary and multi-class) based on multiple input variables, which can be both categorical (like gender: male or female) and numerical (like age or blood pressure).

Imagine you have a patient and you have collected a lot of data about them - age, weight, cholesterol level, blood pressure, whether or not they smoke, etc. You want to predict a binary outcome: will they have a heart attack in the next 10 years or not? 

A Random Forest works a bit like a team of doctors, each of whom asks a series of questions to make their own diagnosis (or prediction). These doctors are analogous to "decision trees" - the building blocks of a Random Forest.

Here's a simplified breakdown of how it works:

1. **Building Decision Trees**: Each "doctor" (or decision tree) in the Random Forest gets a random subset of patients' data. They ask questions like, "Is the patient's age over 60?", "Is their cholesterol level over 200?". Depending on the answers, they follow different paths down the tree, leading to a final prediction. The tree is constructed in a way that the most important questions (those that best split the patients according to the outcome) are asked first.

2. **Making Predictions**: To make a prediction for a new patient, each decision tree in the Random Forest independently makes a prediction. Essentially, each tree "votes" for the outcome it thinks is most likely (heart attack or no heart attack).

3. **Combining the Votes**: The Random Forest combines the votes from all decision trees. The outcome that gets the most votes is the Random Forest's final prediction. This is like asking a team of doctors for their opinions and going with the majority vote.

One of the main strengths of Random Forest is that it can handle complex data with many variables and it doesn't require a lot of data preprocessing (like scaling or normalizing data). Also, it is less prone to "overfitting" compared to individual decision trees. Overfitting is when a model learns the training data too well, to the point where it captures noise and performs poorly when predicting outcomes for new, unseen data.

However, it's important to note that while Random Forest often performs well, it can be somewhat of a "black box", meaning it can be hard to understand why it's making the predictions it's making. It's always crucial to validate the model's predictions against your medical knowledge and context.
"""

gbm_text = """
Gradient Boosting Machines, like Random Forests, are a type of machine learning model that is good at making predictions based on multiple input variables. These variables can be both categorical (like patient sex: male or female) and numerical (like age, heart rate, etc.). 

Again, suppose we're trying to predict a binary outcome: will this patient develop diabetes in the next five years or not?

A GBM also uses decision trees as its building blocks, but there's a crucial difference in how GBM combines these trees compared to Random Forests. Rather than having each tree independently make a prediction and then voting on the final outcome, GBMs build trees in sequence where each new tree is trying to correct the mistakes of the combined existing trees.

Here's a simplified breakdown of how it works:

1. **Building the First Tree**: A single decision tree is built to predict the outcome based on the input variables. However, this tree is usually very simple and doesn't do a great job at making accurate predictions.

2. **Building Subsequent Trees**: New trees are added to the model. Each new tree is constructed to correct the errors made by the existing set of trees. It does this by predicting the 'residual errors' of the previous ensemble of trees. In other words, it tries to predict how much the current model is 'off' for each patient.

3. **Combining the Trees**: The predictions from all trees are added together to make the final prediction. Each tree's contribution is 'weighted', so trees that do a better job at correcting errors have a bigger say in the final prediction.

GBMs are a very powerful method and often perform exceptionally well. Like Random Forests, they can handle complex data with many variables. But they also have a few additional strengths:

- GBMs can capture more complex patterns than Random Forests because they build trees sequentially, each learning from the last.

- GBMs can also give an estimate of the importance of each variable in making predictions, which can be very useful in understanding what's driving your predictions.

However, GBMs do have their challenges:

- They can be prone to overfitting if not properly tuned. Overfitting happens when your model is too complex and starts to capture noise in your data rather than the true underlying patterns.

- They can also be more computationally intensive than other methods, meaning they might take longer to train, especially with larger datasets.

Just like with any model, it's crucial to validate the model's predictions with your medical knowledge and consider the context. It's also important to remember that while GBMs can make very accurate predictions, they don't prove causation. They can identify relationships and patterns in your data, but they can't tell you why those patterns exist.
"""

svm_text = """
Support Vector Machines are a type of machine learning model that can be used for both regression and classification tasks. They can handle both numerical and categorical input variables. In the context of predicting a binary outcome in medical data - let's stick with the example of predicting whether a patient will develop diabetes or not in the next five years - an SVM is a classification tool.

Here's a simplified explanation:

1. **Building the Model**: The SVM algorithm tries to find a hyperplane, or a boundary, that best separates the different classes (in our case, 'will develop diabetes' and 'will not develop diabetes'). This boundary is chosen to be the one that maximizes the distance between the closest points (the "support vectors") in each class, which is why it's called a "Support Vector Machine".

2. **Making Predictions**: Once this boundary is established, new patients can be classified by where they fall in relation to this boundary. If a new patient's data places them on the 'will develop diabetes' side of the boundary, the SVM predicts they will develop diabetes.

Here are some strengths and challenges of SVMs:

Strengths:
- SVMs can model non-linear decision boundaries, and there are many kernels to choose from. This can make them more flexible in capturing complex patterns in the data compared to some other methods.
- They are also fairly robust against overfitting, especially in high-dimensional space.

Challenges:
- However, SVMs are not very easy to interpret compared to models like decision trees or logistic regression. The boundaries they produce can be complex and not easily explainable in terms of the input variables.
- SVMs can be inefficient to train with very large datasets, and they require careful preprocessing of the data and tuning of the parameters.

As with any machine learning model, while an SVM can make predictions about patient health, it's crucial to validate these predictions with medical expertise. Furthermore, an SVM can identify relationships in data, but it doesn't explain why these relationships exist. As always, correlation doesn't imply causation.
"""
dataframe_generation_system_prompt = """You are a medical data expert whose purpose is to generate realistic medical data to populate a dataframe. Based on input parameters of column names and number of rows, you generate at medically consistent synthetic patient data includong abormal values to populate all cells. 
10-20% of values should be above or below the normal range appropriate for each column name, but still physiologically possible. For example, SBP could range from 90 to 190. Creatinine might go from 0.5 to 7.0. Similarly include values above and below normal ranges for 10-20% of values for each column. Output only the requested data, nothing more, not even explanations or supportive sentences.
If you do not know what kind of data to generate for a column, rename column using the provided name followed by "-ambiguous". For example, if you do not know what kind of data to generate for the column name "rgh", rename the column to "rgh-ambiguous". 
Popululate ambiguous columns with randomly selected 1 or 0 values. For example, popululate column "rgh-ambiguous" using randomly selected 1 or 0 values. For diagnoses provided
as column headers, e.g., "diabetes", populate with randomly selected yes or no values. Populate all cells with appropriate values. No missing values.
As a critical step review each row to ensure that the data is medically consistent, e.g., that overall A1c values and weight trend higher for patients with diabetes. If not, regenerate the row or rows.

Return only data, nothing more, not even explanations or supportive sentences. Generate the requested data so it can be processed by the following code into a dataframe:

```

    # Use StringIO to convert the string data into file-like object
    data = io.StringIO(response.choices[0].message.content)

    # Read the data into a DataFrame, skipping the first row
    df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)

```

Your input parameters will be in this format

Columns: ```columns```
Number of rows: ```number```
"""
