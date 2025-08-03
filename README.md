# AutoAnalyzer AI

![AutoAnalyzer Demo](https://github.com/user-attachments/assets/2c4dbbaa-d677-4481-9056-869b97d1beb8)

AutoAnalyzer is an interactive data exploration and machine learning application built with Streamlit. It provides healthcare professionals and researchers with tools to analyze datasets, perform statistical tests, create visualizations, and build predictive models without writing code.

## Features

### Data Exploration
- Upload CSV or Excel files (with size validation)
- Explore demo datasets
- Generate synthetic data
- Assess data readiness
- Filter and preprocess data
- Create visualizations (scatter plots, histograms, box plots, etc.)
- Perform statistical tests (t-test, ANOVA, chi-square, etc.)
- Generate Table 1 summaries

### Machine Learning
- Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, etc.)
- Feature selection and preprocessing
- Model evaluation with comprehensive metrics
- SHAP model explanations
- Cross-validation support

### AI-Powered Analysis
- Natural language queries about your data
- Automated code generation and execution
- Iterative refinement of results
- Export results to Word documents

## New Performance Features

### Resource Monitoring
- Real-time monitoring of memory and CPU usage
- Temporary file tracking
- Data size monitoring
- Resource usage display in the sidebar
- Configurable logging with log rotation to prevent large log files
- Reduced logging frequency to minimize I/O overhead

### Temporary File Management
- Automatic cleanup of temporary directories
- Monitored temporary file creation and deletion
- Prevention of disk space exhaustion

### Data Size Limits and Validation
- Configurable limits for rows, columns, and memory usage
- Validation of uploaded files
- Protection against resource exhaustion
- Clear error messages when limits are exceeded

## Installation

### Docker (Recommended)
```bash
# Build the image
docker build -t autoanalyzer .

# Run the container
docker run -p 8501:8501 autoanalyzer
```

### Local Installation
```bash
# Clone the repository
git clone https://github.com/DrDavidL/auto_analyze.git
cd auto_analyze

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Environment Variables
- `PASSWORD` - Application password
- `OPENAI_API_KEY` - OpenAI API key for GPT features
- `HEALTH_UNIVERSE` - Health Universe API key
- `OPENAI_BASE_URL` - Azure OpenAI endpoint

## Usage
1. Access the application at `http://localhost:8501`
2. Enter the password when prompted
3. Upload your data or select a demo dataset
4. Explore using the sidebar tools

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
David Liebovitz, MD, Northwestern University
