# System Patterns

## System Architecture

The system is designed as a monolithic application written in Python. It appears to be a command-line tool that takes a dataset, processes it, and generates an output file. The presence of a Dockerfile suggests that it can be run in a containerized environment.

## Key Technical Decisions

- **Language:** Python is the primary language, chosen for its strong data science and machine learning libraries.
- **Core Logic:** The main application logic seems to be contained within `main.py`.
- **Modularity:** The project is organized into modules, with data stored in the `data/` directory and explanation-related code in `explanations/`.

## Component Relationships

- `main.py`: The entry point of the application. It likely orchestrates the entire analysis process.
- `prompts.py`: This file probably contains prompts for interacting with a language model, given the name.
- `markdown_to_docx.py`: This suggests that the final report is generated in Markdown and then converted to a DOCX file.
- `Dockerfile`: Defines the environment for running the application in a container, ensuring consistency across different systems.
