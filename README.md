# DS-STAR: A Data Science Agentic Framework

DS-STAR (Data Science - Structured Thought and Action) is a Python-based agentic framework for automating data science tasks. It leverages a multi-agent system powered by Google's Gemini models to analyze data, devise a plan, write and execute code, and iteratively refine the solution to answer a user's query.

This project is an implementation of the paper from Google Research: [DS-STAR: A State-of-the-Art Versatile Data Science Agent](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/). [Paper](https://arxiv.org/pdf/2509.21825)

## Features

- **Agentic Workflow**: Implements a pipeline of specialized AI agents (Analyzer, Planner, Coder, Verifier, Router, Debugger, Finalyzer) that collaborate to solve data science problems.
- **Reproducibility**: Every step of the pipeline is saved, including prompts, generated code, execution results, and metadata. This allows for complete auditability and reproducibility of results.
- **Interactive & Resume-able**: Runs can be paused and resumed. The interactive mode allows for step-by-step execution.
- **Code Editing & Debugging**: Allows users to manually edit the generated code during a run and features an auto-debug agent to fix execution errors.
- **Configuration-driven**: Project settings, model parameters, and run configurations are managed through a `config.yaml` file.

## How it Works

The DS-STAR pipeline is composed of several phases and agents:

1.  **Analysis**: The `Analyzer` agent inspects the initial data files and generates summaries.
2.  **Iterative Planning & Execution**:
    *   The `Planner` creates an initial plan to address the user's query.
    *   The `Coder` generates Python code to execute the current step of the plan.
    *   The code is executed, and the result is captured.
    *   An automatic `Debugger` agent attempts to fix any code that fails.
    *   The `Verifier` checks if the result sufficiently answers the query.
    *   The `Router` decides what to do next: either finalize the plan or add a new step for refinement.
    *   This loop continues until the plan is deemed sufficient or the maximum number of refinement rounds is reached.
3.  **Finalization**: The `Finalyzer` agent takes the final code and results and formats them into a clean, specified output format (e.g., JSON).

All artifacts for each run are stored in the `runs/` directory, organized by `run_id`.

## Project Structure

```
/
├─── dsstar.py               # Main script containing the agent logic and CLI
├─── config.yaml             # Main configuration file
├─── prompt.yaml             # Prompts for the different AI agents
├─── requirements.txt        # Python dependencies
├─── data/                   # Directory for your data files
└─── runs/                   # Directory where all experiment runs and artifacts are stored
```

## Getting Started

### Prerequisites

- Python 3.9+
- An API key for Google's Gemini models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DS_Star_impl
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set your API Key:**
    The application requires a Gemini API key. You can set it as an environment variable:
    ```bash
    export GEMINI_API_KEY='your-api-key'
    ```
    Alternatively, you can add it to the `config.yaml` file.

2.  **Customize `config.yaml`:**
    Create a `config.yaml` file in the root of the project and customize the settings. See the "Configuration" section below for details.

    ```yaml
    # config.yaml
    model_name: 'gemini-1.5-flash'
    max_refinement_rounds: 5
    interactive: false
    # api_key: 'your-api-key' # Alternatively, place it here
    ```

## Usage

Place your data files (e.g., `.xlsx`, `.csv`) in the `data/` directory.

### Starting a New Run

To start a new analysis, you need to provide the data files and a query.

```bash
python dsstar.py --data-files file1.xlsx file2.xlsx --query "What is the total sales for each department?"
```

### Resuming a Run

If a run was interrupted, you can resume it using its `run_id`.

```bash
python dsstar.py --resume <run_id>
```

### Editing Code During a Run

You can manually edit the last generated piece of code and re-run it. This is useful for manual debugging or tweaking the agent's logic.

```bash
python dsstar.py --edit-last --resume <run_id>
```
This will open the last code file in your default text editor (`nano`, `vim`, etc.). After you save and close the editor, the script will re-execute the modified code.

### Interactive Mode

To review each step before proceeding, use the interactive flag.

```bash
python dsstar.py --interactive --data-files ... --query "..."
```

## Configuration

The following options are available in `config.yaml` and can be overridden by CLI arguments:

- `run_id` (string): The ID of a run to resume.
- `max_refinement_rounds` (int): The maximum number of times the agent will try to refine its plan.
- `api_key` (string): Your Gemini API key.
- `model_name` (string): The Gemini model to use (e.g., `gemini-1.5-flash`).
- `interactive` (bool): If true, waits for user input before executing each step.
- `auto_debug` (bool): If true, the `Debugger` agent will automatically try to fix failing code.
- `execution_timeout` (int): Timeout in seconds for code execution.
- `preserve_artifacts` (bool): If true, all step artifacts are saved to the `runs` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

```