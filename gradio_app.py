"""
Gradio interface for Hugging Face Spaces deployment.
Provides an interactive web UI for code quality assessment.
"""
import sys
import json
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_PATH = CURRENT_DIR / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import gradio as gr
from src.models.code_analyzer import CodeQualityAnalyzer
from src.utils.feedback_generator import FeedbackGenerator

# Initialize components
analyzer = CodeQualityAnalyzer()
feedback_gen = FeedbackGenerator()

# Load pre-computed model parameters data
MODEL_PARAMS_FILE = CURRENT_DIR / 'model_params_data.json'
try:
    with open(MODEL_PARAMS_FILE, 'r') as f:
        MODEL_PARAMS_DATA = json.load(f)
except FileNotFoundError:
    MODEL_PARAMS_DATA = {}


def analyze_code(code_text, show_details=True):
    """
    Analyze code quality and return formatted results.

    Args:
        code_text: Python code to analyze
        show_details: Whether to show detailed issue breakdown

    Returns:
        Formatted markdown string with analysis results
    """
    if not code_text.strip():
        return "⚠️ Please enter some code to analyze."

    try:
        # Analyze code
        result = analyzer.analyze_code(code_text)
        feedback = feedback_gen.generate_feedback(result)

        # Determine score emoji and color
        score = result['quality_score']
        if score >= 90:
            score_emoji = "🟢"
            score_label = "Excellent"
        elif score >= 75:
            score_emoji = "🔵"
            score_label = "Good"
        elif score >= 60:
            score_emoji = "🟡"
            score_label = "Fair"
        elif score >= 40:
            score_emoji = "🟠"
            score_label = "Poor"
        else:
            score_emoji = "🔴"
            score_label = "Critical"

        # Build output
        output = f"""# {score_emoji} Code Quality Analysis

## Overall Score: {score}/100 ({score_label})

### Summary
- **Total Issues:** {result['total_issues']}
- **High Severity:** {result['severity_breakdown']['high']} 🔴
- **Medium Severity:** {result['severity_breakdown']['medium']} 🟡
- **Low Severity:** {result['severity_breakdown']['low']} 🟢

---

### 📊 Assessment
{feedback.get('overall_assessment', 'No assessment available.')}

"""

        # Add suggestions
        suggestions = feedback.get('suggestions', [])
        if suggestions:
            output += "### 💡 Suggestions for Improvement\n\n"
            for i, suggestion in enumerate(suggestions[:8], 1):
                output += f"{i}. {suggestion}\n"
            output += "\n"

        # Add detailed issues if requested
        if show_details and result['issues']:
            output += "---\n\n### 🔍 Detailed Issues\n\n"

            # Group by severity
            for severity in ['high', 'medium', 'low']:
                severity_issues = [
                    issue for issue in result['issues']
                    if issue['severity'] == severity
                ]

                if severity_issues:
                    severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[severity]
                    output += f"\n#### {severity_emoji} {severity.title()} Severity Issues\n\n"

                    for issue in severity_issues[:5]:  # Limit to 5 per severity
                        output += f"**{issue['issue_type'].replace('_', ' ').title()}** "
                        if issue['line_number'] > 0:
                            output += f"(Line {issue['line_number']})"
                        output += f"\n- {issue['description']}\n"
                        output += f"- *Suggestion:* {issue['suggestion']}\n"
                        if issue['code_snippet']:
                            output += f"- *Code:* `{issue['code_snippet'][:60]}...`\n"
                        output += "\n"

        # Add learning resources
        resources = feedback.get('learning_resources', [])
        if resources:
            output += "---\n\n### 📚 Learning Resources\n\n"
            for resource in resources[:5]:
                output += f"- {resource}\n"

        return output

    except Exception as e:
        return f"""# ❌ Error

An error occurred during analysis:

```
{str(e)}
```

Please ensure you've entered valid Python code and try again.
"""


def get_example_code(example_type):
    """Get example code based on quality level."""
    examples = {
        "Good Code": '''def calculate_average(numbers: list[float]) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
        numbers: List of numbers to average

    Returns:
        The arithmetic mean of the numbers

    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")

    return sum(numbers) / len(numbers)
''',
        "Poor Code": '''def calc(n):
    x=0
    for i in n:
        x=x+i
    return x/len(n)
''',
        "Code with Issues": '''def processData(userInput):
    result = []
    for item in userInput:
        if item > 100:
            result.append(item * 2)
        else:
            result.append(item)
    return result

def doSomething():
    pass
''',
    }
    return examples.get(example_type, "")


def format_number(num):
    """Format large numbers with K/M/B suffixes."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B ({num:,})"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M ({num:,})"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K ({num:,})"
    else:
        return f"{num:,}"


def get_model_parameters(model_name, with_heads, num_issues):
    """
    Get model parameters from pre-computed data.
    """
    if not MODEL_PARAMS_DATA:
        return "⚠️ Model parameter data not available.", ""

    if model_name not in MODEL_PARAMS_DATA:
        return f"❌ Data for {model_name} not found.", ""

    data = MODEL_PARAMS_DATA[model_name]
    total_params = data["total_params"]
    trainable_params = data["trainable_params"]
    frozen_params = data["frozen_params"]
    size_mb = data["total_size_mb"]
    param_breakdown = data["param_breakdown"]

    # Add multi-task heads if requested
    additional_params = 0
    if with_heads:
        mt_data = MODEL_PARAMS_DATA.get("multi_task_additions", {})
        quality_head_params = mt_data.get("quality_head", {}).get("params", 769)
        issue_head_params = mt_data.get("issue_classification_head_per_issue", {}).get("params", 769)
        additional_params = quality_head_params + (issue_head_params * num_issues)
        total_params += additional_params
        trainable_params += additional_params

    # Format output
    output = f"""# 📊 Model Parameter Analysis

## Model: {model_name}

{data.get('description', '')}

**Architecture**: {data.get('architecture', 'N/A')}

---

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Parameters** | {format_number(total_params)} |
| **Trainable Parameters** | {format_number(trainable_params)} |
| **Frozen Parameters** | {format_number(frozen_params)} |
| **Model Size** | {size_mb:.2f} MB |
| **Trainable Percentage** | {(trainable_params/total_params*100):.2f}% |

---

### Parameter Breakdown by Component

| Component | Parameters | Percentage |
|-----------|------------|------------|
"""

    for component, params in param_breakdown.items():
        percentage = (params / data["total_params"]) * 100
        output += f"| {component} | {format_number(params)} | {percentage:.2f}% |\n"

    if with_heads and additional_params > 0:
        output += f"""

---

### Multi-Task Model Configuration

| Component | Parameters | Purpose |
|-----------|------------|---------|
| **Base Model** | {format_number(data['total_params'])} | Pre-trained transformer |
| **Quality Head** | {format_number(769)} | Regression (quality score 0-100) |
| **Issue Heads** | {format_number(769 * num_issues)} | {num_issues} binary classifiers |
| **Total Additional** | {format_number(additional_params)} | Custom heads |

**Total Model Size**: {format_number(total_params)} parameters ({size_mb + (additional_params * 4 / 1024 / 1024):.2f} MB)
"""

    output += """

---

### About These Models

These are transformer-based models specifically designed for code understanding:

- **CodeBERT**: Pre-trained on code and natural language, excels at code-text tasks
- **GraphCodeBERT**: Adds data flow graph understanding for better code semantics
- **CodeBERTa-small**: Lightweight variant for faster inference

All models use a BERT-like architecture with 12 transformer layers, 768-dimensional hidden states, and 12 attention heads.

💡 **Note**: This data is pre-computed. To analyze custom models or get live parameter counts, run the advanced interface locally with ML dependencies.
"""

    # Create CSV data
    csv_data = "Metric,Value\n"
    csv_data += f"Model,{model_name}\n"
    csv_data += f"Total Parameters,{total_params}\n"
    csv_data += f"Trainable Parameters,{trainable_params}\n"
    csv_data += f"Frozen Parameters,{frozen_params}\n"
    csv_data += f"Size (MB),{size_mb}\n"
    csv_data += "\nComponent,Parameters,Percentage\n"
    for component, params in param_breakdown.items():
        percentage = (params / data["total_params"]) * 100
        csv_data += f"{component},{params},{percentage:.2f}\n"

    return output, csv_data


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Code Quality Assessment") as demo:
    gr.Markdown("""
    # 🔍 AI-Powered Code Quality Assessment

    Analyze Python code quality and explore transformer model architectures for code understanding.
    """)

    with gr.Tabs():
        # Tab 1: Code Analysis
        with gr.Tab("📝 Code Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    code_input = gr.Code(
                        label="Python Code",
                        language="python",
                        lines=15,
                    )

                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze Code", variant="primary", size="lg")
                        clear_btn = gr.ClearButton([code_input], value="Clear")

                    show_details = gr.Checkbox(
                        label="Show detailed issue breakdown",
                        value=True,
                    )

                    gr.Markdown("### 📝 Try Examples")
                    example_dropdown = gr.Dropdown(
                        choices=["Good Code", "Poor Code", "Code with Issues"],
                        label="Load Example",
                    )

                with gr.Column(scale=1):
                    output = gr.Markdown(
                        label="Analysis Results",
                        value="👈 Enter code and click 'Analyze Code' to see results."
                    )

            # Event handlers
            analyze_btn.click(
                fn=analyze_code,
                inputs=[code_input, show_details],
                outputs=output,
            )

            example_dropdown.change(
                fn=get_example_code,
                inputs=example_dropdown,
                outputs=code_input,
            )

        # Tab 2: Model Parameters
        with gr.Tab("🔧 Model Parameters"):
            gr.Markdown("""
            ### Transformer Model Architecture Explorer

            View detailed parameter counts and architecture breakdowns for code understanding models.
            """)

            with gr.Row():
                with gr.Column():
                    model_name_input = gr.Dropdown(
                        choices=[
                            "microsoft/codebert-base",
                            "microsoft/graphcodebert-base",
                            "huggingface/CodeBERTa-small-v1",
                        ],
                        value="microsoft/codebert-base",
                        label="Select Model",
                    )

                    with_heads_check = gr.Checkbox(
                        label="Include Multi-Task Heads (Quality + Issue Detection)",
                        value=False,
                    )

                    num_issues_slider = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Issue Types",
                        visible=False,
                    )

                    analyze_model_btn = gr.Button("📊 Analyze Model", variant="primary")

                with gr.Column():
                    param_output = gr.Markdown(value="👈 Select a model and click 'Analyze Model'")
                    csv_download = gr.File(label="Download CSV", visible=False)

            # Show/hide num_issues slider based on with_heads
            def update_slider_visibility(show):
                return gr.update(visible=show)

            with_heads_check.change(
                fn=update_slider_visibility,
                inputs=with_heads_check,
                outputs=num_issues_slider,
            )

            # Analyze model button
            def analyze_and_save_csv(model, heads, issues):
                output, csv = get_model_parameters(model, heads, issues)
                # Save CSV to a file for download
                if csv:
                    csv_path = CURRENT_DIR / "model_params.csv"
                    with open(csv_path, 'w') as f:
                        f.write(csv)
                    return output, gr.update(value=str(csv_path), visible=True)
                return output, gr.update(visible=False)

            analyze_model_btn.click(
                fn=analyze_and_save_csv,
                inputs=[model_name_input, with_heads_check, num_issues_slider],
                outputs=[param_output, csv_download],
            )

    # Footer
    gr.Markdown("""
    ---

    ## Features

    - ✅ **Code Analysis** - Syntax validation, naming conventions, complexity detection
    - ✅ **Model Explorer** - View transformer model parameters and architecture
    - ✅ **Educational Feedback** - Get actionable improvement suggestions
    - ✅ **CSV Export** - Download parameter analysis data

    ## About

    This tool combines code quality analysis with ML model exploration.
    Perfect for learning, code reviews, and understanding transformer architectures!

    Built with ❤️ using [Gradio](https://gradio.app) and deployed on [Hugging Face Spaces](https://huggingface.co/spaces)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
