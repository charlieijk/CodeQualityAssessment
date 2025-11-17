"""
Advanced Gradio interface with model parameter visualization and fine-tuning.
This version includes ML model analysis and training capabilities.
"""
import sys
from pathlib import Path
import json
import io

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_PATH = CURRENT_DIR / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import gradio as gr
from src.models.code_analyzer import CodeQualityAnalyzer
from src.utils.feedback_generator import FeedbackGenerator

# Try to import ML components
ML_AVAILABLE = False
try:
    import torch
    from src.models.transformer_model import TransformerConfig, MultiTaskCodeQualityModel
    from src.utils.model_analyzer import ModelParameterAnalyzer, analyze_codebert_model
    ML_AVAILABLE = True
except ImportError:
    pass

# Initialize components
analyzer = CodeQualityAnalyzer()
feedback_gen = FeedbackGenerator()


def analyze_code(code_text, show_details=True):
    """
    Analyze code quality and return formatted results.
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

                    for issue in severity_issues[:5]:
                        output += f"**{issue['issue_type'].replace('_', ' ').title()}** "
                        if issue['line_number'] > 0:
                            output += f"(Line {issue['line_number']})"
                        output += f"\n- {issue['description']}\n"
                        output += f"- *Suggestion:* {issue['suggestion']}\n"
                        if issue['code_snippet']:
                            output += f"- *Code:* `{issue['code_snippet'][:60]}...`\n"
                        output += "\n"

        return output

    except Exception as e:
        return f"""# ❌ Error

An error occurred during analysis:

```
{str(e)}
```

Please ensure you've entered valid Python code and try again.
"""


def get_model_parameters(model_name, with_heads, num_issues):
    """
    Get and display model parameters in a table format.
    """
    if not ML_AVAILABLE:
        return "❌ ML dependencies not installed. Install requirements-ml-optional.txt to use this feature."

    try:
        # Analyze the model
        if with_heads:
            config = TransformerConfig(model_name=model_name)
            model = MultiTaskCodeQualityModel(config, num_issue_types=num_issues)
            total_params, trainable_params = model.count_parameters()

            # Get detailed breakdown
            analyzer_tool = ModelParameterAnalyzer()
            summary = analyzer_tool.analyze_model(model, detailed=True)
        else:
            summary = analyze_codebert_model(model_name)
            total_params = summary.total_params
            trainable_params = summary.trainable_params

        # Format as markdown table
        output = f"""# 📊 Model Parameter Analysis

## Model: {model_name}

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | {ModelParameterAnalyzer.format_number(summary.total_params)} |
| Trainable Parameters | {ModelParameterAnalyzer.format_number(summary.trainable_params)} |
| Frozen Parameters | {ModelParameterAnalyzer.format_number(summary.frozen_params)} |
| Model Size | {summary.total_size_mb:.2f} MB |
| Trainable Percentage | {(summary.trainable_params/summary.total_params*100):.2f}% |

---

### Parameter Breakdown by Component

| Component | Parameters | Percentage |
|-----------|------------|------------|
"""

        for component, params in sorted(summary.param_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (params / summary.total_params) * 100
            output += f"| {component} | {ModelParameterAnalyzer.format_number(params)} | {percentage:.2f}% |\n"

        if with_heads:
            output += f"""

---

### Multi-Task Model Configuration

- **Base Model**: {model_name}
- **Quality Head**: Regression output (1 unit)
- **Issue Classification Head**: Multi-label output ({num_issues} issue types)
- **Additional Parameters**: {ModelParameterAnalyzer.format_number(summary.total_params - analyze_codebert_model(model_name).total_params)}
"""

        # Create CSV data for download
        csv_data = "Component,Parameters,Percentage\n"
        csv_data += f"Total,{summary.total_params},100.00\n"
        csv_data += f"Trainable,{summary.trainable_params},{(summary.trainable_params/summary.total_params*100):.2f}\n"
        csv_data += f"Frozen,{summary.frozen_params},{(summary.frozen_params/summary.total_params*100):.2f}\n"
        for component, params in summary.param_breakdown.items():
            percentage = (params / summary.total_params) * 100
            csv_data += f"{component},{params},{percentage:.2f}\n"

        return output, csv_data

    except Exception as e:
        return f"❌ Error analyzing model: {str(e)}", ""


def start_finetuning(model_name, dataset_text, num_epochs, batch_size, learning_rate):
    """
    Start fine-tuning a model (simulated for now).
    """
    if not ML_AVAILABLE:
        return "❌ ML dependencies not installed. Install requirements-ml-optional.txt to use this feature."

    # Parse dataset
    try:
        dataset_lines = [line.strip() for line in dataset_text.strip().split('\n') if line.strip()]
        num_samples = len(dataset_lines)

        if num_samples < 10:
            return "❌ Dataset too small. Please provide at least 10 code samples (one per line)."

        output = f"""# 🚀 Fine-tuning Configuration

## Model Settings
- **Base Model**: {model_name}
- **Number of Epochs**: {num_epochs}
- **Batch Size**: {batch_size}
- **Learning Rate**: {learning_rate}

## Dataset
- **Number of Samples**: {num_samples}
- **Train/Val Split**: 80/20
- **Training Samples**: {int(num_samples * 0.8)}
- **Validation Samples**: {int(num_samples * 0.2)}

---

## Training Plan

1. **Data Preprocessing**
   - Tokenize code samples
   - Generate quality scores and issue labels
   - Create train/validation split

2. **Model Setup**
   - Load pre-trained {model_name}
   - Add quality regression head
   - Add issue classification head
   - Configure optimizer (AdamW)

3. **Training Loop**
   - {num_epochs} epochs with early stopping
   - Batch size: {batch_size}
   - Learning rate: {learning_rate} with warmup

4. **Evaluation**
   - MAE for quality scores
   - F1 score for issue detection
   - Attention visualization

---

**Note**: This is a training plan preview. To actually run training, use:

```bash
python -m src.models.train_transformer \\
  --dataset your_dataset.jsonl \\
  --model-name {model_name} \\
  --num-epochs {num_epochs} \\
  --batch-size {batch_size} \\
  --learning-rate {learning_rate}
```

Estimated training time: ~{num_epochs * num_samples // batch_size // 10} minutes (on GPU)
"""

        return output

    except Exception as e:
        return f"❌ Error preparing training: {str(e)}"


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


# Create Gradio interface with tabs
with gr.Blocks(theme=gr.themes.Soft(), title="Code Quality Assessment - Advanced") as demo:
    gr.Markdown("""
    # 🔍 AI-Powered Code Quality Assessment (Advanced)

    Analyze Python code, visualize model parameters, and fine-tune transformer models.
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
                    analysis_output = gr.Markdown(
                        label="Analysis Results",
                        value="👈 Enter code and click 'Analyze Code' to see results."
                    )

            analyze_btn.click(
                fn=analyze_code,
                inputs=[code_input, show_details],
                outputs=analysis_output,
            )

            example_dropdown.change(
                fn=get_example_code,
                inputs=example_dropdown,
                outputs=code_input,
            )

        # Tab 2: Model Parameters
        with gr.Tab("🔧 Model Parameters"):
            gr.Markdown("""
            ### Explore Transformer Model Architecture

            View detailed parameter counts and breakdowns for CodeBERT and similar models.
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

                    analyze_model_btn = gr.Button("📊 Analyze Model Parameters", variant="primary")

                with gr.Column():
                    param_output = gr.Markdown(value="👈 Select a model and click 'Analyze Model Parameters'")
                    csv_output = gr.File(label="Download CSV", visible=False)

            # Show/hide num_issues slider based on with_heads
            with_heads_check.change(
                lambda x: gr.update(visible=x),
                inputs=with_heads_check,
                outputs=num_issues_slider,
            )

            analyze_model_btn.click(
                fn=get_model_parameters,
                inputs=[model_name_input, with_heads_check, num_issues_slider],
                outputs=[param_output, csv_output],
            )

        # Tab 3: Fine-tuning
        with gr.Tab("🎯 Fine-Tuning"):
            gr.Markdown("""
            ### Fine-tune Transformer Models

            Configure and prepare fine-tuning for your custom code quality dataset.
            """)

            with gr.Row():
                with gr.Column():
                    ft_model_name = gr.Dropdown(
                        choices=[
                            "microsoft/codebert-base",
                            "microsoft/graphcodebert-base",
                            "huggingface/CodeBERTa-small-v1",
                        ],
                        value="microsoft/codebert-base",
                        label="Base Model",
                    )

                    dataset_input = gr.Textbox(
                        label="Training Dataset (one code sample per line)",
                        lines=10,
                        placeholder="def example():\n    pass\n\nclass MyClass:\n    pass\n\n...",
                    )

                    with gr.Row():
                        num_epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of Epochs",
                        )
                        batch_size_slider = gr.Slider(
                            minimum=4,
                            maximum=32,
                            value=16,
                            step=4,
                            label="Batch Size",
                        )

                    learning_rate_input = gr.Number(
                        value=2e-5,
                        label="Learning Rate",
                        precision=6,
                    )

                    finetune_btn = gr.Button("🚀 Prepare Fine-tuning", variant="primary")

                with gr.Column():
                    finetune_output = gr.Markdown(
                        value="👈 Configure settings and click 'Prepare Fine-tuning'"
                    )

            finetune_btn.click(
                fn=start_finetuning,
                inputs=[ft_model_name, dataset_input, num_epochs_slider, batch_size_slider, learning_rate_input],
                outputs=finetune_output,
            )

    # Footer
    gr.Markdown("""
    ---

    ## Features

    - ✅ **Code Analysis** - Detect Python syntax errors, style issues, and anti-patterns
    - ✅ **Model Parameters** - Visualize transformer model architecture and parameter counts
    - ✅ **Fine-tuning** - Configure and prepare custom model training
    - ✅ **Educational Feedback** - Get actionable improvement suggestions

    Built with ❤️ using [Gradio](https://gradio.app) and [Transformers](https://huggingface.co/transformers)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
