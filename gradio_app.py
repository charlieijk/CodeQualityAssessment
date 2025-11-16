"""
Gradio interface for Hugging Face Spaces deployment.
Provides an interactive web UI for code quality assessment.
"""
import sys
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


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Code Quality Assessment") as demo:
    gr.Markdown("""
    # 🔍 AI-Powered Code Quality Assessment

    Upload Python code to get instant quality analysis, issue detection, and improvement suggestions.
    Powered by advanced heuristics and machine learning.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Code(
                label="Python Code",
                language="python",
                lines=15,
                placeholder="Paste your Python code here...",
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

    # Examples section
    gr.Markdown("""
    ---

    ## Features

    - ✅ **Syntax Validation** - Detect Python syntax errors
    - ✅ **Naming Conventions** - Check PEP 8 naming compliance
    - ✅ **Code Structure** - Identify complexity and nesting issues
    - ✅ **Best Practices** - Detect code smells and anti-patterns
    - ✅ **Documentation** - Check for missing docstrings
    - ✅ **Educational Feedback** - Get actionable improvement suggestions

    ## About

    This tool uses advanced code analysis algorithms to assess Python code quality.
    Perfect for learning, code reviews, and improving your programming skills!

    Built with ❤️ using [Gradio](https://gradio.app) and deployed on [Hugging Face Spaces](https://huggingface.co/spaces)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
