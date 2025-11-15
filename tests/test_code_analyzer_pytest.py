"""
Pytest suite for the lightweight CodeQualityAnalyzer heuristics.
"""
from src.models.code_analyzer import CodeQualityAnalyzer


def _run_analysis(code: str):
    analyzer = CodeQualityAnalyzer()
    return analyzer.analyze_code(code)


def test_analyze_code_returns_perfect_score_for_clean_snippet():
    code = """
def add_numbers(x: int, y: int) -> int:
    \"\"\"Add two numbers and return the result.\"\"\"
    result = x + y
    return result


class GoodClass:
    \"\"\"Simple container docstring.\"\"\"
    pass
""".strip()

    result = _run_analysis(code)

    assert result["issues"] == []
    assert result["quality_score"] == 100.0
    assert result["severity_breakdown"] == {"low": 0, "medium": 0, "high": 0}


def test_syntax_error_caps_score_at_40():
    result = _run_analysis("def broken(:\n    pass")

    assert any(issue["issue_type"] == "syntax_error" for issue in result["issues"])
    assert result["quality_score"] == 40.0
    assert result["severity_breakdown"]["high"] >= 1


def test_detects_common_heuristic_issues():
    problematic_code = """
class my_class:
    pass

def BadFunctionName(param1, param2, param3, param4, param5, param6):
    value = 1234
    try:
        risky_call()
    except:
        pass
    return value
""".strip()

    result = _run_analysis(problematic_code)
    issue_types = {issue["issue_type"] for issue in result["issues"]}

    assert "naming_convention" in issue_types  # bad class + function names
    assert "missing_docstring" in issue_types  # def block lacks docstring
    assert "magic_number" in issue_types  # 1234 constant
    assert "empty_except" in issue_types  # bare except with pass
    assert result["severity_breakdown"]["high"] >= 1  # empty except is high severity
