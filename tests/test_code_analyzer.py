import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.code_analyzer import CodeQualityAnalyzer, QualityIssue


class TestCodeQualityAnalyzer:
    def setup_method(self):
        self.analyzer = CodeQualityAnalyzer()

    def test_perfect_code(self):
        """Test code with no issues"""
        code = '''
def calculate_total(items):
    """Calculate the total price of items."""
    return sum(item.price for item in items)

class ShoppingCart:
    """A simple shopping cart implementation."""

    def __init__(self):
        self.items = []

    def add_item(self, item):
        """Add an item to the cart."""
        self.items.append(item)
'''
        result = self.analyzer.analyze_code(code)
        assert result['quality_score'] >= 90
        assert result['total_issues'] <= 2  # May have minor issues like missing docstrings

    def test_syntax_error_detection(self):
        """Test detection of syntax errors"""
        code = '''
def broken_function(
    return "missing closing parenthesis"
'''
        result = self.analyzer.analyze_code(code)
        issues = result['issues']
        syntax_errors = [issue for issue in issues if issue['issue_type'] == 'syntax_error']
        assert len(syntax_errors) > 0
        assert result['quality_score'] < 50

    def test_naming_convention_issues(self):
        """Test detection of naming convention violations"""
        code = '''
def BadFunctionName():
    pass

class bad_class_name:
    pass
'''
        result = self.analyzer.analyze_code(code)
        issues = result['issues']
        naming_issues = [issue for issue in issues if issue['issue_type'] == 'naming_convention']
        assert len(naming_issues) >= 2

    def test_long_line_detection(self):
        """Test detection of overly long lines"""
        long_line = 'x = ' + '"' + 'a' * 100 + '"'  # Create a line longer than 80 characters
        result = self.analyzer.analyze_code(long_line)
        issues = result['issues']
        long_line_issues = [issue for issue in issues if issue['issue_type'] == 'line_length']
        assert len(long_line_issues) > 0

    def test_missing_docstring_detection(self):
        """Test detection of missing docstrings"""
        code = '''
def function_without_docstring():
    return "no documentation"
'''
        result = self.analyzer.analyze_code(code)
        issues = result['issues']
        docstring_issues = [issue for issue in issues if issue['issue_type'] == 'missing_docstring']
        assert len(docstring_issues) > 0

    def test_magic_number_detection(self):
        """Test detection of magic numbers"""
        code = '''
def calculate_score():
    return score * 1000 + bonus * 500
'''
        result = self.analyzer.analyze_code(code)
        issues = result['issues']
        magic_number_issues = [issue for issue in issues if issue['issue_type'] == 'magic_number']
        assert len(magic_number_issues) > 0

    def test_empty_except_detection(self):
        """Test detection of empty except blocks"""
        code = '''
try:
    risky_operation()
except:
    pass
'''
        result = self.analyzer.analyze_code(code)
        issues = result['issues']
        empty_except_issues = [issue for issue in issues if issue['issue_type'] == 'empty_except']
        assert len(empty_except_issues) > 0

    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        # Code with known issues
        bad_code = '''
def BadFunction():
    try:
        x = 1000
        return x
    except:
        pass
'''
        result = self.analyzer.analyze_code(bad_code)
        assert 0 <= result['quality_score'] <= 100
        assert result['total_issues'] > 0

    def test_severity_breakdown(self):
        """Test severity breakdown calculation"""
        code = '''
def BadFunction():  # naming issue (medium)
    try:
        x = 1000  # magic number (low)
        return x
    except:  # empty except (high)
        pass
'''
        result = self.analyzer.analyze_code(code)
        breakdown = result['severity_breakdown']
        assert 'low' in breakdown
        assert 'medium' in breakdown
        assert 'high' in breakdown

    def test_snake_case_conversion(self):
        """Test snake_case conversion utility"""
        assert self.analyzer._to_snake_case('CamelCase') == 'camel_case'
        assert self.analyzer._to_snake_case('XMLHttpRequest') == 'xml_http_request'
        assert self.analyzer._to_snake_case('already_snake') == 'already_snake'

    def test_pascal_case_conversion(self):
        """Test PascalCase conversion utility"""
        assert self.analyzer._to_pascal_case('snake_case') == 'SnakeCase'
        assert self.analyzer._to_pascal_case('multiple_word_example') == 'MultipleWordExample'
        assert self.analyzer._to_pascal_case('AlreadyPascal') == 'Alreadypascal'

    def test_empty_code(self):
        """Test handling of empty code"""
        result = self.analyzer.analyze_code('')
        assert result['quality_score'] == 100  # No code, no issues
        assert result['total_issues'] == 0

    def test_non_python_code(self):
        """Test handling of non-Python code"""
        java_code = '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''
        result = self.analyzer.analyze_code(java_code)
        # Should not crash, may detect some generic issues
        assert isinstance(result, dict)
        assert 'quality_score' in result