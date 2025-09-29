import re
import ast
import keyword
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QualityIssue:
    issue_type: str
    severity: str  # 'low', 'medium', 'high'
    line_number: int
    description: str
    suggestion: str
    code_snippet: str


class CodeQualityAnalyzer:
    def __init__(self):
        self.python_keywords = set(keyword.kwlist)
        self.common_patterns = {
            'naming_convention': {
                'snake_case_function': r'^[a-z_][a-z0-9_]*$',
                'camel_case_class': r'^[A-Z][a-zA-Z0-9]*$',
                'constant': r'^[A-Z][A-Z0-9_]*$'
            },
            'code_smells': {
                'long_line': 80,
                'deep_nesting': 4,
                'too_many_parameters': 5
            }
        }

    def analyze_code(self, code_text: str) -> Dict[str, Any]:
        """
        Analyze extracted code text for quality issues
        """
        lines = code_text.split('\n')
        issues = []

        # Basic syntax and structure analysis
        issues.extend(self._check_syntax_issues(code_text))
        issues.extend(self._check_naming_conventions(lines))
        issues.extend(self._check_code_structure(lines))
        issues.extend(self._check_common_mistakes(lines))

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(issues)

        return {
            'issues': [issue.__dict__ for issue in issues],
            'quality_score': quality_score,
            'total_issues': len(issues),
            'severity_breakdown': self._get_severity_breakdown(issues)
        }

    def _check_syntax_issues(self, code_text: str) -> List[QualityIssue]:
        """
        Check for basic Python syntax issues
        """
        issues = []

        try:
            # Try to parse as Python AST
            ast.parse(code_text)
        except SyntaxError as e:
            issues.append(QualityIssue(
                issue_type='syntax_error',
                severity='high',
                line_number=e.lineno if e.lineno else 1,
                description=f'Syntax Error: {e.msg}',
                suggestion='Fix the syntax error to make the code executable',
                code_snippet=e.text if e.text else ''
            ))
        except Exception:
            # If it's not valid Python, check for common programming patterns
            pass

        return issues

    def _check_naming_conventions(self, lines: List[str]) -> List[QualityIssue]:
        """
        Check for naming convention violations
        """
        issues = []

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Check function definitions
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                if not re.match(self.common_patterns['naming_convention']['snake_case_function'], func_name):
                    issues.append(QualityIssue(
                        issue_type='naming_convention',
                        severity='medium',
                        line_number=i,
                        description=f'Function name "{func_name}" should use snake_case',
                        suggestion=f'Consider renaming to: {self._to_snake_case(func_name)}',
                        code_snippet=line
                    ))

            # Check class definitions
            class_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if class_match:
                class_name = class_match.group(1)
                if not re.match(self.common_patterns['naming_convention']['camel_case_class'], class_name):
                    issues.append(QualityIssue(
                        issue_type='naming_convention',
                        severity='medium',
                        line_number=i,
                        description=f'Class name "{class_name}" should use PascalCase',
                        suggestion=f'Consider renaming to: {self._to_pascal_case(class_name)}',
                        code_snippet=line
                    ))

        return issues

    def _check_code_structure(self, lines: List[str]) -> List[QualityIssue]:
        """
        Check for structural code quality issues
        """
        issues = []

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.common_patterns['code_smells']['long_line']:
                issues.append(QualityIssue(
                    issue_type='line_length',
                    severity='low',
                    line_number=i,
                    description=f'Line too long ({len(line)} characters)',
                    suggestion='Consider breaking long lines into multiple lines',
                    code_snippet=line[:50] + '...' if len(line) > 50 else line
                ))

            # Check indentation depth
            indent_level = len(line) - len(line.lstrip())
            if indent_level > self.common_patterns['code_smells']['deep_nesting'] * 4:
                issues.append(QualityIssue(
                    issue_type='deep_nesting',
                    severity='medium',
                    line_number=i,
                    description='Deep nesting detected',
                    suggestion='Consider refactoring to reduce nesting levels',
                    code_snippet=line.strip()
                ))

        return issues

    def _check_common_mistakes(self, lines: List[str]) -> List[QualityIssue]:
        """
        Check for common programming mistakes
        """
        issues = []

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Check for missing docstrings in functions
            if line_stripped.startswith('def ') and ':' in line_stripped:
                next_line = lines[i] if i < len(lines) else ''
                if not next_line.strip().startswith('"""') and not next_line.strip().startswith("'''"):
                    issues.append(QualityIssue(
                        issue_type='missing_docstring',
                        severity='low',
                        line_number=i,
                        description='Function missing docstring',
                        suggestion='Add a docstring to document the function purpose',
                        code_snippet=line_stripped
                    ))

            # Check for hardcoded values
            if re.search(r'\b\d{3,}\b', line_stripped):  # Numbers with 3+ digits
                issues.append(QualityIssue(
                    issue_type='magic_number',
                    severity='low',
                    line_number=i,
                    description='Potential magic number detected',
                    suggestion='Consider using a named constant',
                    code_snippet=line_stripped
                ))

            # Check for empty except blocks
            if 'except:' in line_stripped and line_stripped.endswith(':'):
                next_line = lines[i] if i < len(lines) else ''
                if next_line.strip() == 'pass':
                    issues.append(QualityIssue(
                        issue_type='empty_except',
                        severity='high',
                        line_number=i,
                        description='Empty except block detected',
                        suggestion='Handle exceptions properly or log the error',
                        code_snippet=line_stripped
                    ))

        return issues

    def _calculate_quality_score(self, issues: List[QualityIssue]) -> float:
        """
        Calculate overall quality score (0-100)
        """
        if not issues:
            return 100.0

        severity_weights = {'low': 1, 'medium': 3, 'high': 5}
        total_penalty = sum(severity_weights.get(issue.severity, 1) for issue in issues)

        # Base score starts at 100, subtract penalties
        score = max(0, 100 - (total_penalty * 2))
        return round(score, 1)

    def _get_severity_breakdown(self, issues: List[QualityIssue]) -> Dict[str, int]:
        """
        Get breakdown of issues by severity
        """
        breakdown = {'low': 0, 'medium': 0, 'high': 0}
        for issue in issues:
            breakdown[issue.severity] += 1
        return breakdown

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase"""
        components = name.split('_')
        return ''.join(word.capitalize() for word in components)