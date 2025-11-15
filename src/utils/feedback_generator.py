from typing import Dict, List, Any
import random


class FeedbackGenerator:
    def __init__(self):
        self.feedback_templates = {
            'syntax_error': {
                'title': 'Syntax Error Detected',
                'explanation': 'Your code contains syntax errors that prevent it from running.',
                'suggestions': [
                    'Check for missing colons, parentheses, or brackets',
                    'Ensure proper indentation',
                    'Verify variable names and keywords are spelled correctly',
                    'Use a code editor with syntax highlighting to catch errors early'
                ],
                'resources': [
                    'Python syntax documentation: https://docs.python.org/3/reference/lexical_analysis.html',
                    'Common Python syntax errors guide'
                ]
            },
            'naming_convention': {
                'title': 'Naming Convention Issues',
                'explanation': 'Consistent naming conventions make code more readable and maintainable.',
                'suggestions': [
                    'Use snake_case for functions and variables (e.g., calculate_total)',
                    'Use PascalCase for classes (e.g., StudentRecord)',
                    'Use UPPER_CASE for constants (e.g., MAX_RETRY_ATTEMPTS)',
                    'Choose descriptive names that explain the purpose'
                ],
                'resources': [
                    'PEP 8 -- Style Guide for Python Code',
                    'Clean Code principles for naming'
                ]
            },
            'line_length': {
                'title': 'Long Lines Detected',
                'explanation': 'Long lines can be hard to read and may not fit on all screens.',
                'suggestions': [
                    'Break long lines into multiple shorter lines',
                    'Use parentheses for implicit line continuation',
                    'Consider extracting complex expressions into variables',
                    'Follow the 79-80 character limit recommendation'
                ],
                'resources': [
                    'PEP 8 line length guidelines',
                    'Code formatting best practices'
                ]
            },
            'deep_nesting': {
                'title': 'Deep Nesting Found',
                'explanation': 'Deeply nested code is harder to understand and maintain.',
                'suggestions': [
                    'Extract nested logic into separate functions',
                    'Use early returns to reduce nesting levels',
                    'Consider using guard clauses',
                    'Break complex conditions into boolean variables'
                ],
                'resources': [
                    'Refactoring techniques for reducing complexity',
                    'Clean Code principles'
                ]
            },
            'missing_docstring': {
                'title': 'Missing Documentation',
                'explanation': 'Documentation helps others understand your code\'s purpose and usage.',
                'suggestions': [
                    'Add docstrings to all functions and classes',
                    'Explain what the function does, its parameters, and return value',
                    'Use clear, concise language',
                    'Follow docstring conventions (e.g., Google style, NumPy style)'
                ],
                'resources': [
                    'PEP 257 -- Docstring Conventions',
                    'Writing effective documentation'
                ]
            },
            'magic_number': {
                'title': 'Magic Numbers Detected',
                'explanation': 'Hard-coded numbers make code less maintainable and unclear.',
                'suggestions': [
                    'Replace magic numbers with named constants',
                    'Use descriptive variable names for numeric values',
                    'Group related constants together',
                    'Add comments explaining the significance of specific values'
                ],
                'resources': [
                    'Clean Code: avoiding magic numbers',
                    'Constants and configuration best practices'
                ]
            },
            'empty_except': {
                'title': 'Empty Exception Handling',
                'explanation': 'Empty except blocks hide errors and make debugging difficult.',
                'suggestions': [
                    'Handle specific exception types instead of bare except',
                    'Log error information for debugging',
                    'Provide meaningful error messages to users',
                    'Only catch exceptions you can actually handle'
                ],
                'resources': [
                    'Python exception handling best practices',
                    'Error handling and logging patterns'
                ]
            }
        }

        self.quality_ranges = {
            'excellent': (90, 100),
            'good': (75, 89),
            'fair': (60, 74),
            'poor': (40, 59),
            'critical': (0, 39)
        }

        self.motivational_messages = {
            'excellent': [
                "Outstanding work! Your code demonstrates excellent quality practices.",
                "Fantastic! You're following best practices consistently.",
                "Excellent code quality! Keep up the great work!"
            ],
            'good': [
                "Great job! Your code is well-written with only minor improvements needed.",
                "Good work! You're on the right track with solid coding practices.",
                "Nice coding! Just a few tweaks and you'll have excellent code."
            ],
            'fair': [
                "Good foundation! Focus on the suggested improvements to enhance your code.",
                "You're making progress! Address the identified issues to improve quality.",
                "Solid start! Work on the feedback points to level up your coding skills."
            ],
            'poor': [
                "Keep learning! Every expert was once a beginner. Focus on the basics.",
                "Don't worry! Coding is a skill that improves with practice and patience.",
                "You're on a learning journey! Each improvement makes you a better programmer."
            ],
            'critical': [
                "Everyone starts somewhere! Focus on understanding the fundamentals first.",
                "Learning to code takes time! Start with fixing the most critical issues.",
                "Remember: every line of code is a step forward in your programming journey!"
            ]
        }

    def generate_feedback(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive educational feedback based on analysis results
        """
        issues = analysis_result.get('issues', [])
        quality_score = analysis_result.get('quality_score', 0)

        # Generate issue-specific feedback
        issue_feedback = self._generate_issue_feedback(issues)

        # Generate overall feedback
        overall_feedback = self._generate_overall_feedback(quality_score, len(issues))

        # Generate learning path
        learning_path = self._generate_learning_path(issues)

        return {
            'overall_assessment': overall_feedback,
            'specific_issues': issue_feedback,
            'learning_path': learning_path,
            'quality_metrics': {
                'score': quality_score,
                'grade': self._get_quality_grade(quality_score),
                'total_issues': len(issues),
                'severity_breakdown': analysis_result.get('severity_breakdown', {})
            }
        }

    def _generate_issue_feedback(self, issues: List[Dict]) -> List[Dict]:
        """
        Generate specific feedback for each type of issue found
        """
        issue_types = {}
        for issue in issues:
            issue_type = issue['issue_type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        feedback_list = []
        for issue_type, issue_list in issue_types.items():
            if issue_type in self.feedback_templates:
                template = self.feedback_templates[issue_type]
                feedback_list.append({
                    'issue_type': issue_type,
                    'count': len(issue_list),
                    'severity': issue_list[0]['severity'],
                    'title': template['title'],
                    'explanation': template['explanation'],
                    'suggestions': template['suggestions'],
                    'examples': [issue['code_snippet'] for issue in issue_list[:3]],  # Show up to 3 examples
                    'resources': template.get('resources', [])
                })

        return feedback_list

    def _generate_overall_feedback(self, quality_score: float, issue_count: int) -> Dict[str, Any]:
        """
        Generate overall assessment and motivational feedback
        """
        grade = self._get_quality_grade(quality_score)
        motivational_message = random.choice(self.motivational_messages[grade])

        if quality_score >= 90:
            summary = "Your code demonstrates excellent quality with minimal issues."
            next_steps = ["Continue following best practices", "Consider mentoring others", "Explore advanced patterns"]
        elif quality_score >= 75:
            summary = "Your code is well-written with room for minor improvements."
            next_steps = ["Address the identified issues", "Review coding standards", "Practice consistent formatting"]
        elif quality_score >= 60:
            summary = "Your code has a solid foundation but needs attention to quality practices."
            next_steps = ["Focus on the high-priority issues first", "Study clean code principles", "Use linting tools"]
        elif quality_score >= 40:
            summary = "Your code needs significant improvement in multiple areas."
            next_steps = ["Start with syntax and critical errors", "Learn fundamental best practices", "Practice regularly"]
        else:
            summary = "Your code requires substantial work on basic programming principles."
            next_steps = ["Focus on syntax and basic structure", "Review programming fundamentals", "Seek help from mentors"]

        return {
            'quality_grade': grade,
            'score': quality_score,
            'summary': summary,
            'motivational_message': motivational_message,
            'next_steps': next_steps,
            'issue_count': issue_count
        }

    def _generate_learning_path(self, issues: List[Dict]) -> Dict[str, Any]:
        """
        Generate a personalized learning path based on identified issues
        """
        high_priority = [issue for issue in issues if issue['severity'] == 'high']
        medium_priority = [issue for issue in issues if issue['severity'] == 'medium']
        low_priority = [issue for issue in issues if issue['severity'] == 'low']

        learning_path = {
            'immediate_focus': [],
            'short_term_goals': [],
            'long_term_improvement': []
        }

        # Immediate focus: High priority issues
        if high_priority:
            learning_path['immediate_focus'] = [
                "Fix all syntax errors and critical issues",
                "Ensure code can run without errors",
                "Address exception handling problems"
            ]

        # Short-term goals: Medium priority issues
        if medium_priority:
            learning_path['short_term_goals'] = [
                "Improve naming conventions",
                "Reduce code complexity and nesting",
                "Follow style guidelines consistently"
            ]

        # Long-term improvement: Low priority and general best practices
        learning_path['long_term_improvement'] = [
            "Add comprehensive documentation",
            "Optimize code performance",
            "Learn advanced design patterns",
            "Implement thorough testing"
        ]

        return learning_path

    def _get_quality_grade(self, score: float) -> str:
        """
        Convert quality score to grade category
        """
        for grade, (min_score, max_score) in self.quality_ranges.items():
            if min_score <= score <= max_score:
                return grade
        return 'critical'

    def get_score_based_feedback(self, score: float) -> Dict[str, str]:
        """
        Get contextual feedback based on quality score alone
        """
        grade = self._get_quality_grade(score)
        message = random.choice(self.motivational_messages[grade])

        return {
            'grade': grade,
            'message': message,
            'score': score
        }
