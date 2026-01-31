from typing import Dict
from pathlib import Path
from jinja2 import Template


class ReportGenerator:
    """Generate comprehensive evaluation reports"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, batch_eval, visualizations: Dict[str, str]):
        """Generate HTML report from evaluation results"""

        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report - {{ batch_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
                .metric-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .visualization { margin: 20px 0; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG Evaluation Report</h1>
                <p>Batch ID: {{ batch_id }}</p>
                <p>Date: {{ timestamp }}</p>
                <p>Total Questions: {{ question_count }}</p>
            </div>
            
            <h2>Aggregated Metrics</h2>
            <div class="metrics-grid">
                {% for name, value in aggregated_metrics.items() %}
                <div class="metric-card">
                    <h3>{{ name }}</h3>
                    <p>{{ value|round(3) }}</p>
                </div>
                {% endfor %}
            </div>
            
            <h2>Visualizations</h2>
            {% for name, img_path in visualizations.items() %}
            <div class="visualization">
                <h3>{{ name }}</h3>
                <img src="{{ img_path }}">
            </div>
            {% endfor %}
            
            <h2>Detailed Results</h2>
            <table border="1">
                <tr>
                    <th>Question</th>
                    <th>Expected Answer</th>
                    <th>Generated Answer</th>
                    <th>Execution Time</th>
                </tr>
                {% for q in questions %}
                <tr>
                    <td>{{ q.question[:50] }}...</td>
                    <td>{{ q.expected_answer[:50] }}...</td>
                    <td>{{ q.generated_answer[:50] }}...</td>
                    <td>{{ q.execution_time|round(2) }}s</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """

        # Render template
        template = Template(template_str)
        html_content = template.render(
            batch_id=batch_eval.batch_id,
            timestamp=batch_eval.timestamp,
            question_count=len(batch_eval.questions),
            aggregated_metrics=batch_eval.aggregated_metrics,
            questions=batch_eval.questions,
            visualizations=visualizations,
        )

        # Save HTML file
        report_path = self.output_dir / f"report_{batch_eval.batch_id}.html"
        report_path.write_text(html_content)

        return str(report_path)

    def generate_markdown_report(self, batch_eval, visualizations: Dict[str, str]):
        """Generate Markdown report"""

        md_content = f"""
# RAG Evaluation Report

**Batch ID**: {batch_eval.batch_id}
**Date**: {batch_eval.timestamp}
**Total Questions**: {len(batch_eval.questions)}

## Aggregated Metrics

| Metric | Value |
|--------|-------|
"""

        for metric_name, metric_value in batch_eval.aggregated_metrics.items():
            if isinstance(metric_value, dict):
                value_str = f"{metric_value.get('mean', 0):.3f}"
            else:
                value_str = f"{metric_value:.3f}"
            md_content += f"| {metric_name} | {value_str} |\n"

        md_content += "\n## Visualizations\n\n"
        for viz_name, viz_path in visualizations.items():
            md_content += f"### {viz_name}\n"
            md_content += f"![{viz_name}]({viz_path})\n\n"

        md_content += "## Sample Questions\n\n"
        for i, question in enumerate(batch_eval.questions[:5], 1):
            md_content += f"### Question {i}\n"
            md_content += f"**Question**: {question.question}\n\n"
            md_content += f"**Expected Answer**: {question.expected_answer}\n\n"
            md_content += f"**Generated Answer**: {question.generated_answer}\n\n"
            md_content += f"**Execution Time**: {question.execution_time:.2f}s\n\n"

        report_path = self.output_dir / f"report_{batch_eval.batch_id}.md"
        report_path.write_text(md_content)

        return str(report_path)
