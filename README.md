# BiasMandala: Semantic Coherence Analysis Framework

## Overview

BiasMandala is an advanced Python framework designed to assess and visualize semantic coherence within textual data. It leverages state-of-the-art natural language processing (NLP) techniques to identify and mitigate biases in machine learning models, ensuring fairness and accuracy in AI-driven applications.

## Features

| Feature                         | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| Semantic Analysis               | Evaluates the coherence and bias in textual data.             |
| Visualization Tools             | Provides graphical representations of semantic relationships. |
| Configurable Components         | Allows customization through YAML configuration files.        |
| Advanced Implementation Support | Includes advanced components for in-depth analysis.           |
| Project Specification Files     | Defines project parameters and specifications in JSON format. |

## Installation

```bash
git clone https://github.com/devedale/BiasMandala.git
cd BiasMandala
pip install -r requirements.txt
```

## Usage

```python
from BiasMandala import SemanticCoherenceAnalysisFramework as SCAF

scaf = SCAF(config_path='scaf_config.yaml')
scaf.analyze(text_data)
```

Ensure that your configuration file (`scaf_config.yaml`) is properly set up to match your project's specifications.

## Configuration

Example `scaf_config.yaml`:

```yaml
analysis:
  method: 'coherence'
  threshold: 0.75
visualization:
  enabled: true
  type: 'heatmap'
components:
  - name: 'bias_detector'
    enabled: true
```

## Examples

### Semantic Coherence Analysis

```python
text_data = ['This is a sample sentence.', 'Another example sentence.']
coherence_score = scaf.analyze(text_data)
print(f'Coherence Score: {coherence_score}')
```

### Visualization Output

![Coherence Heatmap](assets/coherence_heatmap.png)

This heatmap illustrates the semantic relationships between the provided sentences, highlighting areas of high and low coherence.

```
Semantic relationships (example visualization):
+-----------------+-----------------+
| Sentence 1      | Sentence 2      |
+-----------------+-----------------+
| 0.92            | 0.67            |
+-----------------+-----------------+
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

