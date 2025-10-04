# Semantic Coherence Analysis Framework (SCAF)

## 1. Overview

The Semantic Coherence Analysis Framework (SCAF) is a sophisticated, research-grade framework for analyzing the integrity of formal systems. Its primary purpose is to verify that when a transformation is applied to a complex, structured document (e.g., a scientific paper with LaTeX, a system diagram in Mermaid, etc.), the core semantic, topological, and geometric properties of the original are preserved.

In essence, SCAF answers the question: "After changing this document, does it still mean the same thing?"

This framework is designed for deep, mechanistic analysis, making it suitable for research in fields like AI safety, model interpretability, and formal verification.

## 2. Core Concepts

*   **Semantic Coherence**: The degree to which the meaning and logical structure of a system are preserved after a transformation.
*   **Invariant Metrics**: Quantifiable properties (topological, geometric, semantic) that should remain constant during a valid transformation. The framework measures `topological_fidelity` (Φt), `geometric_fidelity` (Φg), and `semantic_fidelity` (Φs).
*   **Bias Detection**: SCAF identifies systematic errors or "biases" that a transformation might introduce. It goes beyond simple error checking to find and diagnose the root causes of these distortions.
*   **Mechanistic Interpretability**: The framework includes tools to look inside "black box" models (like neural networks) to understand *why* a certain bias or error is occurring.

## 3. Key Features

*   **Configurable Analysis Pipeline**: SCAF uses a graph-based, multi-stage pipeline (`parser` -> `ast_extractor` -> `topology_analyzer`, etc.) that is fully defined and controlled via `scaf_config.yaml`. The execution order is determined by a topological sort of the dependency graph.

    ```python
    # From SemanticCoherenceAnalysisFramework.py
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Esecuzione topologicamente ordinata"""
        results = {"input": input_data}
        for node in nx.topological_sort(self.graph):
            stage = self.graph.nodes[node]["stage"]
            # ... (code omitted for brevity)
            result = stage.process(...)
            results[node] = result
        return results
    ```

*   **Advanced Topological Analysis**: Utilizes concepts from Topological Data Analysis (TDA) like persistent homology (via the `gudhi` library) to analyze the "shape" and structure of data.

    ```python
    # From advanced_components.py
    def compute_persistence_diagram(self, points: np.ndarray) -> List[Tuple]:
        """Calcola diagramma di persistenza"""
        rips_complex = gudhi.RipsComplex(points=points, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        persistence = simplex_tree.persistence()
        return self._process_persistence(persistence)
    ```

*   **Causal Inference Engine**: Includes modules to build and analyze causal graphs (`networkx`) to find the root causes of detected biases.

    ```python
    # From SemanticCoherenceAnalysisFramework.py
    def _analyze_causes(self, bias_type: str) -> List[str]:
        """Analisi causale del bias"""
        causes = []
        if bias_type in self.causal_graph:
            ancestors = nx.ancestors(self.causal_graph, bias_type)
            causes = list(ancestors)
        return causes
    ```
*   **Meta-Learning**: Features a `MetaLearner` component that can learn from past errors and suggest corrections for new ones.
*   **Extensible Architecture**: Designed to be easily extended with custom analyzers, validators, and transformation modules, as defined in the configuration.
*   **Multi-Language Support**: Capable of parsing and analyzing a variety of formal languages and diagramming tools.

## 4. Project Structure

This repository is organized into several key files that work together to form the framework.

*   `SemanticCoherenceAnalysisFramework.py`: This is the core of the project. It defines the main classes and architecture, including the `ConfigurablePipeline`, `BiasDetector`, `DeterministicValidator`, and the main `SemanticCoherenceAnalyzer` orchestrator.
*   `advanced_components.py`: This file contains highly advanced, research-oriented modules that extend the core framework. These include `PersistentHomologyAnalyzer`, the `CircuitDecomposer` for neural network interpretability, the `CausalInterventionEngine`, and the `MultimodalConsistencyAnalyzer`.
*   `scaf_advanced_implementation.py`: This file provides more concrete implementations for some of the advanced concepts, bridging the gap between the abstract components and practical analysis.
*   `scaf_config.yaml`: The central configuration file for the entire framework. It controls every aspect of the analysis, from defining the pipeline stages and their dependencies to setting weights for metrics and enabling experimental features.
*   `scaf_project_specification.json`: A high-level project manifest. It outlines the modular structure of the framework, lists the key technologies used in each module, and details the specific formal languages the project is designed to analyze.

*Note: The source code contains comments written in Italian.*

## 5. Configuration

The behavior of SCAF is almost entirely controlled by `scaf_config.yaml`. This allows for deep customization without changing the source code. Key configurable sections include:

*   **`pipeline`**: Define the sequence of analysis stages and their specific parameters.
*   **`invariants`**: Assign weights to the different fidelity metrics (topological, geometric, semantic).
*   **`bias_detection`**: Configure different types of bias detectors, their thresholds, and the causal model used for analysis.
*   **`meta_learning`**: Enable or disable the meta-learner and tune its parameters.
*   **`resources`**: Manage caching, parallelism, and memory limits.
*   **`monitoring`**: Configure logging, tracing (Jaeger), and metrics (Prometheus).
*   **`extensions`**: Register custom-built plugins for analysis, validation, or transformations.

## 6. Supported Formal Languages

As specified in the configuration and project specification, SCAF is designed to analyze a variety of formal languages, including:

*   LaTeX (including TikZ)
*   Mermaid
*   PlantUML
*   Graphviz (DOT)
*   Kroki

## 7. Theoretical Foundations and Further Reading

The methodologies used in SCAF are grounded in established and emerging academic research. For those interested in a deeper dive into the theoretical underpinnings, the following areas and publications are relevant.

*   **Topological Data Analysis (TDA)**: The use of persistent homology to understand the shape of data is a core component of SCAF's analysis. TDA is a powerful tool for finding robust structural features in complex datasets.
    *   *Further Reading*: For a comprehensive overview, consider resources like "A review of Topological Data Analysis and Machine Learning" which discusses the fusion of TDA with modern machine learning frameworks.

*   **Causal Inference and Interpretability**: The framework's `CausalInterventionEngine` and `BiasDetector` are inspired by the growing field of causal machine learning, which aims to move beyond correlation to understand cause-and-effect relationships within models.
    *   *Further Reading*: Papers such as "A Review of the Role of Causality in Developing Trustworthy AI Systems" provide a good starting point for how causality is key to building more reliable and interpretable AI.

*   **Formal Verification of Transformations**: The core goal of SCAF—ensuring a transformation preserves meaning—is related to the field of formal verification, particularly as it applies to model and program transformations.
    *   *Further Reading*: The domain of verifying model transformations is explored in works like "Model transformation specification for automated formal verification," which delves into strategies for ensuring correctness in automated model-based engineering.

## 8. Basic Usage Example

The following demonstrates a conceptual high-level use of the framework, as inspired by the main execution block in `SemanticCoherenceAnalysisFramework.py`.

```python
# main.py
from SemanticCoherenceAnalysisFramework import SemanticCoherenceAnalyzer

# Initialize the framework.
# Configuration is loaded automatically from scaf_config.yaml,
# but can be overridden.
analyzer = SemanticCoherenceAnalyzer({
    "bias_threshold": 0.5,
    "meta_learning_enabled": True
})

# Define a document in a supported formal language
latex_input = r"\begin{equation} E = mc^2 \end{equation}"

# Run the full analysis pipeline
results = analyzer.analyze(latex_input, language="latex")

# Print the results
print(f"Topological invariants: {results['topology']}")
print(f"Detected biases: {len(results['biases'])}")
print(f"Validation: {'PASSED' if results['valid'] else 'FAILED'}")

```