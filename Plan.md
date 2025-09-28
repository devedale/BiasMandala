# SCAF - Research Extensions Development Plan

## 1. Overview

This document outlines the development plan for the research extensions of the SCAF framework. The goal is to integrate advanced features for multi-scale topological analysis, topological-semantic fusion, and economic optimization, while maintaining a modular, configurable, and testable architecture.

The work is divided into four sequential sprints, each with clear objectives and specific deliverables. This plan serves as a roadmap and will be updated at the end of each sprint to reflect progress and adapt to new discoveries.

**Guiding Principles:**
- **Modularity:** Each component is self-contained and interacts through well-defined interfaces.
- **Configurability:** Key parameters (e.g., analysis scales, fusion strategies, optimization weights) are externally configurable.
- **Originality:** Where possible, innovative approaches will be explored, going beyond standard implementations.
- **Mentoring through Code:** Code and comments will be written not only to be functional, but also to explain the "why" behind architectural choices, acting as living documentation and a learning tool.

---

## 2. Sprint Structure

### Sprint 1: Foundations and Multi-Scale Topological Analysis

**Objective:** Build the foundation for future extensions by implementing a topological analyzer capable of operating at different resolution scales in a configurable manner.

**Dependencies:** None.

**Detailed Tasks:**
1.  **Create `MultiScaleTopologicalAnalyzer` Module:**
    -   Develop a new class `MultiScaleTopologicalAnalyzer` in `advanced_components.py`.
    -   This class will handle topological analysis at different resolutions.
2.  **Implement Configurable Scaling Strategy:**
    -   Introduce a "Strategy" design pattern to define scales. We will not be limited to just the Vietoris-Rips filtration value.
    -   **Strategy 1 (Filtration-based):** Analysis at different `max_edge_length` values for persistent homology.
    -   **Strategy 2 (Graph-based):** If the input is a graph, scales can be defined by decompositions (e.g., k-cores) or community detection resolutions.
    -   The strategy and its parameters will be definable via `scaf_config.yaml`.
3.  **Multi-Scale Output:**
    -   The `analyze` method will return a dictionary where keys are scale identifiers (e.g., `scale_0.5`, `scale_1.0`) and values are the topological invariants (persistence diagrams, Betti numbers) calculated at that scale.
4.  **Update Configuration:**
    -   Extend `scaf_config.yaml` to support the configuration of the `MultiScaleTopologicalAnalyzer` and its strategies.
5.  **Unit Testing:**
    -   Create a new test file to validate the `MultiScaleTopologicalAnalyzer` and its strategies.

**Deliverables:**
-   New `MultiScaleTopologicalAnalyzer` class in `advanced_components.py`.
-   Update to `scaf_config.yaml` with the new configuration section.
-   Unit tests for the new analyzer.

---

### Sprint 2: Topological-Semantic Fusion

**Objective:** Develop a mechanism to fuse multi-scale topological invariants with semantic information, creating a unified representation.

**Dependencies:** Sprint 1.

**Detailed Tasks:**
1.  **Design Unified Data Structure:**
    -   Create a new `TopologicalSemanticObject` class in `SemanticCoherenceAnalysisFramework.py`.
    -   This data structure will contain:
        -   The semantic embedding of the object.
        -   Multi-scale topological invariants (output of Sprint 1).
        -   A coherence graph connecting semantic nodes to topological features.
2.  **Implement `TopologicalSemanticFuser`:**
    -   Develop a new class `TopologicalSemanticFuser` in `advanced_components.py`.
    -   This component will take as input the output of the `MultiScaleTopologicalAnalyzer` and a semantic analyzer (e.g., `SemanticSpaceMapper`).
    -   Implement different (configurable) fusion strategies:
        -   **Simple Concatenation:** Joining the feature vectors.
        -   **Joint Embedding Learning:** Using a small autoencoder to learn a common latent representation.
        -   **Cross-Modal Attention:** An attention mechanism to weigh topological features based on the semantic context and vice-versa.
3.  **Pipeline Integration:**
    -   Add a `fusion` stage in the pipeline defined in `scaf_config.yaml`, which uses the `TopologicalSemanticFuser`.
4.  **Integration Testing:**
    -   Create tests to validate the fusion process and the `TopologicalSemanticObject`.

**Deliverables:**
-   New `TopologicalSemanticObject` class.
-   New `TopologicalSemanticFuser` class with multiple fusion strategies.
-   Updated pipeline in `scaf_config.yaml`.
-   Integration tests for the fusion.

---

### Sprint 3: Economic-Topological Optimization

**Objective:** Balance the computational cost of the analysis with the richness of the extracted topological information.

**Dependencies:** Sprint 1.

**Detailed Tasks:**
1.  **Extend `ComputationalEconomyOptimizer`:**
    -   Refactor the existing optimizer in `advanced_components.py` to make it more generic and suitable for selecting discrete scales.
2.  **Define Topological Objective Function:**
    -   The objective function will not optimize generic parameters, but the analysis "scale" of the `MultiScaleTopologicalAnalyzer`.
    -   The objective will be a configurable trade-off between:
        -   **Computational Cost:** Estimated based on the chosen scale.
        -   **Topological Richness:** A metric to quantify the informational value of invariants at a given scale.
3.  **Integration as a Meta-Stage:**
    -   Integrate the optimizer into the `ConfigurablePipeline` as a "meta-stage" that is executed before the topological analysis stage to configure it dynamically based on the budget and objectives defined in `scaf_config.yaml`.
4.  **Unit Testing:**
    -   Create tests to verify that the optimizer correctly selects scales based on budget and richness.

**Deliverables:**
-   Extended version of `ComputationalEconomyOptimizer`.
-   Integration of the optimizer into the `ConfigurablePipeline`.
-   Configuration examples for different trade-off scenarios.
-   Unit tests for the optimizer.

---

### Sprint 4: Benchmark Implementation

**Objective:** Create a benchmarking framework to validate the new features in a robust and reproducible way.

**Dependencies:** Sprints 1, 2, 3.

**Detailed Tasks:**
1.  **Create Benchmark Structure:**
    -   Create a new `benchmarks/` directory.
    -   Each benchmark will be an executable script (e.g., `run_stability_benchmark.py`) that loads a configuration, runs the analysis, and produces a report.
2.  **Benchmark 1: Topological Stability:**
    -   A script that accepts a dataset and a series of configurable perturbations.
    -   It runs the multi-scale topological analysis on the original and perturbed data.
    -   It measures stability by calculating the distance (e.g., bottleneck, Wasserstein) between the persistence diagrams.
    -   Output: A stability score for each scale.
3.  **Benchmark 2: Causal Interpretability Fidelity:**
    -   Will use the existing `CausalInterventionEngine`.
    -   The script will generate synthetic data based on a known causal graph (ground truth).
    -   It will run SCAF's causal discovery algorithm on the data.
    -   It will measure fidelity by comparing the inferred graph with the ground truth (e.g., Structural Hamming Distance).
4.  **Benchmark 3: Semantic and Multimodal Coherence:**
    -   Will use the existing `MultimodalConsistencyAnalyzer`.
    -   Create a small sample dataset with aligned triplets (e.g., `diagram.tex`, `description.txt`, `image.png`).
    -   The script will measure cross-modal coherence and the system's ability to perform retrieval tasks.

**Deliverables:**
-   `benchmarks/` directory with executable and configurable scripts.
-   Small sample datasets for each benchmark.
-   Documentation on how to run the benchmarks and interpret the results.