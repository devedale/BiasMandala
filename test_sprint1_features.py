import unittest
import numpy as np
import networkx as nx
import yaml

from SemanticCoherenceAnalysisFramework import SemanticCoherenceAnalyzer
from advanced_components import MultiScaleTopologicalAnalyzer, FiltrationScalingStrategy, GraphDecompositionStrategy

class TestMultiScaleTopologicalAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test data and configurations."""
        self.point_cloud = np.random.rand(50, 3)
        self.graph = nx.erdos_renyi_graph(30, 0.4)

        self.filtration_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": {
                    "min_radius": 0.1,
                    "max_radius": 1.0,
                    "steps": 5
                }
            }
        }

        self.graph_config = {
            "multi_scale_topology": {
                "strategy": "graph_decomposition"
            }
        }

    def test_filtration_strategy_initialization(self):
        """
        Verify that the analyzer correctly initializes with the filtration strategy.
        """
        analyzer = SemanticCoherenceAnalyzer(config=self.filtration_config)
        self.assertIsNotNone(analyzer.multi_scale_analyzer)
        self.assertIsInstance(analyzer.multi_scale_analyzer.strategy, FiltrationScalingStrategy)
        print("Test Filtration Strategy Initialization: PASSED")

    def test_graph_decomposition_strategy_initialization(self):
        """
        Verify that the analyzer correctly initializes with the graph decomposition strategy.
        """
        analyzer = SemanticCoherenceAnalyzer(config=self.graph_config)
        self.assertIsNotNone(analyzer.multi_scale_analyzer)
        self.assertIsInstance(analyzer.multi_scale_analyzer.strategy, GraphDecompositionStrategy)
        print("Test Graph Decomposition Strategy Initialization: PASSED")

    def test_analysis_with_filtration_strategy(self):
        """
        Test a full analysis run using point cloud data and the filtration strategy.
        """
        analyzer = SemanticCoherenceAnalyzer(config=self.filtration_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        self.assertIn("multi_scale_topology", results)
        topology_results = results["multi_scale_topology"]

        expected_steps = self.filtration_config["multi_scale_topology"]["filtration_config"]["steps"]
        self.assertEqual(len(topology_results), expected_steps)

        first_scale_key = list(topology_results.keys())[0]
        self.assertIn("betti_numbers", topology_results[first_scale_key])
        self.assertIn("persistence_diagram", topology_results[first_scale_key])
        print("Test Analysis with Filtration Strategy: PASSED")

    def test_analysis_with_graph_strategy(self):
        """
        Test a full analysis run using graph data and the graph decomposition strategy.
        """
        analyzer = SemanticCoherenceAnalyzer(config=self.graph_config)
        results = analyzer.analyze(input_data=self.graph, data_type='graph')

        self.assertIn("multi_scale_topology", results)
        topology_results = results["multi_scale_topology"]

        core_numbers = nx.core_number(self.graph)
        max_k_core = max(core_numbers.values()) if core_numbers else 0
        self.assertEqual(len(topology_results), max_k_core)

        if max_k_core > 0:
            first_scale_key = list(topology_results.keys())[0]
            self.assertIn("betti_numbers", topology_results[first_scale_key])
            self.assertIn("persistence_diagram", topology_results[first_scale_key])
        print("Test Analysis with Graph Strategy: PASSED")

    def test_invalid_strategy_configuration(self):
        """
        Test that the framework raises an error for an unknown strategy.
        """
        invalid_config = {
            "multi_scale_topology": {
                "strategy": "non_existent_strategy"
            }
        }
        with self.assertRaises(ValueError):
            SemanticCoherenceAnalyzer(config=invalid_config)
        print("Test Invalid Strategy Configuration: PASSED")

if __name__ == "__main__":
    print("Running SCAF Sprint 1 Feature Tests...")
    unittest.main()
    print("SCAF Sprint 1 Feature Tests Completed.")