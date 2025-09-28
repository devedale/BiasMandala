import unittest
import numpy as np
import networkx as nx
import yaml

from SemanticCoherenceAnalysisFramework import SemanticCoherenceAnalyzer, TopologicalSemanticObject
from advanced_components import MultiScaleTopologicalAnalyzer, FiltrationScalingStrategy, GraphDecompositionStrategy

class TestSprint1MultiScaleAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test data and configurations for Sprint 1 tests."""
        self.point_cloud = np.random.rand(50, 3)
        self.graph = nx.erdos_renyi_graph(30, 0.4)

        # Basic config needed for the analyzer to run
        self.filtration_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": { "steps": 5 }
            }
        }

        self.graph_config = {
            "multi_scale_topology": {
                "strategy": "graph_decomposition"
            }
        }

    def test_filtration_strategy_initialization(self):
        """Verify that the analyzer correctly initializes with the filtration strategy."""
        analyzer = SemanticCoherenceAnalyzer(config=self.filtration_config)
        self.assertIsNotNone(analyzer.multi_scale_analyzer)
        self.assertIsInstance(analyzer.multi_scale_analyzer.strategy, FiltrationScalingStrategy)

    def test_graph_decomposition_strategy_initialization(self):
        """Verify that the analyzer correctly initializes with the graph decomposition strategy."""
        analyzer = SemanticCoherenceAnalyzer(config=self.graph_config)
        self.assertIsNotNone(analyzer.multi_scale_analyzer)
        self.assertIsInstance(analyzer.multi_scale_analyzer.strategy, GraphDecompositionStrategy)

    def test_analysis_with_filtration_strategy(self):
        """Test that a multi-scale analysis runs and populates the ts_object."""
        analyzer = SemanticCoherenceAnalyzer(config=self.filtration_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        self.assertIn("ts_object", results)
        ts_object = results["ts_object"]
        self.assertIsInstance(ts_object, TopologicalSemanticObject)
        topology_results = ts_object.multi_scale_topology

        expected_steps = self.filtration_config["multi_scale_topology"]["filtration_config"]["steps"]
        self.assertEqual(len(topology_results), expected_steps)

        first_scale_key = list(topology_results.keys())[0]
        self.assertIn("betti_numbers", topology_results[first_scale_key])

    def test_analysis_with_graph_strategy(self):
        """Test that a graph-based multi-scale analysis runs and populates the ts_object."""
        analyzer = SemanticCoherenceAnalyzer(config=self.graph_config)
        results = analyzer.analyze(input_data=self.graph, data_type='graph')

        self.assertIn("ts_object", results)
        ts_object = results["ts_object"]
        self.assertIsInstance(ts_object, TopologicalSemanticObject)
        topology_results = ts_object.multi_scale_topology

        core_numbers = nx.core_number(self.graph)
        max_k_core = max(core_numbers.values()) if core_numbers else 0
        self.assertEqual(len(topology_results), max_k_core)

    def test_invalid_strategy_configuration(self):
        """Test that the framework raises an error for an unknown strategy."""
        invalid_config = {"multi_scale_topology": {"strategy": "non_existent_strategy"}}
        with self.assertRaises(ValueError):
            SemanticCoherenceAnalyzer(config=invalid_config)

if __name__ == "__main__":
    unittest.main()