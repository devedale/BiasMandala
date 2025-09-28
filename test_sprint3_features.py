import unittest
import numpy as np

from SemanticCoherenceAnalysisFramework import SemanticCoherenceAnalyzer, TopologicalSemanticObject
from advanced_components import EconomicTopologicalOptimizer

class TestSprint3EconomicOptimization(unittest.TestCase):

    def setUp(self):
        """Set up test data and configurations for optimization tests."""
        self.point_cloud = np.random.rand(50, 3)

        # Config with optimization DISABLED
        self.optimization_disabled_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": {"steps": 5}
            },
            "economic_optimization": {
                "enabled": False
            }
        }

        # Config with optimization ENABLED
        self.optimization_enabled_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": { "min_radius": 0.1, "max_radius": 2.0, "steps": 10 }
            },
            "economic_optimization": {
                "enabled": True,
                "budget": 100.0,
                "richness_weight": 0.5
            }
        }

        # Config with a very LOW budget
        self.low_budget_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": {"steps": 10}
            },
            "economic_optimization": {
                "enabled": True,
                "budget": 0.1, # Budget is too low for any scale
            }
        }

    def test_optimizer_initialization(self):
        """Verify that the optimizer is only initialized when enabled."""
        analyzer_disabled = SemanticCoherenceAnalyzer(config=self.optimization_disabled_config)
        self.assertIsNone(analyzer_disabled.economic_optimizer)

        analyzer_enabled = SemanticCoherenceAnalyzer(config=self.optimization_enabled_config)
        self.assertIsNotNone(analyzer_enabled.economic_optimizer)
        self.assertIsInstance(analyzer_enabled.economic_optimizer, EconomicTopologicalOptimizer)

    def test_analysis_with_optimization_disabled(self):
        """When optimization is disabled, it should perform a full multi-scale analysis."""
        analyzer = SemanticCoherenceAnalyzer(config=self.optimization_disabled_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        ts_object = results["ts_object"]
        expected_scales = self.optimization_disabled_config["multi_scale_topology"]["filtration_config"]["steps"]
        self.assertEqual(len(ts_object.multi_scale_topology), expected_scales)

    def test_analysis_with_optimization_enabled(self):
        """When optimization is enabled, it should analyze exactly one optimal scale."""
        analyzer = SemanticCoherenceAnalyzer(config=self.optimization_enabled_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        ts_object = results["ts_object"]
        self.assertEqual(len(ts_object.multi_scale_topology), 1)

        first_key = list(ts_object.multi_scale_topology.keys())[0]
        self.assertNotIn("error", ts_object.multi_scale_topology[first_key])

    def test_analysis_with_low_budget(self):
        """When the budget is too low, no scale should be analyzed."""
        analyzer = SemanticCoherenceAnalyzer(config=self.low_budget_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        ts_object = results["ts_object"]
        self.assertEqual(len(ts_object.multi_scale_topology), 0)

if __name__ == "__main__":
    unittest.main()