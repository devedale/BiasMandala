import unittest
import numpy as np

from SemanticCoherenceAnalysisFramework import SemanticCoherenceAnalyzer, TopologicalSemanticObject
from advanced_components import TopologicalSemanticFuser, ConcatenationFusionStrategy, AutoencoderFusionStrategy

class TestSprint2TopologicalSemanticFusion(unittest.TestCase):

    def setUp(self):
        """Set up test data and configurations for Sprint 2 fusion tests."""
        self.point_cloud = np.random.rand(50, 3)

        self.concatenation_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": {"steps": 2}
            },
            "fusion": {
                "strategy": "concatenation"
            }
        }

        self.autoencoder_config = {
            "multi_scale_topology": {
                "strategy": "filtration",
                "filtration_config": {"steps": 2}
            },
            "fusion": {
                "strategy": "autoencoder",
                "autoencoder_config": { "model_path": None }
            }
        }

    def test_fuser_initialization_concatenation(self):
        """Verify that the analyzer correctly initializes the fuser with the concatenation strategy."""
        analyzer = SemanticCoherenceAnalyzer(config=self.concatenation_config)
        self.assertIsNotNone(analyzer.fuser)
        self.assertIsInstance(analyzer.fuser.strategy, ConcatenationFusionStrategy)

    def test_fuser_initialization_autoencoder(self):
        """Verify that the analyzer correctly initializes the fuser with the autoencoder strategy."""
        analyzer = SemanticCoherenceAnalyzer(config=self.autoencoder_config)
        self.assertIsNotNone(analyzer.fuser)
        self.assertIsInstance(analyzer.fuser.strategy, AutoencoderFusionStrategy)

    def test_analysis_with_fusion_produces_valid_object(self):
        """Test a full analysis run to ensure the fusion process creates a valid TopologicalSemanticObject."""
        analyzer = SemanticCoherenceAnalyzer(config=self.concatenation_config)
        results = analyzer.analyze(input_data=self.point_cloud, data_type='point_cloud')

        self.assertIn("ts_object", results)
        ts_object = results["ts_object"]
        self.assertIsInstance(ts_object, TopologicalSemanticObject)

        self.assertIsNotNone(ts_object.fused_representation)
        self.assertIsInstance(ts_object.fused_representation, np.ndarray)

        # Semantic embedding (placeholder) is 768.
        # Topology featurization is 7 (4 Betti numbers + 3 lifetime stats).
        expected_dim = 768 + 7
        self.assertEqual(ts_object.fused_representation.shape[0], expected_dim)

    def test_invalid_fusion_strategy_configuration(self):
        """Test that the framework raises an error for an unknown fusion strategy."""
        invalid_config = {
            "fusion": { "strategy": "non_existent_strategy" }
        }
        with self.assertRaises(ValueError):
            SemanticCoherenceAnalyzer(config=invalid_config)

if __name__ == "__main__":
    unittest.main()