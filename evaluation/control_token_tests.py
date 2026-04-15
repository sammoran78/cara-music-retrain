"""
Control-token confound tests for CARA attribution.

Tests whether CARA codewords function as style controls rather than
faithful attribution signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mutual_info_score


class ControlTokenConfoundTester:
    """Test suite for control-token confound evaluation."""
    
    def __init__(self, model_path: Path, attribution_head_path: Path):
        self.model_path = model_path
        self.attribution_head_path = attribution_head_path
        self.results = {}
    
    def test_counterfactual_codeword_injection(
        self,
        test_prompts: list[str],
        codewords: list[str],
        num_seeds: int = 5
    ) -> dict[str, Any]:
        """
        Test 1: Counterfactual codeword injection.
        
        Hold prompt constant, vary only the codeword, measure output drift.
        If codewords are pure style controllers, outputs will shift significantly.
        """
        results = {
            "prompt_results": [],
            "average_drift": 0.0,
            "max_drift": 0.0
        }
        
        for prompt in test_prompts:
            prompt_data = {
                "prompt": prompt,
                "codeword_drifts": {}
            }
            
            # Generate baseline with first codeword
            baseline_codeword = codewords[0]
            baseline_outputs = []
            
            # TODO: Replace with actual model inference
            # baseline_outputs = self._generate_outputs(prompt, baseline_codeword, num_seeds)
            
            # Test each alternative codeword
            for alt_codeword in codewords[1:]:
                # TODO: Generate with alternative codeword
                # alt_outputs = self._generate_outputs(prompt, alt_codeword, num_seeds)
                
                # Compute embedding distance
                # drift = self._compute_output_drift(baseline_outputs, alt_outputs)
                drift = np.random.random()  # Placeholder
                
                prompt_data["codeword_drifts"][alt_codeword] = drift
            
            results["prompt_results"].append(prompt_data)
        
        # Aggregate statistics
        all_drifts = [
            drift 
            for r in results["prompt_results"] 
            for drift in r["codeword_drifts"].values()
        ]
        results["average_drift"] = np.mean(all_drifts)
        results["max_drift"] = np.max(all_drifts)
        results["interpretation"] = self._interpret_drift_scores(results["average_drift"])
        
        return results
    
    def test_attribution_invariance(
        self,
        test_prompts: list[str],
        num_variations: int = 10
    ) -> dict[str, Any]:
        """
        Test 2: Attribution invariance.
        
        For near-duplicate outputs (same prompt, different seeds),
        measure whether attribution is stable.
        """
        results = {
            "prompt_results": [],
            "average_stability": 0.0
        }
        
        for prompt in test_prompts:
            # TODO: Generate multiple variations
            # variations = self._generate_variations(prompt, num_variations)
            # attributions = [self._get_attribution(v) for v in variations]
            
            # Placeholder: simulate attribution distributions
            attributions = [
                {
                    "Freesound-CC0-Electronic": np.random.dirichlet([2, 1, 1, 1])[0],
                    "Freesound-CC-BY-Ambient": np.random.dirichlet([1, 2, 1, 1])[1],
                    "FMA-CC0-Jazz": np.random.dirichlet([1, 1, 2, 1])[2],
                    "FMA-CC-BY-Classical": np.random.dirichlet([1, 1, 1, 2])[3]
                }
                for _ in range(num_variations)
            ]
            
            # Compute stability metrics
            stability = self._compute_attribution_stability(attributions)
            
            results["prompt_results"].append({
                "prompt": prompt,
                "stability_score": stability,
                "num_variations": num_variations
            })
        
        results["average_stability"] = np.mean([
            r["stability_score"] for r in results["prompt_results"]
        ])
        results["interpretation"] = self._interpret_stability_scores(results["average_stability"])
        
        return results
    
    def test_mutual_information(
        self,
        generated_samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Test 3: Mutual information analysis.
        
        Estimate MI between emitted codewords and output features
        beyond what is explained by ground-truth pool membership.
        """
        # Extract features and attributions
        features = []
        attributions = []
        ground_truth = []
        
        for sample in generated_samples:
            # TODO: Extract actual features from generated audio
            # features.append(self._extract_audio_features(sample["audio"]))
            features.append(np.random.randn(128))  # Placeholder feature vector
            
            attributions.append(sample.get("attributed_pool", "Unknown"))
            ground_truth.append(sample.get("ground_truth_pool", "Unknown"))
        
        # Compute mutual information
        mi_attribution_features = mutual_info_score(attributions, 
                                                   self._discretize_features(features))
        mi_ground_truth_features = mutual_info_score(ground_truth,
                                                    self._discretize_features(features))
        
        excess_mi = mi_attribution_features - mi_ground_truth_features
        
        results = {
            "mi_attribution_features": mi_attribution_features,
            "mi_ground_truth_features": mi_ground_truth_features,
            "excess_mutual_information": excess_mi,
            "interpretation": self._interpret_mi_scores(excess_mi)
        }
        
        return results
    
    def _compute_attribution_stability(self, attributions: list[dict[str, float]]) -> float:
        """Compute stability of attribution across variations."""
        if len(attributions) < 2:
            return 1.0
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(attributions)):
            for j in range(i + 1, len(attributions)):
                # Convert to vectors for comparison
                pools = sorted(set(attributions[i].keys()) | set(attributions[j].keys()))
                vec1 = [attributions[i].get(p, 0.0) for p in pools]
                vec2 = [attributions[j].get(p, 0.0) for p in pools]
                
                # Cosine similarity
                if np.any(vec1) and np.any(vec2):
                    sim = 1 - cosine(vec1, vec2)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _discretize_features(self, features: list[np.ndarray]) -> list[int]:
        """Discretize continuous features for MI calculation."""
        # Simple k-means clustering for discretization
        from sklearn.cluster import KMeans
        
        features_array = np.array(features)
        n_clusters = min(10, len(features) // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(features_array).tolist()
    
    def _interpret_drift_scores(self, avg_drift: float) -> str:
        """Interpret drift scores."""
        if avg_drift < 0.1:
            return "Low drift: Attribution likely faithful"
        elif avg_drift < 0.3:
            return "Moderate drift: Some style control present"
        else:
            return "High drift: Strong style control confound"
    
    def _interpret_stability_scores(self, avg_stability: float) -> str:
        """Interpret stability scores."""
        if avg_stability > 0.9:
            return "High stability: Consistent attribution"
        elif avg_stability > 0.7:
            return "Moderate stability: Generally consistent"
        else:
            return "Low stability: Inconsistent attribution"
    
    def _interpret_mi_scores(self, excess_mi: float) -> str:
        """Interpret mutual information scores."""
        if excess_mi < 0.1:
            return "Low excess MI: Attribution aligned with ground truth"
        elif excess_mi < 0.3:
            return "Moderate excess MI: Some style leakage"
        else:
            return "High excess MI: Significant style control"
    
    def run_all_tests(
        self,
        test_prompts: list[str],
        codewords: list[str],
        generated_samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run all control-token confound tests."""
        
        print("Running counterfactual codeword injection test...")
        injection_results = self.test_counterfactual_codeword_injection(
            test_prompts, codewords
        )
        
        print("Running attribution invariance test...")
        invariance_results = self.test_attribution_invariance(test_prompts)
        
        print("Running mutual information analysis...")
        mi_results = self.test_mutual_information(generated_samples)
        
        # Aggregate results
        overall_results = {
            "counterfactual_injection": injection_results,
            "attribution_invariance": invariance_results,
            "mutual_information": mi_results,
            "overall_assessment": self._overall_assessment(
                injection_results, invariance_results, mi_results
            )
        }
        
        return overall_results
    
    def _overall_assessment(
        self,
        injection_results: dict[str, Any],
        invariance_results: dict[str, Any],
        mi_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Provide overall assessment of control-token confound."""
        
        # Score each dimension
        injection_score = 1.0 - min(injection_results["average_drift"], 1.0)
        invariance_score = invariance_results["average_stability"]
        mi_score = 1.0 - min(mi_results["excess_mutual_information"], 1.0)
        
        overall_score = np.mean([injection_score, invariance_score, mi_score])
        
        assessment = {
            "scores": {
                "injection_faithfulness": injection_score,
                "attribution_stability": invariance_score,
                "ground_truth_alignment": mi_score,
                "overall": overall_score
            },
            "interpretation": self._interpret_overall_score(overall_score),
            "recommendations": self._generate_recommendations(
                injection_score, invariance_score, mi_score
            )
        }
        
        return assessment
    
    def _interpret_overall_score(self, score: float) -> str:
        """Interpret overall faithfulness score."""
        if score > 0.8:
            return "High faithfulness: Attribution appears reliable"
        elif score > 0.6:
            return "Moderate faithfulness: Some control-token behavior present"
        else:
            return "Low faithfulness: Significant control-token confound"
    
    def _generate_recommendations(
        self,
        injection_score: float,
        invariance_score: float,
        mi_score: float
    ) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if injection_score < 0.7:
            recommendations.append(
                "Consider architectural changes to reduce codeword influence on style"
            )
        
        if invariance_score < 0.7:
            recommendations.append(
                "Improve attribution head consistency across similar outputs"
            )
        
        if mi_score < 0.7:
            recommendations.append(
                "Strengthen alignment between attribution and ground-truth pools"
            )
        
        if not recommendations:
            recommendations.append("Attribution system appears well-calibrated")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    tester = ControlTokenConfoundTester(
        model_path=Path("models/cara_finetuned"),
        attribution_head_path=Path("models/attribution_head")
    )
    
    test_prompts = [
        "Electronic dance music with heavy bass",
        "Calm ambient soundscape",
        "Upbeat jazz piano trio",
        "Classical string quartet"
    ]
    
    codewords = [
        "Freesound-CC0-Electronic",
        "Freesound-CC-BY-Ambient",
        "FMA-CC0-Jazz",
        "FMA-CC-BY-Classical"
    ]
    
    # Placeholder for actual generated samples
    generated_samples = [
        {"attributed_pool": "Freesound-CC0-Electronic", "ground_truth_pool": "Freesound-CC0-Electronic"},
        {"attributed_pool": "FMA-CC0-Jazz", "ground_truth_pool": "FMA-CC0-Jazz"},
    ]
    
    results = tester.run_all_tests(test_prompts, codewords, generated_samples)
    
    # Save results
    output_path = Path("evaluation/control_token_test_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print(f"Overall assessment: {results['overall_assessment']['interpretation']}")
