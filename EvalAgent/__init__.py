# -*- coding: utf-8 -*-
"""
EvalAgent 评估智能体框架
包含5个核心模块的独立评估系统
"""

from .data_collection_module import DataCollectionModule
from .simulation_environment_module import SimulationEnvironmentModule  
from .actual_effect_analysis_module import ActualEffectAnalysisModule
from .cognitive_impact_assessment_module import CognitiveImpactAssessmentModule
from .optimization_feedback_module import OptimizationFeedbackModule
from .eval_agent_core import EvalAgent, EvaluationMetrics, DistributionExpectation, create_eval_agent

__all__ = [
    'DataCollectionModule',
    'SimulationEnvironmentModule',
    'ActualEffectAnalysisModule', 
    'CognitiveImpactAssessmentModule',
    'OptimizationFeedbackModule',
    'EvalAgent',
    'EvaluationMetrics',
    'DistributionExpectation',
    'create_eval_agent'
]

__version__ = "1.0.0"
