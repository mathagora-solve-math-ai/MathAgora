"""Flow Map Generator - Input/Output Schemas"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Step:
    """Individual step from an LLM solution"""
    step_idx: int
    title: str
    content: str


@dataclass
class ModelSolution:
    """Solution from a single model"""
    model_name: str
    steps: List[Step]


@dataclass
class FlowMapInput:
    """Input to Flow Map Generator"""
    problem_text: str
    solutions: List[ModelSolution]


# ─────────────────────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────────────────────


@dataclass
class GroupedStep:
    """A step within a group, maintaining its model and original index"""
    model: str
    step_idx: int
    title: str
    content: str


@dataclass
class StepGroup:
    """A group of similar steps from different models"""
    group_id: int
    group_name: str  # 중제목 (e.g., "도함수 구하기")
    steps: List[GroupedStep]


@dataclass
class FlowConnection:
    """Flow connection from one step to another (within same model)"""
    model: str
    from_step: int
    to_step: int


@dataclass
class FlowMap:
    """Complete Flow Map output"""
    groups: List[StepGroup]
    flows: List[FlowConnection]

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            "groups": [
                {
                    "group_id": g.group_id,
                    "group_name": g.group_name,
                    "steps": [
                        {
                            "model": s.model,
                            "step_idx": s.step_idx,
                            "title": s.title,
                            "content": s.content
                        }
                        for s in g.steps
                    ]
                }
                for g in self.groups
            ],
            "flows": [
                {
                    "model": f.model,
                    "from_step": f.from_step,
                    "to_step": f.to_step
                }
                for f in self.flows
            ]
        }
