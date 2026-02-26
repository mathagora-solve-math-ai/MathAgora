import { useEffect, useMemo, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { AlignedModelResult, AlignedStep } from "../alignment.ts";
import {
  FLOW_COMMON_STAGE_TEMPLATES,
  type FlowCommonStageTemplate,
  type ModelResult,
} from "../mockData.ts";
import { MathText } from "./MathText.tsx";

type WaveFlowViewProps = {
  problemId?: string | null;
  problemText: string;
  structuredResults: ModelResult[];
  alignedResults: AlignedModelResult[];
};

type ViewDepth = 1 | 2 | 3;

type FlowNodeType = "problem" | "model" | "strategy" | "step" | "answer";

type FlowNode = {
  id: string;
  type: FlowNodeType;
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  body?: string;
  color?: string;
  isShared?: boolean;
  modelIds?: string[];
  modelId?: string;
  stepIndex?: number;
};

type FlowEdge = {
  id: string;
  from: string;
  to: string;
  modelId: string;
};

type StepCluster = {
  key: string;
  id: string;
  title: string;
  members: Array<{ modelId: string; step: AlignedStep; stepIndex: number }>;
  shared: boolean;
  modelIds: string[];
  meanStepIndex: number;
  meanModelIndex: number;
};

const MODEL_COLORS = [
  "#22d3ee",
  "#3b82f6",
  "#22c55e",
  "#f59e0b",
  "#f43f5e",
  "#a855f7",
  "#14b8a6",
];

const STOPWORDS = new Set([
  "the",
  "and",
  "or",
  "to",
  "of",
  "in",
  "for",
  "with",
  "a",
  "an",
  "is",
  "are",
  "이",
  "가",
  "은",
  "는",
  "을",
  "를",
  "의",
  "에",
  "에서",
  "으로",
  "로",
  "및",
  "또는",
  "그리고",
  "경우",
  "한다",
  "하기",
  "구하기",
]);

const hashString = (value: string) => {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const jitter = (seed: string, amount: number) =>
  ((hashString(seed) % 10000) / 10000 - 0.5) * amount;

const normalizeAnswer = (answer: string) =>
  answer
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[^\p{L}\p{N}\.\-+/]/gu, "")
    .trim();

const tokenize = (text: string) =>
  text
    .toLowerCase()
    .replace(/\\[a-zA-Z]+/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((token) => token && !STOPWORDS.has(token));

const summarizeClusterTitle = (steps: Array<{ title: string; body: string }>) => {
  const titleSet = Array.from(new Set(steps.map((step) => step.title.trim()).filter(Boolean)));
  if (titleSet.length === 1) return titleSet[0];

  const freq = new Map<string, number>();
  steps.forEach((step) => {
    tokenize(`${step.title} ${step.body}`).forEach((token) => {
      freq.set(token, (freq.get(token) ?? 0) + 1);
    });
  });

  const top = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([token]) => token);

  if (top.length > 0) return `공통 단계: ${top.join(" · ")}`;
  return titleSet[0] || "공통 풀이 단계";
};

const normalizeForMatch = (value: string) =>
  value.toLowerCase().replace(/\s+/g, "").replace(/\\/g, "");

const pickTemplateTitle = (
  templates: FlowCommonStageTemplate[],
  text: string,
): string | null => {
  const source = normalizeForMatch(text);
  let bestTitle: string | null = null;
  let bestScore = 0;

  templates.forEach((template) => {
    const score = template.keywords.reduce((acc, keyword) => {
      const hit = normalizeForMatch(keyword);
      return hit && source.includes(hit) ? acc + 1 : acc;
    }, 0);
    if (score > bestScore) {
      bestScore = score;
      bestTitle = template.title;
    }
  });

  return bestScore >= 2 ? bestTitle : null;
};

const colorByIndex = (index: number) => MODEL_COLORS[index % MODEL_COLORS.length];

const makeOffsets = (count: number, xGap: number, yGap: number) => {
  if (count <= 1) return [{ x: 0, y: 0 }];
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);
  return Array.from({ length: count }, (_, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    return {
      x: (col - (cols - 1) / 2) * xGap,
      y: (row - (rows - 1) / 2) * yGap,
    };
  });
};

const nodeStyle = (node: FlowNode) => {
  if (node.type === "problem") {
    return {
      background: "#ffffff",
      border: "1.2px solid #dbe2f0",
      title: "#475569",
      text: "#0f172a",
      sub: "#334155",
      glow: "none",
    };
  }

  if (node.type === "answer") {
    return {
      background: "#091022",
      border: "1.2px solid #b7f0e0",
      title: "#9cc7e9",
      text: "#e7ebff",
      sub: "#c8d0ee",
      glow: "0 0 0.65rem #34d39955, 0 0 1rem #22c55e33",
    };
  }

  const singleton = !node.color;
  if (singleton) {
    return {
      background: "#ffffff",
      border: "1.2px solid #dbe2f0",
      title: "#475569",
      text: "#0f172a",
      sub: "#334155",
      glow: "none",
    };
  }

  return {
    background: "#091022",
    border: `1.2px solid ${node.color}`,
    title: "#93a4c5",
    text: "#e7ebff",
    sub: "#c8d0ee",
    glow: `0 0 0.6rem ${node.color}55, 0 0 1rem ${node.color}30`,
  };
};

export default function WaveFlowView({
  problemId,
  problemText,
  structuredResults,
  alignedResults,
}: WaveFlowViewProps) {
  const [viewDepth, setViewDepth] = useState<ViewDepth>(2);
  const [nodeOffsets, setNodeOffsets] = useState<Record<string, { x: number; y: number }>>({});
  const [dragState, setDragState] = useState<{
    nodeId: string;
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startOffsetX: number;
    startOffsetY: number;
  } | null>(null);

  useEffect(() => {
    setNodeOffsets({});
    setDragState(null);
  }, [viewDepth, problemText, structuredResults.length, alignedResults.length]);

  const alignedByModel = useMemo(
    () => new Map(alignedResults.map((result) => [result.modelId, result])),
    [alignedResults],
  );

  const modelOrder = useMemo(
    () => new Map(structuredResults.map((result, index) => [result.modelId, index])),
    [structuredResults],
  );

  const modelXById = useMemo(() => {
    const map = new Map<string, number>();
    const modelCardWidth = 176;
    const modelGap = 16;
    const sidePadding = 34;
    structuredResults.forEach((result, index) => {
      map.set(result.modelId, sidePadding + index * (modelCardWidth + modelGap));
    });
    return map;
  }, [structuredResults]);

  const stepClusters = useMemo(() => {
    const groups = new Map<
      string,
      {
        steps: Array<{ modelId: string; step: AlignedStep; stepIndex: number }>;
        models: Set<string>;
      }
    >();

    structuredResults.forEach((result) => {
      const aligned = alignedByModel.get(result.modelId);
      const steps: AlignedStep[] = aligned
        ? aligned.steps
        : result.steps.map((step) => ({
            ...step,
            alignmentGroupId: undefined,
            similarityHint: undefined,
          }));
      steps.forEach((step, stepIndex) => {
        if (!step.alignmentGroupId) return;
        const current = groups.get(step.alignmentGroupId) ?? {
          steps: [],
          models: new Set<string>(),
        };
        current.steps.push({ modelId: result.modelId, step, stepIndex });
        current.models.add(result.modelId);
        groups.set(step.alignmentGroupId, current);
      });
    });

    const out: StepCluster[] = [];
    groups.forEach((group, id) => {
      const models = [...group.models];
      const meanStepIndex =
        group.steps.reduce((sum, step) => sum + step.stepIndex, 0) / group.steps.length;
      const meanModelIndex =
        models.reduce((sum, modelId) => sum + (modelOrder.get(modelId) ?? 0), 0) /
        Math.max(1, models.length);
      const defaultTitle = summarizeClusterTitle(
        group.steps.map((item) => ({
          title: item.step.title,
          body: item.step.body,
        })),
      );
      const problemTemplates = problemId
        ? FLOW_COMMON_STAGE_TEMPLATES[problemId] ?? []
        : [];
      const templateTitle =
        models.length >= 2
          ? pickTemplateTitle(
              problemTemplates,
              group.steps
                .map((item) => `${item.step.title} ${item.step.body}`)
                .join(" "),
            )
          : null;

      out.push({
        key: `shared:${id}`,
        id: `shared-${id}`,
        shared: models.length >= 2,
        title: templateTitle ?? defaultTitle,
        members: group.steps.map((item) => ({
          modelId: item.modelId,
          step: item.step,
          stepIndex: item.stepIndex,
        })),
        modelIds: models,
        meanStepIndex,
        meanModelIndex,
      });
    });

    structuredResults.forEach((result) => {
      const aligned = alignedByModel.get(result.modelId);
      const steps: AlignedStep[] = aligned
        ? aligned.steps
        : result.steps.map((step) => ({
            ...step,
            alignmentGroupId: undefined,
            similarityHint: undefined,
          }));
      steps.forEach((step, stepIndex) => {
        if (step.alignmentGroupId) return;
        const modelIdx = modelOrder.get(result.modelId) ?? 0;
        out.push({
          key: `solo:${result.modelId}:${stepIndex}`,
          id: `solo-${result.modelId}-${stepIndex}`,
          title: step.title || `Step ${stepIndex + 1}`,
          members: [{ modelId: result.modelId, step, stepIndex }],
          shared: false,
          modelIds: [result.modelId],
          meanStepIndex: stepIndex,
          meanModelIndex: modelIdx,
        });
      });
    });

    return out.sort((a, b) => {
      if (a.meanStepIndex !== b.meanStepIndex) return a.meanStepIndex - b.meanStepIndex;
      return a.meanModelIndex - b.meanModelIndex;
    });
  }, [alignedByModel, modelOrder, problemId, structuredResults]);

  const answerGroups = useMemo(() => {
    const grouped = new Map<string, { answer: string; models: ModelResult[] }>();

    structuredResults.forEach((result) => {
      const key = normalizeAnswer(result.finalAnswer) || "(empty)";
      const current = grouped.get(key) ?? {
        answer: result.finalAnswer || "(empty)",
        models: [],
      };
      current.models.push(result);
      grouped.set(key, current);
    });

    return [...grouped.entries()].map(([id, value]) => ({
      id,
      answer: value.answer,
      models: value.models,
    }));
  }, [structuredResults]);

  const { nodes, edges, width, height, modelColorById } = useMemo(() => {
    const nodesAcc: FlowNode[] = [];
    const edgesAcc: FlowEdge[] = [];

    const modelCount = structuredResults.length;
    const modelCardWidth = 176;
    const modelGap = 16;
    const sidePadding = 34;
    const flowWidth = Math.max(
      980,
      sidePadding * 2 + modelCount * modelCardWidth + Math.max(0, modelCount - 1) * modelGap,
    );

    const modelColorMap = new Map<string, string>();
    structuredResults.forEach((result, index) => {
      modelColorMap.set(result.modelId, colorByIndex(index));
    });

    const problemWidth = 320;
    const problemX = (flowWidth - problemWidth) / 2;

    nodesAcc.push({
      id: "problem",
      type: "problem",
      x: problemX,
      y: 36,
      width: problemWidth,
      height: 96,
      title: "Problem",
      body: problemText || "Selected problem",
    });

    const modelY = 186;
    const modelNodeById = new Map<string, string>();
    const tailNodeByModel = new Map<string, string>();

    structuredResults.forEach((result) => {
      const modelX = modelXById.get(result.modelId) ?? sidePadding;
      const modelNodeId = `model-${result.modelId}`;

      nodesAcc.push({
        id: modelNodeId,
        type: "model",
        x: modelX,
        y: modelY,
        width: modelCardWidth,
        height: 104,
        title: result.modelName,
        body: result.version,
        color: modelColorMap.get(result.modelId),
        modelId: result.modelId,
      });

      edgesAcc.push({
        id: `e-problem-${result.modelId}`,
        from: "problem",
        to: modelNodeId,
        modelId: result.modelId,
      });

      modelNodeById.set(result.modelId, modelNodeId);
      tailNodeByModel.set(result.modelId, modelNodeId);
    });

    if (viewDepth >= 2) {
      const strategyY = 330;
      structuredResults.forEach((result) => {
        const strategyNodeId = `strategy-${result.modelId}`;
        const modelX = modelXById.get(result.modelId) ?? sidePadding;

        nodesAcc.push({
          id: strategyNodeId,
          type: "strategy",
          x: modelX + jitter(`strategy-${result.modelId}`, 28),
          y: strategyY + jitter(`strategy-y-${result.modelId}`, 18),
          width: modelCardWidth,
          height: 84,
          title: "Strategy",
          body: viewDepth === 3 ? result.strategy : undefined,
          color: modelColorMap.get(result.modelId),
          modelId: result.modelId,
        });

        const from = modelNodeById.get(result.modelId) as string;
        edgesAcc.push({
          id: `e-model-strategy-${result.modelId}`,
          from,
          to: strategyNodeId,
          modelId: result.modelId,
        });

        tailNodeByModel.set(result.modelId, strategyNodeId);
      });
    }

    if (viewDepth === 2) {
      const clusterYStart = 468;
      const clusterGapY = 142;
      const centerX = flowWidth / 2;
      const modelMid = (modelCount - 1) / 2;
      const sharedOffsets = [-220, -80, 80, 220];
      const sharedRowByStage = new Map<number, number>();
      const soloRowByModel = new Map<string, number>();
      const stepNodeIdsByModel = new Map<string, Array<{ stepIndex: number; nodeId: string }>>();
      structuredResults.forEach((result) => stepNodeIdsByModel.set(result.modelId, []));

      stepClusters.forEach((cluster) => {
        const stage = Math.round(cluster.meanStepIndex);
        const stageY = clusterYStart + stage * clusterGapY;
        const nodeId = `cluster-title-${cluster.id}`;

        let x = centerX;
        let y = stageY + jitter(`title-y-${cluster.key}`, 40);

        if (cluster.shared) {
          const sharedRow = sharedRowByStage.get(stage) ?? 0;
          const slot = sharedRow % sharedOffsets.length;
          const band = Math.floor(sharedRow / sharedOffsets.length);
          x =
            centerX +
            sharedOffsets[slot] +
            band * 36 +
            jitter(`title-shared-x-${cluster.key}`, 54);
          y += band * 24;
          sharedRowByStage.set(stage, sharedRow + 1);
        } else {
          const ownerModelId = cluster.members[0]?.modelId ?? structuredResults[0]?.modelId;
          const ownerIdx = ownerModelId ? modelOrder.get(ownerModelId) ?? 0 : 0;
          const side = ownerIdx <= modelMid ? -1 : 1;
          const distance = 330 + Math.abs(ownerIdx - modelMid) * 82;
          const row = soloRowByModel.get(ownerModelId ?? "solo") ?? 0;
          x = centerX + side * distance + jitter(`title-solo-x-${cluster.key}`, 96);
          y += row * 34;
          soloRowByModel.set(ownerModelId ?? "solo", row + 1);
        }

        nodesAcc.push({
          id: nodeId,
          type: "step",
          x,
          y,
          width: 236,
          height: 82,
          title: "STEP",
          body: cluster.title,
          color: cluster.shared ? "#67e8f9" : undefined,
          isShared: cluster.shared,
          modelIds: cluster.modelIds,
        });

        cluster.members.forEach((member) => {
          const list = stepNodeIdsByModel.get(member.modelId) ?? [];
          list.push({ stepIndex: member.stepIndex, nodeId });
          stepNodeIdsByModel.set(member.modelId, list);
        });
      });

      structuredResults.forEach((result) => {
        const seq = (stepNodeIdsByModel.get(result.modelId) ?? []).sort(
          (a, b) => a.stepIndex - b.stepIndex,
        );
        seq.forEach((entry, index) => {
          const from =
            index === 0
              ? (tailNodeByModel.get(result.modelId) as string)
              : seq[index - 1].nodeId;
          edgesAcc.push({
            id: `e-title-${result.modelId}-${index}`,
            from,
            to: entry.nodeId,
            modelId: result.modelId,
          });
        });
        if (seq.length > 0) {
          tailNodeByModel.set(result.modelId, seq[seq.length - 1].nodeId);
        }
      });
    }

    if (viewDepth === 3) {
      const stepStartY = 498;
      const stepGapY = 176;
      const centerX = flowWidth / 2;
      const modelMid = (modelCount - 1) / 2;
      const sharedOffsets = [-250, -90, 90, 250];
      const sharedRowByStage = new Map<number, number>();
      const soloRowByModelAndStage = new Map<string, number>();
      const stepNodeIdsByModel = new Map<string, Array<{ stepIndex: number; nodeId: string }>>();
      structuredResults.forEach((result) => stepNodeIdsByModel.set(result.modelId, []));

      stepClusters.forEach((cluster) => {
        const stage = Math.round(cluster.meanStepIndex);
        const stageY = stepStartY + stage * stepGapY;
        let centerY = stageY + jitter(`detail-y-${cluster.key}`, 42);
        let centerClusterX = centerX;

        if (cluster.shared) {
          const sharedRow = sharedRowByStage.get(stage) ?? 0;
          const slot = sharedRow % sharedOffsets.length;
          const band = Math.floor(sharedRow / sharedOffsets.length);
          centerClusterX =
            centerX +
            sharedOffsets[slot] +
            band * 40 +
            jitter(`detail-shared-x-${cluster.key}`, 66);
          centerY += band * 34;
          sharedRowByStage.set(stage, sharedRow + 1);
        } else {
          const ownerModelId = cluster.members[0]?.modelId ?? structuredResults[0]?.modelId;
          const ownerIdx = ownerModelId ? modelOrder.get(ownerModelId) ?? 0 : 0;
          const side = ownerIdx <= modelMid ? -1 : 1;
          const distance = 430 + Math.abs(ownerIdx - modelMid) * 95;
          const key = `${ownerModelId ?? "solo"}:${stage}`;
          const row = soloRowByModelAndStage.get(key) ?? 0;
          centerClusterX =
            centerX + side * distance + jitter(`detail-solo-x-${cluster.key}`, 128);
          centerY += row * 58;
          soloRowByModelAndStage.set(key, row + 1);
        }

        const offsets = makeOffsets(
          cluster.members.length,
          cluster.shared ? 42 : 16,
          cluster.shared ? 30 : 18,
        );

        cluster.members.forEach((item, memberIndex) => {
          const offset = offsets[memberIndex] ?? { x: 0, y: 0 };
          const nodeId = `step-${item.modelId}-${item.step.id}-${item.stepIndex}`;

          nodesAcc.push({
            id: nodeId,
            type: "step",
            x: centerClusterX + offset.x,
            y: centerY + offset.y,
            width: 236,
            height: 118,
            title: item.step.title || `Step ${item.stepIndex + 1}`,
            body: item.step.body,
            color: cluster.shared ? "#67e8f9" : undefined,
            isShared: cluster.shared,
            modelId: item.modelId,
            stepIndex: item.stepIndex,
          });

          const list = stepNodeIdsByModel.get(item.modelId) ?? [];
          list.push({ stepIndex: item.stepIndex, nodeId });
          stepNodeIdsByModel.set(item.modelId, list);
        });
      });

      structuredResults.forEach((result) => {
        const seq = (stepNodeIdsByModel.get(result.modelId) ?? []).sort(
          (a, b) => a.stepIndex - b.stepIndex,
        );

        seq.forEach((entry, index) => {
          const from =
            index === 0
              ? (tailNodeByModel.get(result.modelId) as string)
              : seq[index - 1].nodeId;
          edgesAcc.push({
            id: `e-step-${result.modelId}-${index}`,
            from,
            to: entry.nodeId,
            modelId: result.modelId,
          });
        });

        if (seq.length > 0) {
          tailNodeByModel.set(result.modelId, seq[seq.length - 1].nodeId);
        }
      });
    }

    const maxBottom = Math.max(...nodesAcc.map((node) => node.y + node.height));
    const answerY = maxBottom + 170;

    const answerWidth = 246;
    const answerGap = 24;
    const totalAnswerWidth =
      answerGroups.length * answerWidth + Math.max(0, answerGroups.length - 1) * answerGap;
    const answerStartX = Math.max(24, (flowWidth - totalAnswerWidth) / 2);

    const answerNodeByModel = new Map<string, string>();
    answerGroups.forEach((group, index) => {
      const nodeId = `answer-${group.id}`;
      nodesAcc.push({
        id: nodeId,
        type: "answer",
        x: answerStartX + index * (answerWidth + answerGap),
        y: answerY,
        width: answerWidth,
        height: 118,
        title: "Answer",
        body: group.answer,
        modelIds: group.models.map((model) => model.modelId),
      });

      group.models.forEach((model) => {
        answerNodeByModel.set(model.modelId, nodeId);
      });
    });

    structuredResults.forEach((result) => {
      const from = tailNodeByModel.get(result.modelId) ?? `model-${result.modelId}`;
      const to = answerNodeByModel.get(result.modelId);
      if (!to) return;
      edgesAcc.push({
        id: `e-answer-${result.modelId}`,
        from,
        to,
        modelId: result.modelId,
      });
    });

    return {
      nodes: nodesAcc,
      edges: edgesAcc,
      width: flowWidth,
      height: answerY + 180,
      modelColorById: modelColorMap,
    };
  }, [
    alignedByModel,
    answerGroups,
    modelOrder,
    modelXById,
    problemText,
    stepClusters,
    structuredResults,
    viewDepth,
  ]);

  const positionedNodes = useMemo(
    () =>
      nodes.map((node) => {
        const offset = nodeOffsets[node.id] ?? { x: 0, y: 0 };
        return {
          ...node,
          x: node.x + offset.x,
          y: node.y + offset.y,
        };
      }),
    [nodes, nodeOffsets],
  );

  const nodeById = useMemo(
    () => new Map(positionedNodes.map((node) => [node.id, node])),
    [positionedNodes],
  );

  const canvasHeight = useMemo(
    () => Math.max(height, ...positionedNodes.map((node) => node.y + node.height + 40)),
    [height, positionedNodes],
  );

  const handleNodePointerDown = (
    nodeId: string,
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
    if (event.button !== 0) return;
    const currentOffset = nodeOffsets[nodeId] ?? { x: 0, y: 0 };
    setDragState({
      nodeId,
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startOffsetX: currentOffset.x,
      startOffsetY: currentOffset.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const dx = event.clientX - dragState.startClientX;
    const dy = event.clientY - dragState.startClientY;
    setNodeOffsets((prev) => ({
      ...prev,
      [dragState.nodeId]: {
        x: dragState.startOffsetX + dx,
        y: dragState.startOffsetY + dy,
      },
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    setDragState(null);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-2xl border border-slate-800 bg-[#070b14] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-100">Flow Map</div>
            <p className="mt-1 text-xs text-slate-300">
              모델 박스의 +/- 토글로 같은 맵을 축소/확장합니다.
            </p>
          </div>
          <span className="rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1 text-xs font-semibold text-slate-200">
            {viewDepth === 1 ? "요약" : viewDepth === 2 ? "중간" : "상세"}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-[#04070f] p-3">
        <div
          className="relative touch-none"
          style={{ width, height: canvasHeight }}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onPointerLeave={handlePointerUp}
        >
          <svg width={width} height={canvasHeight} className="absolute inset-0">
            <defs>
              <radialGradient id="bg" cx="50%" cy="48%" r="80%">
                <stop offset="0%" stopColor="#0f1630" />
                <stop offset="45%" stopColor="#090d1c" />
                <stop offset="100%" stopColor="#03050b" />
              </radialGradient>
              <filter id="edgeGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              {edges.map((edge) => {
                const from = nodeById.get(edge.from);
                const to = nodeById.get(edge.to);
                if (!from || !to) return null;
                const x1 = from.x + from.width / 2;
                const y1 = from.y + from.height;
                const x2 = to.x + to.width / 2;
                const y2 = to.y;
                const color = modelColorById.get(edge.modelId) ?? "#64748b";
                return (
                  <linearGradient
                    key={`grad-${edge.id}`}
                    id={`grad-${edge.id}`}
                    gradientUnits="userSpaceOnUse"
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                  >
                    <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                    <stop offset="40%" stopColor={color} stopOpacity="0.95" />
                    <stop offset="100%" stopColor={color} stopOpacity="0.2" />
                  </linearGradient>
                );
              })}
            </defs>

            <rect x={0} y={0} width={width} height={canvasHeight} fill="url(#bg)" rx={16} />

            {edges.map((edge) => {
              const from = nodeById.get(edge.from);
              const to = nodeById.get(edge.to);
              if (!from || !to) return null;
              const x1 = from.x + from.width / 2;
              const y1 = from.y + from.height;
              const x2 = to.x + to.width / 2;
              const y2 = to.y;
              const bendY = Math.max(54, (y2 - y1) * 0.45);
              const d = `M ${x1} ${y1} C ${x1} ${y1 + bendY}, ${x2} ${y2 - bendY}, ${x2} ${y2}`;
              return (
                <g key={edge.id}>
                  <path
                    d={d}
                    fill="none"
                    stroke={`url(#grad-${edge.id})`}
                    strokeWidth={7}
                    strokeOpacity={0.2}
                    filter="url(#edgeGlow)"
                  />
                  <path
                    d={d}
                    fill="none"
                    stroke={`url(#grad-${edge.id})`}
                    strokeWidth={2.2}
                    strokeOpacity={0.96}
                    strokeLinecap="round"
                  />
                </g>
              );
            })}
          </svg>

          <div className="absolute inset-0">
            {positionedNodes.map((node) => {
              const style = nodeStyle(node);
              const bodyMaxHeight =
                viewDepth === 3
                  ? node.type === "step"
                    ? 200
                    : 110
                  : node.type === "step"
                    ? 74
                    : 58;
              return (
                <div
                  key={node.id}
                  onPointerDown={(event) => handleNodePointerDown(node.id, event)}
                  className="absolute rounded-[10px]"
                  style={{
                    left: node.x,
                    top: node.y,
                    width: node.width,
                    minHeight: node.height,
                    background: style.background,
                    border: style.border,
                    boxShadow: style.glow,
                    cursor: dragState?.nodeId === node.id ? "grabbing" : "grab",
                    padding: "8px 10px 12px 10px",
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      color: style.title,
                      marginBottom: 5,
                      textTransform: "uppercase",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 8,
                    }}
                  >
                    <span>{node.title}</span>
                    {node.type === "model" ? (
                      <span
                        className="inline-flex items-center gap-1"
                        onPointerDown={(event) => event.stopPropagation()}
                      >
                        <button
                          type="button"
                          disabled={viewDepth === 1}
                          onClick={() =>
                            setViewDepth((prev) => Math.max(1, prev - 1) as ViewDepth)
                          }
                          className="h-5 w-5 rounded-full border border-slate-500 bg-slate-900/70 text-[10px] font-bold text-slate-200 disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          -
                        </button>
                        <button
                          type="button"
                          disabled={viewDepth === 3}
                          onClick={() =>
                            setViewDepth((prev) => Math.min(3, prev + 1) as ViewDepth)
                          }
                          className="h-5 w-5 rounded-full border border-slate-500 bg-slate-900/70 text-[10px] font-bold text-slate-200 disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          +
                        </button>
                      </span>
                    ) : null}
                  </div>
                  {node.body ? (
                    <div
                      className="flow-node-body"
                      style={{
                        fontSize: 12,
                        fontWeight: 500,
                        lineHeight: 1.35,
                        color: style.text,
                        wordBreak: "normal",
                        overflowWrap: "normal",
                        maxHeight: bodyMaxHeight,
                        overflow: "hidden",
                      }}
                    >
                      <MathText text={node.body} />
                    </div>
                  ) : null}

                  {node.type === "answer" && node.modelIds?.length ? (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {node.modelIds.map((modelId) => {
                        const idx = modelOrder.get(modelId) ?? 0;
                        const name =
                          structuredResults.find((result) => result.modelId === modelId)
                            ?.modelName ?? modelId;
                        return (
                          <span
                            key={`${node.id}-${modelId}`}
                            className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px]"
                            style={{
                              color: style.text,
                              borderColor: "#64748b",
                              backgroundColor: "rgba(15, 23, 42, 0.35)",
                            }}
                          >
                            <span
                              className="h-1.5 w-1.5 rounded-full"
                              style={{ backgroundColor: colorByIndex(idx) }}
                            />
                            {name}
                          </span>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-[#070b14] p-4">
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">
          Model Paths
        </div>
        <div className="mt-2 flex flex-wrap gap-3 text-sm">
          {structuredResults.map((result, idx) => (
            <div
              key={result.modelId}
              className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colorByIndex(idx) }}
              />
              <span className="text-slate-200">{result.modelName}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
