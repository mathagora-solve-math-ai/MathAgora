import { useEffect, useMemo, useRef, useState } from "react";
import { MathText } from "./MathText.tsx";
import { FLOWMAP_GROUP_COLORS, ANSWER_PALETTE } from "../theme/flowmapPalette.ts";

/** Model logo path by provider prefix; same as ModelOutputView. */
function getModelLogo(modelId: string): string {
  if (modelId.startsWith("openai/")) return "/logos/openai.png";
  if (modelId.startsWith("anthropic/")) return "/logos/claude.png";
  if (modelId.startsWith("google/")) return "/logos/gemini.jpeg";
  if (modelId.startsWith("x-ai/")) return "/logos/grok.png";
  return "/logos/openai.png";
}

export type FlowMapStep = {
  model: string;
  step_idx: number;
  title: string;
  content: string;
};

export type FlowMapGroup = {
  group_id: number;
  group_name: string;
  steps: FlowMapStep[];
};

export type FlowMapData = {
  groups: FlowMapGroup[];
  flows: { model: string; from_step: number; to_step: number }[];
};

type ModelInfo = {
  name: string;
  color: string;
  modelId?: string;
};

type Props = {
  data: FlowMapData;
  modelInfos: ModelInfo[];
  finalAnswers?: Record<string, string>;
};

const SIDEBAR_W = 148;
const HEADER_H = 48;
const MIN_COL_W = 160;
const COL_GAP = 12;
const ROW_GAP = 10;
const MIN_ROW_H = 72;
const OUTER_PAD = 16;
const STEP_CARD_H = 64;
const EXPANDED_CONTENT_H = 200;
const EXPANDED_CARD_H = STEP_CARD_H + EXPANDED_CONTENT_H;
const FINAL_ROW_H = 72;
const FINAL_ROW_EXTRA_GAP = 18;

function hexToRgba(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const normalized =
    value.length === 3
      ? value
          .split("")
          .map((ch) => ch + ch)
          .join("")
      : value;
  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function FlowMapView({ data, modelInfos, finalAnswers }: Props) {
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const modelNames = useMemo(() => modelInfos.map((m) => m.name), [modelInfos]);

  const colorByGroup = useMemo(
    () =>
      new Map(
        data.groups.map((group) => [
          group.group_id,
          FLOWMAP_GROUP_COLORS[group.group_id % FLOWMAP_GROUP_COLORS.length],
        ]),
      ),
    [data.groups],
  );
  const groupLegendItems = useMemo(
    () =>
      data.groups.map((group) => ({
        groupId: group.group_id,
        color:
          colorByGroup.get(group.group_id) ??
          FLOWMAP_GROUP_COLORS[group.group_id % FLOWMAP_GROUP_COLORS.length],
      })),
    [data.groups, colorByGroup],
  );

  // Dynamic column width: fill available container width
  const colW = useMemo(() => {
    if (containerWidth <= 0 || modelNames.length === 0) return MIN_COL_W;
    const available = containerWidth - OUTER_PAD * 2 - SIDEBAR_W - COL_GAP * (modelNames.length + 1);
    return Math.max(MIN_COL_W, Math.floor(available / modelNames.length));
  }, [containerWidth, modelNames.length]);

  const totalWidth = containerWidth > 0
    ? containerWidth
    : OUTER_PAD + SIDEBAR_W + COL_GAP + modelNames.length * (MIN_COL_W + COL_GAP) + OUTER_PAD;

  // Row heights: expand the selected group row
  const rowHeights = useMemo(
    () =>
      data.groups.map((group) => {
        const maxSteps = Math.max(
          1,
          ...modelNames.map(
            (name) => group.steps.filter((s) => s.model === name).length,
          ),
        );
        const cardH =
          group.group_id === expandedGroupId ? EXPANDED_CARD_H : STEP_CARD_H;
        return Math.max(MIN_ROW_H, maxSteps * (cardH + 6) + 12);
      }),
    [data.groups, modelNames, expandedGroupId],
  );

  // Cumulative row top positions
  const rowTops = useMemo(() => {
    const tops: number[] = [];
    let y = OUTER_PAD + HEADER_H;
    rowHeights.forEach((h) => {
      tops.push(y);
      y += h + ROW_GAP;
    });
    return tops;
  }, [rowHeights]);

  const hasFinalAnswers =
    !!finalAnswers && Object.keys(finalAnswers).some((k) => !!finalAnswers[k]);

  /** Same value → same color as Model Output. Empty/— → gray; rest sorted and same palette. */
  const answerValueToColor = useMemo(() => {
    if (!finalAnswers) return new Map<string, { bg: string; border: string }>();
    const values = [
      ...new Set(
        Object.values(finalAnswers)
          .filter((v) => v != null && String(v).trim() !== "" && String(v).trim() !== "—"),
      ),
    ].sort((a, b) => String(a).localeCompare(String(b)));
    const map = new Map<string, { bg: string; border: string }>();
    values.forEach((v, i) => {
      const c = ANSWER_PALETTE[i % ANSWER_PALETTE.length];
      map.set(String(v).trim(), c);
    });
    return map;
  }, [finalAnswers]);

  const emptyAnswerStyle = { bg: "#f1f5f9", border: "#e2e8f0" };

  const finalRowTop =
    (rowTops.at(-1) ?? OUTER_PAD + HEADER_H) +
    (rowHeights.at(-1) ?? 0) +
    ROW_GAP +
    FINAL_ROW_EXTRA_GAP;

  const totalHeight = hasFinalAnswers
    ? finalRowTop + FINAL_ROW_H + OUTER_PAD * 2
    : (rowTops.at(-1) ?? OUTER_PAD + HEADER_H) +
      (rowHeights.at(-1) ?? 0) +
      OUTER_PAD * 2;

  const colCenterX = useMemo(
    () =>
      modelNames.map(
        (_, i) =>
          OUTER_PAD +
          SIDEBAR_W +
          COL_GAP +
          i * (colW + COL_GAP) +
          colW / 2,
      ),
    [modelNames, colW],
  );

  const arrows = useMemo(() => {
    const result: { cx: number; y1: number; y2: number; color: string }[] = [];

    modelNames.forEach((name, colIdx) => {
      const cx = colCenterX[colIdx];

      const appearances = data.groups
        .map((g, i) => ({ hasStep: g.steps.some((s) => s.model === name), i, groupId: g.group_id }))
        .filter((item) => item.hasStep)
        .map((item) => ({ rowIndex: item.i, groupId: item.groupId }));

      for (let k = 0; k + 1 < appearances.length; k++) {
        const from = appearances[k];
        const to = appearances[k + 1];
        const color =
          colorByGroup.get(from.groupId)?.border ??
          FLOWMAP_GROUP_COLORS[from.groupId % FLOWMAP_GROUP_COLORS.length].border;
        result.push({
          cx,
          y1: rowTops[from.rowIndex] + rowHeights[from.rowIndex],
          y2: rowTops[to.rowIndex],
          color,
        });
      }

      if (hasFinalAnswers && appearances.length > 0) {
        const last = appearances[appearances.length - 1];
        const color =
          colorByGroup.get(last.groupId)?.border ??
          FLOWMAP_GROUP_COLORS[last.groupId % FLOWMAP_GROUP_COLORS.length].border;
        result.push({
          cx,
          y1: rowTops[last.rowIndex] + rowHeights[last.rowIndex],
          y2: finalRowTop,
          color,
        });
      }
    });

    return result;
  }, [modelNames, colCenterX, colorByGroup, data.groups, rowTops, rowHeights, hasFinalAnswers, finalRowTop]);

  const toggleGroup = (groupId: number) =>
    setExpandedGroupId((prev) => (prev === groupId ? null : groupId));

  return (
    <div ref={wrapperRef} className="overflow-x-auto rounded-2xl border border-slate-200 bg-white">
      <div style={{ position: "relative", width: totalWidth, height: totalHeight }}>
        {/* SVG layer */}
        <svg
          width={totalWidth}
          height={totalHeight}
          className="absolute inset-0 pointer-events-none"
        >
          <defs>
            <radialGradient id="fmbg" cx="50%" cy="40%" r="80%">
              <stop offset="0%" stopColor="#ffffff" />
              <stop offset="100%" stopColor="#f8fafc" />
            </radialGradient>
          </defs>
          <rect width={totalWidth} height={totalHeight} fill="url(#fmbg)" rx={12} />

          {arrows.map((a, i) => (
            <g key={i}>
              <line x1={a.cx} y1={a.y1} x2={a.cx} y2={a.y2}
                stroke={a.color} strokeWidth={2} strokeOpacity={0.38} />
              <polygon
                points={`${a.cx},${a.y2} ${a.cx - 4},${a.y2 - 8} ${a.cx + 4},${a.y2 - 8}`}
                fill={a.color} fillOpacity={0.5}
              />
            </g>
          ))}
        </svg>

        {/* Header */}
        <div
          style={{
            position: "absolute",
            left: OUTER_PAD + SIDEBAR_W + COL_GAP,
            top: OUTER_PAD,
            height: HEADER_H,
            display: "flex",
            gap: COL_GAP,
          }}
        >
          {modelInfos.map((model) => (
            <div
              key={model.name}
              style={{
                width: colW,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
              }}
            >
              {model.modelId ? (
                <img
                  src={getModelLogo(model.modelId)}
                  alt=""
                  style={{ width: 24, height: 24, borderRadius: 4, objectFit: "contain", flexShrink: 0 }}
                />
              ) : (
                <span
                  style={{
                    width: 10, height: 10, borderRadius: "50%",
                    backgroundColor: model.color, flexShrink: 0,
                  }}
                />
              )}
              <span style={{ fontSize: 12, fontWeight: 700, color: "#0f172a", letterSpacing: "0.04em" }}>
                {model.name}
              </span>
            </div>
          ))}
        </div>

        {/* Group rows */}
        {data.groups.map((group, groupIdx) => {
          const top = rowTops[groupIdx];
          const height = rowHeights[groupIdx];
          const isExpanded = expandedGroupId === group.group_id;
          const modelsInGroup = new Set(group.steps.map((s) => s.model));
          const isShared = modelsInGroup.size >= 2;
          const groupColor =
            colorByGroup.get(group.group_id) ??
            FLOWMAP_GROUP_COLORS[group.group_id % FLOWMAP_GROUP_COLORS.length];
          const rowBg = isExpanded
            ? hexToRgba(groupColor.border, 0.18)
            : groupColor.bg;
          const accentColor = groupColor.border;

          return (
            <div
              key={group.group_id}
              style={{
                position: "absolute",
                left: OUTER_PAD,
                top,
                width: totalWidth - OUTER_PAD * 2,
                height,
                background: rowBg,
                borderRadius: 8,
                borderLeft: `3px solid ${hexToRgba(accentColor, isExpanded ? 0.72 : 0.52)}`,
                transition: "background 0.2s",
              }}
            >
              {/* Group label (sidebar) — click to toggle expand */}
              <div
                onClick={() => toggleGroup(group.group_id)}
                style={{
                  position: "absolute",
                  left: 0, top: 0,
                  width: SIDEBAR_W, height,
                  display: "flex",
                  alignItems: "center",
                  paddingLeft: 12, paddingRight: 8,
                  cursor: "pointer",
                }}
              >
                <div style={{ width: "100%" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div
                      style={{
                        fontSize: 11, fontWeight: 600, lineHeight: 1.35,
                        color: accentColor,
                      }}
                    >
                      G{group.group_id + 1} · {group.group_name}
                    </div>
                    <span style={{ fontSize: 9, color: hexToRgba(accentColor, 0.55), marginLeft: 4, flexShrink: 0 }}>
                      {isExpanded ? "▾" : "▸"}
                    </span>
                  </div>
                  {isShared && !isExpanded && (
                    <div style={{ fontSize: 9, color: hexToRgba(accentColor, 0.7), marginTop: 3 }}>
                      {modelsInGroup.size} models shared
                    </div>
                  )}
                </div>
              </div>

              {/* Model cells */}
              {modelNames.map((modelName, colIdx) => {
                const stepsInCell = group.steps.filter((s) => s.model === modelName);
                const color = accentColor;
                const cellLeft = SIDEBAR_W + COL_GAP + colIdx * (colW + COL_GAP);

                return (
                  <div
                    key={modelName}
                    style={{
                      position: "absolute",
                      left: cellLeft,
                      top: 6,
                      width: colW,
                      height: height - 12,
                      display: "flex",
                      flexDirection: "column",
                      gap: 6,
                    }}
                  >
                    {stepsInCell.length === 0 ? null : stepsInCell.map((step) => (
                      <div
                        key={step.step_idx}
                        onClick={() => toggleGroup(group.group_id)}
                        style={{
                          flex: "1 1 0",
                          minHeight: isExpanded ? EXPANDED_CARD_H : STEP_CARD_H,
                          background: "#ffffff",
                          border: `1px solid ${hexToRgba(color, isExpanded ? 0.62 : 0.42)}`,
                          borderRadius: 8,
                          padding: "6px 10px",
                          boxShadow: "0 1px 2px rgba(15,23,42,0.06), 0 4px 12px rgba(15,23,42,0.06)",
                          cursor: "pointer",
                          display: "flex",
                          flexDirection: "column",
                          overflow: "hidden",
                          transition: "border-color 0.15s, box-shadow 0.15s",
                        }}
                        onMouseEnter={(e) => {
                          (e.currentTarget as HTMLDivElement).style.borderColor = hexToRgba(color, 0.72);
                          (e.currentTarget as HTMLDivElement).style.boxShadow =
                            "0 2px 5px rgba(15,23,42,0.08), 0 8px 20px rgba(15,23,42,0.08)";
                        }}
                        onMouseLeave={(e) => {
                          (e.currentTarget as HTMLDivElement).style.borderColor =
                            hexToRgba(color, isExpanded ? 0.62 : 0.42);
                          (e.currentTarget as HTMLDivElement).style.boxShadow =
                            "0 1px 2px rgba(15,23,42,0.06), 0 4px 12px rgba(15,23,42,0.06)";
                        }}
                      >
                        {/* Step header row */}
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4, flexShrink: 0 }}>
                          <div style={{
                            fontSize: 9, fontWeight: 700, color,
                            textTransform: "uppercase", letterSpacing: "0.08em",
                          }}>
                            step {step.step_idx + 1}
                          </div>
                          <span style={{ fontSize: 8, color: color + "88" }}>
                            {isExpanded ? "▾" : "▸"}
                          </span>
                        </div>

                        {/* Title */}
                        <div style={{
                          fontSize: 11,
                          fontWeight: isExpanded ? 600 : 400,
                          color: "#0f172a",
                          lineHeight: 1.4,
                          flexShrink: 0,
                          ...(isExpanded
                            ? {}
                            : {
                                overflow: "hidden",
                                display: "-webkit-box",
                                WebkitLineClamp: 3,
                                WebkitBoxOrient: "vertical",
                              }),
                        }}>
                          {step.title}
                        </div>

                        {/* Content — only when expanded */}
                        {isExpanded && (
                          <div
                            style={{
                              marginTop: 8,
                              flex: 1,
                              overflowY: "auto",
                              fontSize: 10,
                              color: "#475569",
                              lineHeight: 1.65,
                              paddingRight: 2,
                              borderTop: `1px solid ${hexToRgba(color, 0.22)}`,
                              paddingTop: 8,
                            }}
                          >
                            <MathText text={step.content} />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          );
        })}

        {/* Final Answer row */}
        {hasFinalAnswers && (
          <div
            style={{
              position: "absolute",
              left: OUTER_PAD,
              top: finalRowTop,
              width: totalWidth - OUTER_PAD * 2,
              height: FINAL_ROW_H,
              background: "rgba(245,158,11,0.08)",
              borderRadius: 8,
              borderLeft: "3px solid #f59e0b66",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: 0, top: 0,
                width: SIDEBAR_W, height: FINAL_ROW_H,
                display: "flex", alignItems: "center",
                paddingLeft: 12, paddingRight: 8,
              }}
            >
              <div style={{
                fontSize: 11, fontWeight: 700, color: "#0f172a",
                textTransform: "uppercase", letterSpacing: "0.06em",
              }}>
                Answer
              </div>
            </div>

            {modelNames.map((modelName, colIdx) => {
              const cellLeft = SIDEBAR_W + COL_GAP + colIdx * (colW + COL_GAP);
              const raw = (finalAnswers?.[modelName] ?? "").toString().trim();
              const displayAnswer = raw || "—";
              const answerColor =
                raw && raw !== "—" ? answerValueToColor.get(raw) : undefined;
              const cellBg = answerColor?.bg ?? emptyAnswerStyle.bg;
              const cellBorder = answerColor
                ? hexToRgba(answerColor.border, 0.5)
                : emptyAnswerStyle.border;
              return (
                <div
                  key={modelName}
                  style={{
                    position: "absolute",
                    left: cellLeft, top: 6,
                    width: colW, height: FINAL_ROW_H - 12,
                    background: cellBg,
                    border: `1.5px solid ${cellBorder}`,
                    borderRadius: 8,
                    padding: "8px 12px",
                    boxShadow: "0 1px 2px rgba(15,23,42,0.06), 0 4px 12px rgba(15,23,42,0.06)",
                    display: "flex", flexDirection: "column", justifyContent: "center",
                  }}
                >
                  <div style={{
                    fontSize: 9, fontWeight: 700, color: "#475569",
                    marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.08em",
                  }}>
                    {modelName}
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#0f172a", letterSpacing: "0.02em" }}>
                    {displayAnswer}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 px-4 py-3">
        {groupLegendItems.map((item) => (
          <span
            key={item.groupId}
            className="inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold"
            style={{ borderColor: item.color.border, backgroundColor: item.color.bg, color: item.color.border }}
          >
            G{item.groupId + 1}
          </span>
        ))}
      </div>
    </div>
  );
}
