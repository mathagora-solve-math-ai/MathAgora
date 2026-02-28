export type FlowmapGroupColor = {
  border: string;
  bg: string;
};

/**
 * Yellow/amber spectrum palette for answer value coloring.
 * 100-level backgrounds: clearly tinted but not glaring, borders provide main differentiation.
 */
export const ANSWER_PALETTE: FlowmapGroupColor[] = [
  { border: "#a16207", bg: "#fef9c3" }, // yellow-700 / yellow-100 — lemon
  { border: "#c2410c", bg: "#ffedd5" }, // orange-700 / orange-100 — peach
  { border: "#78350f", bg: "#fffbeb" }, // amber-900 / amber-50  — cream
  { border: "#d97706", bg: "#fef3c7" }, // amber-600 / amber-100 — amber
  { border: "#ea580c", bg: "#fff7ed" }, // orange-600 / orange-50 — pale orange
];

/**
 * Paper/demo-friendly palette for group_id coloring.
 * High-contrast borders + soft backgrounds for readability on white canvases.
 */
export const FLOWMAP_GROUP_COLORS: FlowmapGroupColor[] = [
  { border: "#2563eb", bg: "#dbeafe" }, // blue
  { border: "#0f766e", bg: "#ccfbf1" }, // teal
  { border: "#b45309", bg: "#fef3c7" }, // amber
  { border: "#7c3aed", bg: "#ede9fe" }, // violet
  { border: "#be123c", bg: "#ffe4e6" }, // rose
  { border: "#0369a1", bg: "#e0f2fe" }, // sky
  { border: "#4d7c0f", bg: "#ecfccb" }, // lime
  { border: "#9a3412", bg: "#ffedd5" }, // orange
];
