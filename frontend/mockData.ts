export type UploadedImage = {
    dataUrl: string;
    name: string;
    updatedAt: number;
    isDemo?: boolean;
    source?: "dataset" | "upload" | "demo";
    datasetMeta?: {
      year: string;
      set: string;
      page: string;
    };
  };
  
  export type Detection = {
    id: string;
    x: number;
    y: number;
    w: number;
    h: number;
    label: string;
    text?: string;
    cropUrl?: string;
  };
  
  /** Result of detect API: document type (from classifier) and list of cropped problems with OCR. */
  export type DetectResult = {
    documentType: "csat" | "sat" | null;
    detections: Detection[];
  };
  
  export type Step = {
    id: string;
    title: string;
    body: string;
  };
  
  export type ModelResult = {
    modelId: string;
    modelName: string;
    version: string;
    latencyMs: number;
    temperature: number;
    strategy: string;
    steps: Step[];
    finalAnswer: string;
  };
  
  export type FlowCommonStageTemplate = {
    id: string;
    title: string;
    keywords: string[];
  };
  
  export const FLOW_COMMON_STAGE_TEMPLATES: Record<string, FlowCommonStageTemplate[]> = {
    "30": [
      {
        id: "cond",
        title: "극값 조건 정리",
        keywords: ["h'(x)", "h(x)", "접선", "f(x)-g(x)", "극값", "조건"],
      },
      {
        id: "derivative-analysis",
        title: "f'(x)=|sin x|cos x 분석",
        keywords: ["f'(x)", "영점", "극점", "그래프", "cos", "sin"],
      },
      {
        id: "case-split",
        title: "sin(a)=0 / sin(a)≠0 분기",
        keywords: ["sin(a)=0", "sin(a)≠0", "분류", "경우", "cos2a", "f''(a)=0"],
      },
      {
        id: "sequence",
        title: "a_n 나열 및 a2,a6 추출",
        keywords: ["a_n", "나열", "a2", "a6", "수열", "오름차순"],
      },
      {
        id: "final",
        title: "최종 계산",
        keywords: ["최종", "계산", "100", "pi", "\\pi", "답"],
      },
    ],
  };
  
  const demoSvg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
    <defs>
      <linearGradient id="paper" x1="0" x2="1">
        <stop offset="0" stop-color="#fdfcf9" />
        <stop offset="1" stop-color="#f4efe6" />
      </linearGradient>
    </defs>
    <rect width="1200" height="900" rx="32" fill="url(#paper)" />
    <rect x="60" y="80" width="1080" height="740" rx="24" fill="#fffaf2" stroke="#eadfcd" stroke-width="6" />
    <g fill="none" stroke="#d7c9b2" stroke-width="3" opacity="0.6">
      ${Array.from({ length: 18 })
        .map((_, i) => `<line x1="110" y1="${150 + i * 36}" x2="1090" y2="${150 + i * 36}" />`)
        .join("\n")}
    </g>
    <g stroke="#2b2a24" stroke-width="5" stroke-linecap="round" font-family="Space Grotesk, sans-serif" font-size="34" fill="#2b2a24">
      <text x="140" y="190">1) Simplify: 3x + 5 = 2x + 17</text>
      <text x="140" y="330">2) Factor: x^2 - 9 = 0</text>
      <text x="140" y="470">3) Area of trapezoid, bases 8 and 14, height 6</text>
      <text x="140" y="610">4) Convert: 0.125 to fraction</text>
    </g>
    <g stroke="#c04b4b" stroke-width="6">
      <line x1="80" y1="120" x2="1120" y2="120" />
    </g>
    <g fill="#c04b4b" font-family="Instrument Serif, serif" font-size="42">
      <text x="120" y="120">Workbook Practice</text>
    </g>
  </svg>
  `;
  
  export const MOCK_IMAGE_DATA_URL = `data:image/svg+xml;utf8,${encodeURIComponent(
    demoSvg.trim(),
  )}`;
  
  export const MOCK_DETECTIONS: Detection[] = [
    { id: "p1", x: 120, y: 135, w: 936, h: 108, label: "Linear equation" },
    { id: "p2", x: 120, y: 288, w: 840, h: 108, label: "Quadratic factor" },
    { id: "p3", x: 120, y: 450, w: 1020, h: 108, label: "Trapezoid area" },
    { id: "p4", x: 120, y: 603, w: 720, h: 108, label: "Decimal to fraction" },
  ];
  
  export const MOCK_LLM_RESULTS: Record<string, ModelResult[]> = {
    "8": [
      {
        modelId: "gpt-4o",
        modelName: "GPT-4o",
        version: "v2024.11",
        latencyMs: 980,
        temperature: 0.2,
        strategy:
          "주어진 항등식을 정리해 f(x)를 명시적으로 구한 뒤, 대칭성으로 적분을 빠르게 계산한다.",
        steps: [
          {
            id: "gpt8_1",
            title: "항등식 정리",
            body: "조건 \\(x f(x) - f(x) = 3x^4 - 3x\\)을 \\((x-1)f(x)=3x(x^3-1)\\)로 정리한다.",
          },
          {
            id: "gpt8_2",
            title: "인수분해로 f(x) 도출",
            body: "\\(3x(x^3-1)=3x(x-1)(x^2+x+1)\\)이므로 \\(x \\ne 1\\)에서 \\(f(x)=3x(x^2+x+1)\\)이다.",
          },
          {
            id: "gpt8_3",
            title: "다항식 일치 확인",
            body: "f가 삼차함수(다항식)이므로 \\(x=1\\)에서도 같은 식이어야 해 \\(f(x)=3x^3+3x^2+3x\\)로 확정된다.",
          },
          {
            id: "gpt8_4",
            title: "적분에서 홀짝 분리",
            body: "\\(\\int_{-2}^{2}(3x^3+3x^2+3x)dx\\)에서 홀수항 \\(3x^3,3x\\)는 대칭 구간 적분이 0이다.",
          },
          {
            id: "gpt8_5",
            title: "남은 항 계산",
            body: "\\(\\int_{-2}^{2}3x^2dx=3\\cdot\\frac{16}{3}=16\\)이므로 적분값은 16이다.",
          },
        ],
        finalAnswer: "16",
      },
      {
        modelId: "claude-3.5-sonnet",
        modelName: "Claude Sonnet 3.5",
        version: "2024-10",
        latencyMs: 1120,
        temperature: 0.3,
        strategy:
          "식의 인수분해로 f(x)를 결정하고, 대칭구간 적분의 성질을 활용해 계산을 단순화한다.",
        steps: [
          {
            id: "cl8_1",
            title: "양변 재정렬",
            body: "\\(x f(x)-f(x)=(x-1)f(x)\\)로 묶어 좌변을 단순화한다.",
          },
          {
            id: "cl8_2",
            title: "우변 인수분해",
            body: "\\(3x^4-3x=3x(x^3-1)=3x(x-1)(x^2+x+1)\\)로 전개한다.",
          },
          {
            id: "cl8_3",
            title: "함수 결정",
            body: "양변의 \\((x-1)\\)을 약분하여 \\(f(x)=3x(x^2+x+1)=3x^3+3x^2+3x\\)로 확정한다. 다항식이므로 \\(x=1\\)에서도 성립.",
          },
          {
            id: "cl8_4",
            title: "대칭성 사용",
            body: "대칭구간에서 홀수함수 적분은 0이므로 \\(3x^3,3x\\) 항은 사라진다.",
          },
          {
            id: "cl8_5",
            title: "최종 적분",
            body: "\\(\\int_{-2}^{2}3x^2dx=3\\cdot(16/3)=16\\).",
          },
        ],
        finalAnswer: "16",
      },
      {
        modelId: "gemini-2.5",
        modelName: "Gemini 2.5",
        version: "v2.5-pro",
        latencyMs: 900,
        temperature: 0.25,
        strategy:
          "주어진 항등식에서 f(x)를 직접 구하고, 적분은 홀짝 분해로 빠르게 처리한다.",
        steps: [
          {
            id: "gm8_1",
            title: "좌변 묶기",
            body: "\\(x f(x)-f(x)=(x-1)f(x)\\)로 정리한다.",
          },
          {
            id: "gm8_2",
            title: "우변 분해",
            body: "\\(3x^4-3x=3x(x^3-1)=3x(x-1)(x^2+x+1)\\).",
          },
          {
            id: "gm8_3",
            title: "f(x) 확정",
            body: "\\(x\\ne1\\)에서 \\(f(x)=3x(x^2+x+1)=3x^3+3x^2+3x\\). 다항식이므로 전체 구간에서 동일.",
          },
          {
            id: "gm8_4",
            title: "홀수항 제거",
            body: "\\(\\int_{-2}^{2}3x^3dx=\\int_{-2}^{2}3xdx=0\\)이므로 짝수항만 계산한다.",
          },
          {
            id: "gm8_5",
            title: "적분값 계산",
            body: "\\(\\int_{-2}^{2}3x^2dx=16\\).",
          },
        ],
        finalAnswer: "16",
      },
    ],
    "30": [
      {
        modelId: "gpt-4o",
        modelName: "GPT-4o",
        version: "v2024.11",
        latencyMs: 980,
        temperature: 0.2,
        strategy:
          "도함수의 영점을 찾아 극값 후보를 정렬하고, 필요한 항의 차이를 계산해 식에 대입한다.",
        steps: [
          {
            id: "gpt30_1",
            title: "조건 정리",
            body: "조건은 \\(f'(x)=|\\sin x|\\cos x\\), 점 \\(a\\)에서의 접선 \\(y=g(x)\\), 그리고 \\(h(x)=\\int_0^x (f(t)-g(t))dt\\)이다. \\(a\\)가 극대 또는 극소가 되는 지점을 오름차순으로 \\(a_n\\)이라 두고 \\(\\frac{100}{\\pi}(a_6-a_2)\\)를 구한다.",
          },
          {
            id: "gpt30_2",
            title: "도함수 영점 찾기",
            body: "\\(f'(x)=|\\sin x|\\cos x=0\\) 이려면 \\(|\\sin x|=0\\) 또는 \\(\\cos x=0\\)이어야 한다. 따라서 후보는 \\(x=n\\pi\\) 또는 \\(x=\\frac{\\pi}{2}+n\\pi\\) (\\(n\\in\\mathbb{Z}\\))이다.",
          },
          {
            id: "gpt30_3",
            title: "수열에서 필요한 항 선택",
            body: "양의 방향에서 정렬하면 \\(a_n\\)은 \\(0,\\frac{\\pi}{2},\\pi,\\frac{3\\pi}{2},2\\pi,\\frac{5\\pi}{2},\\dots\\) 이므로 \\(a_2=\\frac{\\pi}{2}\\), \\(a_6=\\frac{5\\pi}{2}\\)이다.",
          },
          {
            id: "gpt30_4",
            title: "최종 계산",
            body: "\\(a_6-a_2=\\frac{5\\pi}{2}-\\frac{\\pi}{2}=2\\pi\\). 따라서 \\(\\frac{100}{\\pi}(a_6-a_2)=\\frac{100}{\\pi}\\cdot 2\\pi=200\\)이다.",
          },
        ],
        finalAnswer: "200",
      },
      {
        modelId: "gemini-3",
        modelName: "Gemini 3",
        version: "v3",
        latencyMs: 940,
        temperature: 0.25,
        strategy:
          "h(x)의 극값 조건을 분석한 뒤, f'(x)=|sin x|cos x의 극점을 이용해 a_n을 나열하고 최종값을 계산한다.",
        steps: [
          {
            id: "gm30_1",
            title: "h(x)의 극대/극소 조건 분석",
            body: "\\(h'(x)=f(x)-g(x)\\)이고 \\(g(x)\\)는 \\(x=a\\)에서의 접선이므로 \\(h'(a)=0\\)이다. 제시 풀이에서는 \\(h\\)가 \\(x=a\\)에서 극값이 되려면 \\(f(x)-g(x)\\)의 부호 변화가 필요하고, 이를 위해 \\(a\\)가 곡선 \\(y=f(x)\\)의 변곡점에 해당한다고 본다.",
          },
          {
            id: "gm30_2",
            title: "도함수 f'(x)의 그래프 및 극점 분석",
            body: "\\(f'(x)=|\\sin x|\\cos x\\)를 구간별로 나누어 보고, \\(f'(x)\\)가 로컬 극대/극소를 가지는 점을 찾는다. 또한 경계점 \\(x=n\\pi\\) 부근의 좌우 거동을 비교해 극값 여부를 판단한다.",
          },
          {
            id: "gm30_3",
            title: "수열 a_n의 값 구하기",
            body: "제시 풀이에서는 양의 \\(a\\)들을 작은 순서대로 나열해 \\(a_n\\)을 구성하고, 그중 \\(a_2\\), \\(a_6\\)를 읽어낸다.",
          },
          {
            id: "gm30_4",
            title: "최종 결과 계산",
            body: "문제의 식 \\(\\frac{100}{\\pi}(a_6-a_2)\\)에 위 값을 대입해 계산하고, 최종값을 125로 정리한다.",
          },
        ],
        finalAnswer: "125",
      },
      {
        modelId: "gpt-5.2",
        modelName: "GPT-5.2",
        version: "gpt-5.2",
        latencyMs: 142000,
        temperature: 0.2,
        strategy:
          "원함수를 복원한 뒤 h'(x)=f(x)-g(x)의 부호변화 조건으로 극값 가능 a를 분류해 a_n을 계산한다.",
        steps: [
          {
            id: "gpt52_1",
            title: "도함수로부터 원함수 형태 정리",
            body: "주어진 도함수는 \\(f'(x)=|\\sin x|\\cos x\\)이다. \\(\\sin x|\\sin x|\\)를 미분하면 \\(2|\\sin x|\\cos x\\)가 되므로 \\(f'(x)=\\frac12\\frac{d}{dx}(\\sin x|\\sin x|)\\). 따라서 어떤 상수 \\(C\\)에 대해 \\(f(x)=\\frac12\\sin x|\\sin x|+C\\).",
          },
          {
            id: "gpt52_2",
            title: "h의 극값 조건을 h'(x)로 바꾸기",
            body: "점 \\((a,f(a))\\)에서의 접선 \\(y=g(x)\\)는 \\(g(x)=f(a)+f'(a)(x-a)\\). 또 \\(h(x)=\\int_0^x (f(t)-g(t))dt\\)이므로 \\(h'(x)=f(x)-g(x)\\). 특히 \\(g(a)=f(a)\\)라서 \\(h'(a)=0\\)은 항상 성립하고, \\(x=a\\)에서 극대/극소가 되려면 \\(h'(x)=f(x)-g(x)\\)의 부호가 \\(a\\)를 기준으로 바뀌어야 한다.",
          },
          {
            id: "gpt52_3",
            title: "sin(a)≠0 인 경우 분류",
            body: "\\(\\sin a\\neq0\\)이면 근방에서 \\(|\\sin x|=\\pm\\sin x\\)로 매끄럽게 표현된다. 이때 \\(f''(a)=\\operatorname{sgn}(\\sin a)\\cos2a\\). 접선과의 차이가 부호를 바꾸려면 \\(f''(a)=0\\Rightarrow\\cos2a=0\\), 즉 \\(a=\\frac{(2k+1)\\pi}{4}\\). 또한 이 경우 \\(f(x)-g(x)\\sim(x-a)^3\\) 꼴이 되어 부호가 바뀐다.",
          },
          {
            id: "gpt52_4",
            title: "sin(a)=0 인 경우 분류",
            body: "\\(a=k\\pi\\,(k\\in\\mathbb{Z}_{>0})\\)이면 \\(f'(a)=0\\)이고 접선은 \\(g(x)=f(a)\\). 따라서 \\(h'(x)=f(x)-f(a)=\\frac12\\sin x|\\sin x|\\). \\(x=a\\) 근처에서 \\(\\sin x\\approx(-1)^k(x-a)\\)이므로 \\(\\sin x|\\sin x|\\approx(-1)^k(x-a)|x-a|\\)가 되어 부호가 바뀐다. 따라서 \\(a=k\\pi\\)도 모두 조건을 만족한다.",
          },
          {
            id: "gpt52_5",
            title: "a2, a6 계산",
            body: "양수 해의 합집합은 \\(a=\\frac{(2k+1)\\pi}{4}\\,(k=0,1,2,\\dots)\\)와 \\(a=k\\pi\\,(k=1,2,3,\\dots)\\). 오름차순으로 \\(a_1=\\frac\\pi4, a_2=\\frac{3\\pi}4, a_3=\\pi, a_4=\\frac{5\\pi}4, a_5=\\frac{7\\pi}4, a_6=2\\pi\\). 그래서 \\(a_6-a_2=2\\pi-\\frac{3\\pi}4=\\frac{5\\pi}4\\), \\(\\frac{100}{\\pi}(a_6-a_2)=125\\).",
          },
        ],
        finalAnswer: "125",
      },
      {
        modelId: "sonnet-4.5",
        modelName: "Sonnet 4.5",
        version: "4.5",
        latencyMs: 1160,
        temperature: 0.2,
        strategy:
          "h'(x)=f(x)-g(x)의 부호변화 관점으로 조건을 정리하고, f''(a)=0 후보를 나열해 필요한 항을 계산한다.",
        steps: [
          {
            id: "sn45_1",
            title: "주어진 조건 정리",
            body: "\\(f'(x)=|\\sin x|\\cos x\\), 점 \\((a,f(a))\\)에서의 접선 \\(y=g(x)\\), \\(h(x)=\\int_0^x (f(t)-g(t))dt\\)가 주어졌고, \\(x=a\\)에서 \\(h\\)가 극값을 갖는 양수 \\(a\\)를 \\(a_n\\)이라 할 때 \\(\\frac{100}{\\pi}(a_6-a_2)\\)를 구한다.",
          },
          {
            id: "sn45_2",
            title: "극값 조건 분석",
            body: "접선은 \\(g(x)=f(a)+f'(a)(x-a)\\)이고 \\(g(a)=f(a)\\)이므로 \\(h'(a)=f(a)-g(a)=0\\)은 자동 성립한다. 따라서 핵심은 \\(h'(x)=f(x)-g(x)\\)가 \\(x=a\\) 근처에서 부호를 바꾸는지 확인하는 것이다.",
          },
          {
            id: "sn45_3",
            title: "2차 도함수 관찰",
            body: "\\(h''(a)=f'(a)-g'(a)=f'(a)-f'(a)=0\\)이라 2차만으로는 판정이 안 된다. 제시 풀이에서는 \\(h'(x)=f(x)-f(a)-f'(a)(x-a)\\)를 전개해, 부호변화를 위해 \\(f''(a)=0\\) 후보를 먼저 찾고 고차항으로 극값 여부를 본다.",
          },
          {
            id: "sn45_4",
            title: "f''(a)=0 조건 찾기",
            body: "\\(\\sin x>0\\)이면 \\(f'(x)=\\sin x\\cos x=\\frac12\\sin2x\\Rightarrow f''(x)=\\cos2x\\), \\(\\sin x<0\\)이면 \\(f'(x)=-\\sin x\\cos x=-\\frac12\\sin2x\\Rightarrow f''(x)=-\\cos2x\\). 두 경우 모두 \\(f''(a)=0\\Rightarrow \\cos2a=0\\), 즉 \\(a=\\frac\\pi4+\\frac{n\\pi}2\\).",
          },
          {
            id: "sn45_5",
            title: "a_n 나열",
            body: "양수 해를 작은 순서로 나열하면 \\(a_1=\\frac\\pi4, a_2=\\frac{3\\pi}4, a_3=\\frac{5\\pi}4, a_4=\\frac{7\\pi}4, a_5=\\frac{9\\pi}4, a_6=\\frac{11\\pi}4\\).",
          },
          {
            id: "sn45_6",
            title: "최종 계산",
            body: "\\(a_6-a_2=\\frac{11\\pi}4-\\frac{3\\pi}4=2\\pi\\). 따라서 \\(\\frac{100}{\\pi}(a_6-a_2)=200\\).",
          },
        ],
        finalAnswer: "200",
      },
      {
        modelId: "haiku-4.5",
        modelName: "Haiku 4.5",
        version: "4.5",
        latencyMs: 980,
        temperature: 0.25,
        strategy:
          "h'(x)=f(x)-g(x) 조건을 통해 극값 후보를 찾고, f''(x)=0이 되는 점들을 나열해 식에 대입한다.",
        steps: [
          {
            id: "hk45_1",
            title: "h'(x) 구하기",
            body: "미적분학 기본정리에 의해 \\(h'(x)=f(x)-g(x)\\). \\(x=a\\)에서 극값을 가지려면 \\(h'(a)=0\\)이어야 하고, 접선 성질로 \\(f(a)=g(a)\\)는 자동으로 성립한다.",
          },
          {
            id: "hk45_2",
            title: "h''(x) 및 극값 조건",
            body: "\\(h''(x)=f'(x)-g'(x)\\), 그리고 \\(x=a\\)에서 \\(g'(a)=f'(a)\\)라서 \\(h''(a)=0\\). 제시 풀이에서는 \\(h'(x)=f(x)-g(x)\\)가 \\(x=a\\)를 지나며 부호가 바뀌려면 \\(f''(a)=0\\)이고 부호 변화가 있어야 한다고 본다.",
          },
          {
            id: "hk45_3",
            title: "f''(x) 구하기",
            body: "\\(f'(x)=|\\sin x|\\cos x\\). \\(\\sin x>0\\)이면 \\(f'(x)=\\frac12\\sin2x\\Rightarrow f''(x)=\\cos2x\\), \\(\\sin x<0\\)이면 \\(f'(x)=-\\frac12\\sin2x\\Rightarrow f''(x)=-\\cos2x\\).",
          },
          {
            id: "hk45_4",
            title: "f''(x)=0인 점 찾기",
            body: "\\(f''(a)=0\\Rightarrow \\cos2a=0\\), 따라서 \\(a=\\frac\\pi4,\\frac{3\\pi}4,\\frac{5\\pi}4,\\frac{7\\pi}4,\\dots\\). 제시된 표기대로 \\(a_0=\\frac\\pi4, a_1=\\frac{3\\pi}4, a_2=\\frac{5\\pi}4\\).",
          },
          {
            id: "hk45_5",
            title: "a_0 - a_2 계산",
            body: "\\(a_0-a_2=\\frac\\pi4-\\frac{5\\pi}4=-\\pi\\).",
          },
          {
            id: "hk45_6",
            title: "최종 계산",
            body: "\\(\\frac{100}{\\pi}(a_0-a_2)=-100\\)이지만, 제시 답변에서는 절댓값 해석을 반영해 최종값을 100으로 정리한다.",
          },
        ],
        finalAnswer: "100",
      },
    ],
    p1: [
      {
        modelId: "alpha",
        modelName: "Aster-XL",
        version: "v3.2",
        latencyMs: 840,
        temperature: 0.2,
        strategy:
          "이항을 한쪽으로 모아 미지수 항을 분리한 뒤, 정리해서 값을 구한다.",
        steps: [
          {
            id: "a1",
            title: "식 확인 및 정리",
            body: "주어진 방정식 3x + 5 = 2x + 17을 확인하고 정리할 준비를 한다.",
          },
          {
            id: "a2",
            title: "미지수 항 이동",
            body: "양변에서 2x를 빼서 x 항을 한쪽으로 모은다.",
          },
          {
            id: "a3",
            title: "상수항 이동",
            body: "양변에서 5를 빼면 x = 12가 된다.",
          },
          {
            id: "a4",
            title: "해 검산",
            body: "x=12를 대입해 양변이 같음을 확인한다.",
          },
        ],
        finalAnswer: "x = 12",
      },
      {
        modelId: "beta",
        modelName: "Gauss-Reason",
        version: "v2.6",
        latencyMs: 910,
        temperature: 0.35,
        strategy:
          "항을 재배열해 x를 고립시키고, 간단히 대입 검산으로 확인한다.",
        steps: [
          {
            id: "b1",
            title: "항 재배열",
            body: "3x + 5 = 2x + 17에서 x 항과 상수항을 정리할 준비를 한다.",
          },
          {
            id: "b2",
            title: "x 항 비교",
            body: "3x - 2x = 17 - 5로 변형해 x를 고립시키는 형태로 만든다.",
          },
          {
            id: "b3",
            title: "간단화",
            body: "좌우를 계산하면 x = 12가 된다.",
          },
          {
            id: "b4",
            title: "대입 검산",
            body: "3(12)+5와 2(12)+17이 동일함을 확인한다.",
          },
        ],
        finalAnswer: "x = 12",
      },
      {
        modelId: "gamma",
        modelName: "Orchid-Math",
        version: "v1.9",
        latencyMs: 760,
        temperature: 0.15,
        strategy:
          "식 양변을 단순하게 정리해 x만 남기는 직관적 풀이를 사용한다.",
        steps: [
          {
            id: "c1",
            title: "식 단순화",
            body: "3x + 5 = 2x + 17에서 2x를 빼서 x + 5 = 17로 만든다.",
          },
          {
            id: "c2",
            title: "상수항 제거",
            body: "양변에서 5를 빼면 x = 12가 된다.",
          },
        ],
        finalAnswer: "x = 12",
      },
    ],
    p2: [
      {
        modelId: "alpha",
        modelName: "Aster-XL",
        version: "v3.2",
        latencyMs: 980,
        temperature: 0.25,
        strategy:
          "차이의 제곱 공식으로 인수분해한 뒤, 각각의 근을 구한다.",
        steps: [
          {
            id: "a1",
            title: "차이의 제곱 인식",
            body: "x^2 - 9는 a^2 - b^2 형태이므로 차이의 제곱으로 본다.",
          },
          {
            id: "a2",
            title: "인수분해",
            body: "(x - 3)(x + 3) = 0으로 분해한다.",
          },
          {
            id: "a3",
            title: "각 인수 해석",
            body: "각 인수를 0으로 두어 x = 3, x = -3을 얻는다.",
          },
        ],
        finalAnswer: "x = 3, -3",
      },
      {
        modelId: "beta",
        modelName: "Gauss-Reason",
        version: "v2.6",
        latencyMs: 1030,
        temperature: 0.4,
        strategy:
          "인수분해 공식을 적용해 해의 후보를 빠르게 도출한다.",
        steps: [
          {
            id: "b1",
            title: "공식 적용",
            body: "a^2 - b^2 = (a-b)(a+b)를 이용해 식을 인수분해한다.",
          },
          {
            id: "b2",
            title: "근 설정",
            body: "x - 3 = 0 또는 x + 3 = 0을 풀어 해를 찾는다.",
          },
          {
            id: "b3",
            title: "해 정리",
            body: "따라서 x = 3, x = -3이다.",
          },
        ],
        finalAnswer: "x = 3, -3",
      },
      {
        modelId: "gamma",
        modelName: "Orchid-Math",
        version: "v1.9",
        latencyMs: 820,
        temperature: 0.2,
        strategy:
          "기본 인수분해 패턴을 이용해 해를 즉시 구한다.",
        steps: [
          {
            id: "c1",
            title: "패턴 인식",
            body: "x^2 - 9는 차이의 제곱 형태다.",
          },
          {
            id: "c2",
            title: "인수분해 및 해",
            body: "(x-3)(x+3)=0이므로 x = 3, -3이다.",
          },
        ],
        finalAnswer: "x = 3, -3",
      },
    ],
    p3: [
      {
        modelId: "alpha",
        modelName: "Aster-XL",
        version: "v3.2",
        latencyMs: 940,
        temperature: 0.25,
        strategy:
          "사다리꼴 넓이 공식에 값을 대입하고 단계적으로 계산한다.",
        steps: [
          {
            id: "a1",
            title: "공식 확인",
            body: "사다리꼴 넓이 A = (b1+b2)/2 × h 공식을 사용한다.",
          },
          {
            id: "a2",
            title: "값 대입",
            body: "밑변 8, 14와 높이 6을 대입한다.",
          },
          {
            id: "a3",
            title: "계산",
            body: "(8+14)/2 = 11, 따라서 넓이 = 11×6 = 66.",
          },
        ],
        finalAnswer: "66 square units",
      },
      {
        modelId: "beta",
        modelName: "Gauss-Reason",
        version: "v2.6",
        latencyMs: 1100,
        temperature: 0.35,
        strategy:
          "평균 밑변을 먼저 구한 뒤 높이를 곱해 넓이를 계산한다.",
        steps: [
          {
            id: "b1",
            title: "평균 밑변 계산",
            body: "(8+14)/2 = 11로 평균 밑변을 구한다.",
          },
          {
            id: "b2",
            title: "높이 곱하기",
            body: "11×6 = 66으로 넓이를 계산한다.",
          },
        ],
        finalAnswer: "66 square units",
      },
      {
        modelId: "gamma",
        modelName: "Orchid-Math",
        version: "v1.9",
        latencyMs: 880,
        temperature: 0.2,
        strategy:
          "공식을 간단화해 한 번에 계산한다.",
        steps: [
          {
            id: "c1",
            title: "공식 변형",
            body: "A = (b1+b2)h/2로 계산을 정리한다.",
          },
          {
            id: "c2",
            title: "즉시 계산",
            body: "22×6/2 = 66으로 넓이를 얻는다.",
          },
        ],
        finalAnswer: "66 square units",
      },
    ],
    p4: [
      {
        modelId: "alpha",
        modelName: "Aster-XL",
        version: "v3.2",
        latencyMs: 780,
        temperature: 0.2,
        strategy:
          "소수를 분수로 변환한 뒤 약분한다.",
        steps: [
          {
            id: "a1",
            title: "분수로 변환",
            body: "0.125를 125/1000으로 바꾼다.",
          },
          {
            id: "a2",
            title: "약분",
            body: "분자·분모를 125로 나눠 1/8을 얻는다.",
          },
        ],
        finalAnswer: "1/8",
      },
      {
        modelId: "beta",
        modelName: "Gauss-Reason",
        version: "v2.6",
        latencyMs: 880,
        temperature: 0.3,
        strategy:
          "자리수를 이동해 분수로 만들고 최대한 단순화한다.",
        steps: [
          {
            id: "b1",
            title: "자리수 이동",
            body: "소수점을 3칸 이동해 125/1000으로 표현한다.",
          },
          {
            id: "b2",
            title: "기약분수",
            body: "125로 약분해 1/8을 얻는다.",
          },
        ],
        finalAnswer: "1/8",
      },
      {
        modelId: "gamma",
        modelName: "Orchid-Math",
        version: "v1.9",
        latencyMs: 720,
        temperature: 0.15,
        strategy:
          "소수를 분수로 바꾼 뒤 가장 간단한 형태로 정리한다.",
        steps: [
          {
            id: "c1",
            title: "분수 형태",
            body: "0.125 = 125/1000으로 바꾼다.",
          },
          {
            id: "c2",
            title: "약분",
            body: "125로 나누어 1/8로 만든다.",
          },
        ],
        finalAnswer: "1/8",
      },
    ],
  };
  
  export const getMockDetections = () => MOCK_DETECTIONS;
  
  const buildGenericResults = (problemId: string): ModelResult[] => [
    {
      modelId: "alpha",
      modelName: "Aster-XL",
      version: "v3.2",
      latencyMs: 820,
      temperature: 0.2,
      strategy: "핵심 조건을 정리한 뒤 수식을 구성해 계산한다.",
      steps: [
        {
          id: `${problemId}_a1`,
          title: "조건 정리",
          body: "문제의 조건과 목표를 정리해 풀이 방향을 세운다.",
        },
        {
          id: `${problemId}_a2`,
          title: "식 구성",
          body: "조건을 만족하는 식이나 관계식을 구성한다.",
        },
        {
          id: `${problemId}_a3`,
          title: "계산 및 정리",
          body: "식에 따라 계산을 진행하고 결과를 정리한다.",
        },
      ],
      finalAnswer: "Computed solution (mock).",
    },
    {
      modelId: "beta",
      modelName: "Gauss-Reason",
      version: "v2.6",
      latencyMs: 940,
      temperature: 0.35,
      strategy: "문장 해석 후 식을 세우고 검산으로 확인한다.",
      steps: [
        {
          id: `${problemId}_b1`,
          title: "조건 해석",
          body: "문장의 핵심 조건을 해석해 풀이에 필요한 정보를 얻는다.",
        },
        {
          id: `${problemId}_b2`,
          title: "식 설정",
          body: "조건을 만족하는 식을 설정하고 미지수를 정한다.",
        },
        {
          id: `${problemId}_b3`,
          title: "계산 및 검산",
          body: "식을 계산해 답을 얻고 조건을 만족하는지 확인한다.",
        },
      ],
      finalAnswer: "Computed solution (mock).",
    },
    {
      modelId: "gamma",
      modelName: "Orchid-Math",
      version: "v1.9",
      latencyMs: 760,
      temperature: 0.15,
      strategy: "수학적 형태로 변환해 간단화한 뒤 답을 도출한다.",
      steps: [
        {
          id: `${problemId}_c1`,
          title: "조건 확인",
          body: "핵심 조건을 확인해 필요한 수식 전개를 준비한다.",
        },
        {
          id: `${problemId}_c2`,
          title: "식 세우기",
          body: "조건을 반영해 식을 세우고 간단화한다.",
        },
        {
          id: `${problemId}_c3`,
          title: "계산 결과",
          body: "정리된 식을 계산해 최종 값을 결정한다.",
        },
      ],
      finalAnswer: "Computed solution (mock).",
    },
  ];
  
  export const getMockResults = (problemId: string): ModelResult[] =>
    MOCK_LLM_RESULTS[problemId] || buildGenericResults(problemId);
  