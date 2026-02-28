declare module "react-katex" {
  import * as React from "react";

  export interface KaTeXProps {
    math?: string;
    children?: React.ReactNode;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
  }

  export const InlineMath: React.ComponentType<KaTeXProps>;
  export const BlockMath: React.ComponentType<KaTeXProps>;
}
