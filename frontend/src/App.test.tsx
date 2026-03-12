/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("./components/Layout/MainLayout", () => {
  return function MockMainLayout({
    children,
    onToggleTheme,
    isDarkMode,
  }: {
    children: React.ReactNode;
    onToggleTheme: () => void;
    isDarkMode: boolean;
  }) {
    return (
      <div data-testid="main-layout" data-dark-mode={isDarkMode}>
        <button onClick={onToggleTheme} data-testid="toggle-theme">
          Toggle Theme
        </button>
        {children}
      </div>
    );
  };
});

jest.mock("./components/Builder/PromptBuilderPage", () => {
  return function MockPromptBuilderPage() {
    return <div data-testid="prompt-builder-page">Prompt Builder</div>;
  };
});

describe("App", () => {
  it("renders the layout and prompt builder", () => {
    render(<App />);

    expect(screen.getByTestId("main-layout")).toBeInTheDocument();
    expect(screen.getByTestId("prompt-builder-page")).toBeInTheDocument();
  });

  it("starts in dark mode", () => {
    render(<App />);

    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );
  });

  it("toggles theme when requested", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("toggle-theme"));
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "false"
    );

    fireEvent.click(screen.getByTestId("toggle-theme"));
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );
  });
});
