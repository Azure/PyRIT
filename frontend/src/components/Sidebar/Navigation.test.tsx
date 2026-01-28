/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import Navigation from "./Navigation";

const renderWithProvider = (ui: React.ReactElement) => {
  return render(<FluentProvider theme={webLightTheme}>{ui}</FluentProvider>);
};

describe("Navigation", () => {
  it("renders the chat button (disabled)", () => {
    renderWithProvider(
      <Navigation onToggleTheme={jest.fn()} isDarkMode={false} />
    );

    const chatButton = screen.getByTitle("Chat");
    expect(chatButton).toBeInTheDocument();
    expect(chatButton).toBeDisabled();
  });

  it("renders theme toggle button with light mode title when in dark mode", () => {
    renderWithProvider(
      <Navigation onToggleTheme={jest.fn()} isDarkMode={true} />
    );

    const themeButton = screen.getByTitle("Light Mode");
    expect(themeButton).toBeInTheDocument();
  });

  it("renders theme toggle button with dark mode title when in light mode", () => {
    renderWithProvider(
      <Navigation onToggleTheme={jest.fn()} isDarkMode={false} />
    );

    const themeButton = screen.getByTitle("Dark Mode");
    expect(themeButton).toBeInTheDocument();
  });

  it("calls onToggleTheme when theme button is clicked", () => {
    const mockToggleTheme = jest.fn();
    renderWithProvider(
      <Navigation onToggleTheme={mockToggleTheme} isDarkMode={false} />
    );

    const themeButton = screen.getByTitle("Dark Mode");
    fireEvent.click(themeButton);

    expect(mockToggleTheme).toHaveBeenCalledTimes(1);
  });

  it("theme button is not disabled", () => {
    renderWithProvider(
      <Navigation onToggleTheme={jest.fn()} isDarkMode={false} />
    );

    const themeButton = screen.getByTitle("Dark Mode");
    expect(themeButton).not.toBeDisabled();
  });
});
