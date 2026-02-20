/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import App from "./App";

// Mock the child components to isolate App logic
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

jest.mock("./components/Chat/ChatWindow", () => {
  return function MockChatWindow({
    messages,
    onSendMessage,
    onReceiveMessage,
    onNewChat,
  }: {
    messages: Array<{ id: string; content: string }>;
    onSendMessage: (msg: { id: string; content: string }) => void;
    onReceiveMessage: (msg: { id: string; content: string }) => void;
    onNewChat: () => void;
  }) {
    return (
      <div data-testid="chat-window">
        <span data-testid="message-count">{messages.length}</span>
        <button
          onClick={() => onSendMessage({ id: "1", content: "test" })}
          data-testid="send-message"
        >
          Send
        </button>
        <button
          onClick={() => onReceiveMessage({ id: "2", content: "response" })}
          data-testid="receive-message"
        >
          Receive
        </button>
        <button onClick={onNewChat} data-testid="new-chat">
          New Chat
        </button>
      </div>
    );
  };
});

describe("App", () => {
  it("renders with FluentProvider and MainLayout", () => {
    render(<App />);
    expect(screen.getByTestId("main-layout")).toBeInTheDocument();
    expect(screen.getByTestId("chat-window")).toBeInTheDocument();
  });

  it("starts in dark mode", () => {
    render(<App />);
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );
  });

  it("toggles theme when onToggleTheme is called", () => {
    render(<App />);

    // Initially dark mode
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );

    // Toggle to light mode
    fireEvent.click(screen.getByTestId("toggle-theme"));
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "false"
    );

    // Toggle back to dark mode
    fireEvent.click(screen.getByTestId("toggle-theme"));
    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );
  });

  it("starts with empty messages", () => {
    render(<App />);
    expect(screen.getByTestId("message-count")).toHaveTextContent("0");
  });

  it("adds messages when handleSendMessage is called", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("send-message"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("1");
  });

  it("adds messages when handleReceiveMessage is called", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("receive-message"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("1");
  });

  it("clears messages when handleNewChat is called", () => {
    render(<App />);

    // Add some messages first
    fireEvent.click(screen.getByTestId("send-message"));
    fireEvent.click(screen.getByTestId("receive-message"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("2");

    // Clear messages
    fireEvent.click(screen.getByTestId("new-chat"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("0");
  });
});
