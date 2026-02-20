/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import App from "./App";

// Mock the child components to isolate App logic
jest.mock("./components/Layout/MainLayout", () => {
  const MockMainLayout = ({
    children,
    onToggleTheme,
    isDarkMode,
    currentView,
    onNavigate,
  }: {
    children: React.ReactNode;
    onToggleTheme: () => void;
    isDarkMode: boolean;
    currentView: string;
    onNavigate: (view: string) => void;
  }) => {
    return (
      <div data-testid="main-layout" data-dark-mode={isDarkMode} data-current-view={currentView}>
        <button onClick={onToggleTheme} data-testid="toggle-theme">
          Toggle Theme
        </button>
        <button onClick={() => onNavigate("config")} data-testid="nav-config">
          Config
        </button>
        <button onClick={() => onNavigate("chat")} data-testid="nav-chat">
          Chat
        </button>
        {children}
      </div>
    );
  };
  MockMainLayout.displayName = "MockMainLayout";
  return {
    __esModule: true,
    default: MockMainLayout,
  };
});

jest.mock("./components/Chat/ChatWindow", () => {
  const MockChatWindow = ({
    messages,
    onSendMessage,
    onReceiveMessage,
    onNewChat,
    activeTarget,
    conversationId,
    onConversationCreated,
  }: {
    messages: Array<{ id: string; content: string }>;
    onSendMessage: (msg: { id: string; content: string }) => void;
    onReceiveMessage: (msg: { id: string; content: string }) => void;
    onNewChat: () => void;
    activeTarget: unknown;
    conversationId: string | null;
    onConversationCreated: (id: string) => void;
  }) => {
    return (
      <div data-testid="chat-window">
        <span data-testid="message-count">{messages.length}</span>
        <span data-testid="conversation-id">{conversationId ?? "none"}</span>
        <span data-testid="has-target">{activeTarget ? "yes" : "no"}</span>
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
        <button
          onClick={() => onConversationCreated("conv-123")}
          data-testid="set-conversation"
        >
          Set Conv
        </button>
      </div>
    );
  };
  MockChatWindow.displayName = "MockChatWindow";
  return {
    __esModule: true,
    default: MockChatWindow,
  };
});

jest.mock("./components/Config/TargetConfig", () => {
  const MockTargetConfig = ({
    activeTarget,
    onSetActiveTarget,
  }: {
    activeTarget: unknown;
    onSetActiveTarget: (t: unknown) => void;
  }) => {
    return (
      <div data-testid="target-config">
        <span data-testid="active-target-name">
          {(activeTarget as { target_registry_name?: string })?.target_registry_name ?? "none"}
        </span>
        <button
          onClick={() =>
            onSetActiveTarget({
              target_id: "t1",
              target_registry_name: "test_target",
              target_type: "OpenAIChatTarget",
              status: "active",
            })
          }
          data-testid="set-target"
        >
          Set Target
        </button>
      </div>
    );
  };
  MockTargetConfig.displayName = "MockTargetConfig";
  return {
    __esModule: true,
    default: MockTargetConfig,
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

    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-dark-mode",
      "true"
    );

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

    fireEvent.click(screen.getByTestId("send-message"));
    fireEvent.click(screen.getByTestId("receive-message"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("2");

    fireEvent.click(screen.getByTestId("new-chat"));
    expect(screen.getByTestId("message-count")).toHaveTextContent("0");
  });

  it("starts in chat view", () => {
    render(<App />);

    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-current-view",
      "chat"
    );
    expect(screen.getByTestId("chat-window")).toBeInTheDocument();
  });

  it("switches to config view", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("nav-config"));

    expect(screen.getByTestId("main-layout")).toHaveAttribute(
      "data-current-view",
      "config"
    );
    expect(screen.getByTestId("target-config")).toBeInTheDocument();
  });

  it("switches back to chat from config", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("nav-config"));
    expect(screen.getByTestId("target-config")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("nav-chat"));
    expect(screen.getByTestId("chat-window")).toBeInTheDocument();
  });

  it("sets conversationId from chat window", () => {
    render(<App />);

    expect(screen.getByTestId("conversation-id")).toHaveTextContent("none");

    fireEvent.click(screen.getByTestId("set-conversation"));
    expect(screen.getByTestId("conversation-id")).toHaveTextContent("conv-123");
  });

  it("clears conversationId on new chat", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("set-conversation"));
    expect(screen.getByTestId("conversation-id")).toHaveTextContent("conv-123");

    fireEvent.click(screen.getByTestId("new-chat"));
    expect(screen.getByTestId("conversation-id")).toHaveTextContent("none");
  });

  it("sets active target from config page and passes to chat", () => {
    render(<App />);

    // No target initially
    expect(screen.getByTestId("has-target")).toHaveTextContent("no");

    // Switch to config and set target
    fireEvent.click(screen.getByTestId("nav-config"));
    fireEvent.click(screen.getByTestId("set-target"));

    // Switch back to chat — target should be present
    fireEvent.click(screen.getByTestId("nav-chat"));
    expect(screen.getByTestId("has-target")).toHaveTextContent("yes");
  });
});
