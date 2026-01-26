import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import ChatWindow from "./ChatWindow";
import { Message } from "../../types";

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
);

describe("ChatWindow Integration", () => {
  const mockMessages: Message[] = [
    {
      role: "user",
      content: "Hello",
      timestamp: new Date().toISOString(),
    },
    {
      role: "assistant",
      content: "Hi there!",
      timestamp: new Date().toISOString(),
    },
  ];

  const defaultProps = {
    messages: [],
    onSendMessage: jest.fn(),
    onReceiveMessage: jest.fn(),
    onNewChat: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("should render chat window with all components", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText("PyRIT Frontend")).toBeInTheDocument();
    expect(screen.getByText("New Chat")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });

  it("should display existing messages", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} messages={mockMessages} />
      </TestWrapper>
    );

    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should call onNewChat when New Chat button is clicked", async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onNewChat = jest.fn();

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} onNewChat={onNewChat} />
      </TestWrapper>
    );

    await user.click(screen.getByText("New Chat"));

    expect(onNewChat).toHaveBeenCalled();
  });

  it("should call onSendMessage when message is sent", async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onSendMessage = jest.fn();

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} onSendMessage={onSendMessage} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test message");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(onSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        role: "user",
        content: "Test message",
      })
    );
  });

  it("should call onReceiveMessage after sending", async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onReceiveMessage = jest.fn();

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} onReceiveMessage={onReceiveMessage} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    // Advance timers to trigger the echo response (wrapped in act)
    await act(async () => {
      jest.advanceTimersByTime(600);
    });

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          content: "Echo: Hello",
        })
      );
    });
  });

  it("should disable input while sending", async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test");
    await user.click(screen.getByRole("button", { name: /send/i }));

    // Input should be disabled while waiting for response
    expect(input).toBeDisabled();

    // Advance timers to complete the send (wrapped in act)
    await act(async () => {
      jest.advanceTimersByTime(600);
    });

    await waitFor(() => {
      expect(input).not.toBeDisabled();
    });
  });
});
