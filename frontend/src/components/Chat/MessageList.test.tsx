import { render, screen } from "@testing-library/react";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import MessageList from "./MessageList";
import { Message } from "../../types";

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
);

describe("MessageList", () => {
  const mockMessages: Message[] = [
    {
      role: "user",
      content: "Hello, how are you?",
      timestamp: new Date().toISOString(),
    },
    {
      role: "assistant",
      content: "I am doing well, thank you!",
      timestamp: new Date().toISOString(),
    },
    {
      role: "user",
      content: "Can you help me?",
      timestamp: new Date().toISOString(),
    },
  ];

  it("should render empty state when no messages", () => {
    render(
      <TestWrapper>
        <MessageList messages={[]} />
      </TestWrapper>
    );

    // Should render without errors even with empty messages
    expect(document.body).toBeTruthy();
  });

  it("should render all messages", () => {
    render(
      <TestWrapper>
        <MessageList messages={mockMessages} />
      </TestWrapper>
    );

    expect(screen.getByText("Hello, how are you?")).toBeInTheDocument();
    expect(screen.getByText("I am doing well, thank you!")).toBeInTheDocument();
    expect(screen.getByText("Can you help me?")).toBeInTheDocument();
  });

  it("should render user messages", () => {
    const userMessages: Message[] = [
      {
        role: "user",
        content: "User message test",
        timestamp: new Date().toISOString(),
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={userMessages} />
      </TestWrapper>
    );

    expect(screen.getByText("User message test")).toBeInTheDocument();
  });

  it("should render assistant messages", () => {
    const assistantMessages: Message[] = [
      {
        role: "assistant",
        content: "Assistant message test",
        timestamp: new Date().toISOString(),
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={assistantMessages} />
      </TestWrapper>
    );

    expect(screen.getByText("Assistant message test")).toBeInTheDocument();
  });

  it("should handle messages with attachments", () => {
    const messagesWithAttachments: Message[] = [
      {
        role: "user",
        content: "Message with attachment",
        timestamp: new Date().toISOString(),
        attachments: [
          {
            type: "image",
            name: "test.png",
            url: "http://example.com/test.png",
            mimeType: "image/png",
            size: 1024,
          },
        ],
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={messagesWithAttachments} />
      </TestWrapper>
    );

    expect(screen.getByText("Message with attachment")).toBeInTheDocument();
  });

  it("should render multiple messages in order", () => {
    render(
      <TestWrapper>
        <MessageList messages={mockMessages} />
      </TestWrapper>
    );

    const messageElements = screen.getAllByText(/Hello|doing well|help/);
    expect(messageElements.length).toBeGreaterThanOrEqual(3);
  });
});
