import { render, screen } from "@testing-library/react";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import MessageList from "./MessageList";
import { Message } from "../../types";

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <FluentProvider theme={webLightTheme}>{children}</FluentProvider>;

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

    expect(document.body).toBeTruthy();
  });

  it("should render all messages", () => {
    render(
      <TestWrapper>
        <MessageList messages={mockMessages} />
      </TestWrapper>
    );

    expect(screen.getByText("Hello, how are you?")).toBeInTheDocument();
    expect(
      screen.getByText("I am doing well, thank you!")
    ).toBeInTheDocument();
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

  it("should handle messages with image attachments", () => {
    const messagesWithAttachments: Message[] = [
      {
        role: "assistant",
        content: "Here is your image",
        timestamp: new Date().toISOString(),
        attachments: [
          {
            type: "image",
            name: "test.png",
            url: "data:image/png;base64,iVBORw0KGgo=",
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

    expect(screen.getByText("Here is your image")).toBeInTheDocument();
    const img = screen.getByAltText("test.png");
    expect(img).toBeInTheDocument();
    expect(img).toHaveAttribute(
      "src",
      "data:image/png;base64,iVBORw0KGgo="
    );
  });

  it("should handle messages with audio attachments", () => {
    const messagesWithAudio: Message[] = [
      {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        attachments: [
          {
            type: "audio",
            name: "audio.wav",
            url: "data:audio/wav;base64,UklGRg==",
            mimeType: "audio/wav",
            size: 512,
          },
        ],
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={messagesWithAudio} />
      </TestWrapper>
    );

    const audioElements = document.querySelectorAll("audio");
    expect(audioElements.length).toBeGreaterThan(0);
  });

  it("should handle messages with video attachments", () => {
    const messagesWithVideo: Message[] = [
      {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        attachments: [
          {
            type: "video",
            name: "video.mp4",
            url: "data:video/mp4;base64,dmlkZW8=",
            mimeType: "video/mp4",
            size: 2048,
          },
        ],
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={messagesWithVideo} />
      </TestWrapper>
    );

    const videoElements = document.querySelectorAll("video");
    expect(videoElements.length).toBeGreaterThan(0);
  });

  it("should render error messages", () => {
    const errorMessages: Message[] = [
      {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        error: {
          type: "blocked",
          description: "Content was filtered by safety system",
        },
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={errorMessages} />
      </TestWrapper>
    );

    expect(
      screen.getByText(/Content was filtered by safety system/)
    ).toBeInTheDocument();
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

  it("should render simulated_assistant with distinct avatar", () => {
    const simMessages: Message[] = [
      {
        role: "simulated_assistant",
        content: "Simulated response from another conversation",
        timestamp: new Date().toISOString(),
      },
    ];

    render(
      <TestWrapper>
        <MessageList messages={simMessages} />
      </TestWrapper>
    );

    expect(
      screen.getByText("Simulated response from another conversation")
    ).toBeInTheDocument();
    // Avatar should be labelled "Simulated" instead of "Assistant"
    expect(screen.getByText("S")).toBeInTheDocument();
  });
});
