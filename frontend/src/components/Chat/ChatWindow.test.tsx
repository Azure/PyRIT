import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import ChatWindow from "./ChatWindow";
import { Message, TargetInstance } from "../../types";
import { attacksApi } from "../../services/api";
import * as messageMapper from "../../utils/messageMapper";

jest.mock("../../services/api", () => ({
  attacksApi: {
    createAttack: jest.fn(),
    addMessage: jest.fn(),
  },
}));

jest.mock("../../utils/messageMapper", () => ({
  buildMessagePieces: jest.fn(),
  backendMessagesToFrontend: jest.fn(),
}));

const mockedAttacksApi = attacksApi as jest.Mocked<typeof attacksApi>;
const mockedMapper = messageMapper as jest.Mocked<typeof messageMapper>;

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <FluentProvider theme={webLightTheme}>{children}</FluentProvider>;

const mockTarget: TargetInstance = {
  target_registry_name: "openai_chat_1",
  target_type: "OpenAIChatTarget",
  endpoint: "https://api.openai.com",
  model_name: "gpt-4",
};

// ---------------------------------------------------------------------------
// Helpers to build mock backend responses
// ---------------------------------------------------------------------------

function makeTextResponse(text: string) {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-resp",
              original_value_data_type: "text",
              converted_value_data_type: "text",
              original_value: text,
              converted_value: text,
              scores: [],
              response_error: "none",
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

function makeImageResponse() {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-img",
              original_value_data_type: "text",
              converted_value_data_type: "image_path",
              original_value: "generated image",
              converted_value: "iVBORw0KGgo=",
              converted_value_mime_type: "image/png",
              scores: [],
              response_error: "none",
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

function makeAudioResponse() {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-aud",
              original_value_data_type: "text",
              converted_value_data_type: "audio_path",
              original_value: "spoken text",
              converted_value: "UklGRg==",
              converted_value_mime_type: "audio/wav",
              scores: [],
              response_error: "none",
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

function makeVideoResponse() {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-vid",
              original_value_data_type: "text",
              converted_value_data_type: "video_path",
              original_value: "generated video",
              converted_value: "dmlkZW8=",
              converted_value_mime_type: "video/mp4",
              scores: [],
              response_error: "none",
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

function makeMultiModalResponse() {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-text",
              original_value_data_type: "text",
              converted_value_data_type: "text",
              original_value: "Here is the result:",
              converted_value: "Here is the result:",
              scores: [],
              response_error: "none",
            },
            {
              piece_id: "p-img2",
              original_value_data_type: "text",
              converted_value_data_type: "image_path",
              original_value: "image content",
              converted_value: "aW1hZ2U=",
              converted_value_mime_type: "image/jpeg",
              scores: [],
              response_error: "none",
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

function makeErrorResponse(errorType: string, description: string) {
  return {
    messages: {
      messages: [
        {
          turn_number: 1,
          role: "assistant",
          pieces: [
            {
              piece_id: "p-err",
              original_value_data_type: "text",
              converted_value_data_type: "text",
              original_value: "",
              converted_value: "",
              scores: [],
              response_error: errorType,
              response_error_description: description,
            },
          ],
          created_at: "2026-01-01T00:00:01Z",
        },
      ],
    },
  };
}

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
    messages: [] as Message[],
    onSendMessage: jest.fn(),
    onReceiveMessage: jest.fn(),
    onNewChat: jest.fn(),
    activeTarget: mockTarget,
    conversationId: null as string | null,
    onConversationCreated: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // -----------------------------------------------------------------------
  // Basic rendering
  // -----------------------------------------------------------------------

  it("should render chat window with all components", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText("PyRIT Attack")).toBeInTheDocument();
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

  it("should show target info when target is active", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText(/OpenAIChatTarget/)).toBeInTheDocument();
    expect(screen.getByText(/gpt-4/)).toBeInTheDocument();
  });

  it("should show no-target message when target is null", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} activeTarget={null} />
      </TestWrapper>
    );

    expect(
      screen.getByText(/No target selected/)
    ).toBeInTheDocument();
  });

  it("should call onNewChat when New Chat button is clicked", async () => {
    const user = userEvent.setup();
    const onNewChat = jest.fn();

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} onNewChat={onNewChat} />
      </TestWrapper>
    );

    await user.click(screen.getByText("New Chat"));

    expect(onNewChat).toHaveBeenCalled();
  });

  it("should disable input when no target is selected", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} activeTarget={null} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    expect(input).toBeDisabled();
  });

  // -----------------------------------------------------------------------
  // Target info display for various target types
  // -----------------------------------------------------------------------

  it("should display target without model name", () => {
    const targetNoModel: TargetInstance = {
      ...mockTarget,
      model_name: null,
    };

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} activeTarget={targetNoModel} />
      </TestWrapper>
    );

    expect(screen.getByText(/OpenAIChatTarget/)).toBeInTheDocument();
    expect(screen.queryByText(/gpt/)).not.toBeInTheDocument();
  });

  // -----------------------------------------------------------------------
  // First message → create attack + send
  // -----------------------------------------------------------------------

  it("should create attack and send text message on first message", async () => {
    const user = userEvent.setup();
    const onSendMessage = jest.fn();
    const onReceiveMessage = jest.fn();
    const onConversationCreated = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Hello" },
    ]);
    mockedAttacksApi.createAttack.mockResolvedValue({
      conversation_id: "conv-1",
      created_at: "2026-01-01T00:00:00Z",
    });
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Hello back!") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Hello back!",
        timestamp: "2026-01-01T00:00:00Z",
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          onSendMessage={onSendMessage}
          onReceiveMessage={onReceiveMessage}
          onConversationCreated={onConversationCreated}
          conversationId={null}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onSendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ role: "user", content: "Hello" })
      );
      expect(mockedAttacksApi.createAttack).toHaveBeenCalledWith({
        target_registry_name: "openai_chat_1",
      });
      expect(onConversationCreated).toHaveBeenCalledWith("conv-1");
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith("conv-1", {
        role: "user",
        pieces: [{ data_type: "text", original_value: "Hello" }],
        send: true,
        target_registry_name: "openai_chat_1",
      });
    });
  });

  // -----------------------------------------------------------------------
  // Subsequent messages → reuse conversation ID
  // -----------------------------------------------------------------------

  it("should reuse conversationId on subsequent messages", async () => {
    const user = userEvent.setup();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Second" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Response") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Response",
        timestamp: "2026-01-01T00:00:01Z",
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} conversationId="existing-conv" />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Second");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.createAttack).not.toHaveBeenCalled();
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith(
        "existing-conv",
        expect.any(Object)
      );
    });
  });

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------

  it("should show error message when API call fails", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "test" },
    ]);
    mockedAttacksApi.createAttack.mockRejectedValue(
      new Error("Network error")
    );

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId={null}
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "test");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          error: expect.objectContaining({
            type: "unknown",
            description: "Network error",
          }),
        })
      );
    });
  });

  it("should show error message when addMessage fails", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "test" },
    ]);
    mockedAttacksApi.createAttack.mockResolvedValue({
      conversation_id: "conv-err",
      created_at: "2026-01-01T00:00:00Z",
    });
    mockedAttacksApi.addMessage.mockRejectedValue(
      new Error("Request failed with status code 404")
    );

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId={null}
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "test");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          error: expect.objectContaining({
            description: "Request failed with status code 404",
          }),
        })
      );
    });
  });

  it("should show generic error for non-Error thrown values", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "test" },
    ]);
    mockedAttacksApi.addMessage.mockRejectedValue("string error");

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-x"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "test");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            description: "Failed to send message",
          }),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Loading indicator flow
  // -----------------------------------------------------------------------

  it("should show loading then replace with response", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Hello" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Hi!") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Hi!",
        timestamp: "2026-01-01T00:00:01Z",
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-2"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      // First call: loading message
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({ content: "...", isLoading: true })
      );
      // Second call: actual response
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({ content: "Hi!" })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-modal: image response
  // -----------------------------------------------------------------------

  it("should handle image response from backend", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Generate an image" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeImageResponse() as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "",
        timestamp: "2026-01-01T00:00:01Z",
        attachments: [
          {
            type: "image" as const,
            name: "image_path_p-img",
            url: "data:image/png;base64,iVBORw0KGgo=",
            mimeType: "image/png",
            size: 12,
          },
        ],
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-img"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Generate an image");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      // The response should include the image attachment
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          attachments: expect.arrayContaining([
            expect.objectContaining({ type: "image" }),
          ]),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-modal: audio response
  // -----------------------------------------------------------------------

  it("should handle audio response from backend", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Read this aloud" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeAudioResponse() as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "",
        timestamp: "2026-01-01T00:00:01Z",
        attachments: [
          {
            type: "audio" as const,
            name: "audio_path_p-aud",
            url: "data:audio/wav;base64,UklGRg==",
            mimeType: "audio/wav",
            size: 8,
          },
        ],
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-audio"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Read this aloud");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          attachments: expect.arrayContaining([
            expect.objectContaining({ type: "audio" }),
          ]),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-modal: video response
  // -----------------------------------------------------------------------

  it("should handle video response from backend", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Create a video" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeVideoResponse() as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "",
        timestamp: "2026-01-01T00:00:01Z",
        attachments: [
          {
            type: "video" as const,
            name: "video_path_p-vid",
            url: "data:video/mp4;base64,dmlkZW8=",
            mimeType: "video/mp4",
            size: 8,
          },
        ],
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-video"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Create a video");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          attachments: expect.arrayContaining([
            expect.objectContaining({ type: "video" }),
          ]),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-modal: mixed text + image response
  // -----------------------------------------------------------------------

  it("should handle mixed text + image response", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Describe and show" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeMultiModalResponse() as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Here is the result:",
        timestamp: "2026-01-01T00:00:01Z",
        attachments: [
          {
            type: "image" as const,
            name: "image_path_p-img2",
            url: "data:image/jpeg;base64,aW1hZ2U=",
            mimeType: "image/jpeg",
            size: 8,
          },
        ],
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-multi"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Describe and show");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          content: "Here is the result:",
          attachments: expect.arrayContaining([
            expect.objectContaining({ type: "image" }),
          ]),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Sending image attachment
  // -----------------------------------------------------------------------

  it("should send image attachment alongside text", async () => {
    const user = userEvent.setup();
    const onSendMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "What is this?" },
      {
        data_type: "image_path",
        original_value: "iVBORw0KGgo=",
        mime_type: "image/png",
      },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("It's a cat.") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "It's a cat.",
        timestamp: "2026-01-01T00:00:01Z",
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-attach"
          onSendMessage={onSendMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "What is this?");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith(
        "conv-attach",
        expect.objectContaining({
          pieces: [
            { data_type: "text", original_value: "What is this?" },
            {
              data_type: "image_path",
              original_value: "iVBORw0KGgo=",
              mime_type: "image/png",
            },
          ],
          send: true,
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Sending audio attachment
  // -----------------------------------------------------------------------

  it("should send audio attachment", async () => {
    const user = userEvent.setup();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      {
        data_type: "audio_path",
        original_value: "UklGRg==",
        mime_type: "audio/wav",
      },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(
      makeTextResponse("Transcribed: hello") as never
    );
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Transcribed: hello",
        timestamp: "2026-01-01T00:00:01Z",
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} conversationId="conv-aud-send" />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Listen");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith(
        "conv-aud-send",
        expect.objectContaining({
          pieces: [
            {
              data_type: "audio_path",
              original_value: "UklGRg==",
              mime_type: "audio/wav",
            },
          ],
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Backend error in response piece (blocked, processing, etc.)
  // -----------------------------------------------------------------------

  it("should handle blocked response from target", async () => {
    const user = userEvent.setup();
    const onReceiveMessage = jest.fn();

    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "bad prompt" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(
      makeErrorResponse("blocked", "Content was filtered by safety system") as never
    );
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "",
        timestamp: "2026-01-01T00:00:01Z",
        error: {
          type: "blocked",
          description: "Content was filtered by safety system",
        },
      },
    ]);

    render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-block"
          onReceiveMessage={onReceiveMessage}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "bad prompt");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(onReceiveMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          role: "assistant",
          error: expect.objectContaining({ type: "blocked" }),
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-turn conversation
  // -----------------------------------------------------------------------

  it("should support multi-turn: create on first, reuse on second", async () => {
    const user = userEvent.setup();
    const onConversationCreated = jest.fn();
    const onSendMessage = jest.fn();
    const onReceiveMessage = jest.fn();

    // First message
    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Turn 1" },
    ]);
    mockedAttacksApi.createAttack.mockResolvedValue({
      conversation_id: "conv-multi-turn",
      created_at: "2026-01-01T00:00:00Z",
    });
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Reply 1") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Reply 1",
        timestamp: "2026-01-01T00:00:01Z",
      },
    ]);

    const { rerender } = render(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId={null}
          onSendMessage={onSendMessage}
          onReceiveMessage={onReceiveMessage}
          onConversationCreated={onConversationCreated}
        />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Turn 1");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.createAttack).toHaveBeenCalledTimes(1);
      expect(onConversationCreated).toHaveBeenCalledWith("conv-multi-turn");
    });

    // Now rerender with the conversation ID set (simulating parent state update)
    jest.clearAllMocks();
    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Turn 2" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Reply 2") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      {
        role: "assistant",
        content: "Reply 2",
        timestamp: "2026-01-01T00:00:02Z",
      },
    ]);

    rerender(
      <TestWrapper>
        <ChatWindow
          {...defaultProps}
          conversationId="conv-multi-turn"
          onSendMessage={onSendMessage}
          onReceiveMessage={onReceiveMessage}
          onConversationCreated={onConversationCreated}
        />
      </TestWrapper>
    );

    await user.type(screen.getByRole("textbox"), "Turn 2");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.createAttack).not.toHaveBeenCalled();
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith(
        "conv-multi-turn",
        expect.objectContaining({
          pieces: [{ data_type: "text", original_value: "Turn 2" }],
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // Multi-turn with mixed modalities
  // -----------------------------------------------------------------------

  it("should support sending text first then image in second turn", async () => {
    const user = userEvent.setup();

    // Turn 1: text
    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "Hello" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("Hi!") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      { role: "assistant", content: "Hi!", timestamp: "2026-01-01T00:00:01Z" },
    ]);

    const { rerender } = render(
      <TestWrapper>
        <ChatWindow {...defaultProps} conversationId="conv-mixed-turns" />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledTimes(1);
    });

    // Turn 2: text + image
    jest.clearAllMocks();
    mockedMapper.buildMessagePieces.mockResolvedValue([
      { data_type: "text", original_value: "What is this?" },
      { data_type: "image_path", original_value: "base64data", mime_type: "image/png" },
    ]);
    mockedAttacksApi.addMessage.mockResolvedValue(makeTextResponse("A cat") as never);
    mockedMapper.backendMessagesToFrontend.mockReturnValue([
      { role: "assistant", content: "A cat", timestamp: "2026-01-01T00:00:02Z" },
    ]);

    rerender(
      <TestWrapper>
        <ChatWindow {...defaultProps} conversationId="conv-mixed-turns" />
      </TestWrapper>
    );

    await user.type(screen.getByRole("textbox"), "What is this?");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      expect(mockedAttacksApi.addMessage).toHaveBeenCalledWith(
        "conv-mixed-turns",
        expect.objectContaining({
          pieces: [
            { data_type: "text", original_value: "What is this?" },
            { data_type: "image_path", original_value: "base64data", mime_type: "image/png" },
          ],
        })
      );
    });
  });

  // -----------------------------------------------------------------------
  // No message sent when target is null (guard)
  // -----------------------------------------------------------------------

  it("should not send message when active target is null", () => {
    render(
      <TestWrapper>
        <ChatWindow {...defaultProps} activeTarget={null} />
      </TestWrapper>
    );

    // Input and send should be disabled
    expect(screen.getByRole("textbox")).toBeDisabled();
  });
});
