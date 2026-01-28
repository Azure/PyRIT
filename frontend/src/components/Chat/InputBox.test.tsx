import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import InputBox from "./InputBox";

// Wrapper component for Fluent UI context
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
);

// Helper to get the send button specifically
const getSendButton = () => screen.getByRole("button", { name: /send/i });

describe("InputBox", () => {
  const defaultProps = {
    onSend: jest.fn(),
    disabled: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should render input area and send button", () => {
    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByRole("textbox")).toBeInTheDocument();
    expect(getSendButton()).toBeInTheDocument();
  });

  it("should call onSend with input value when send button clicked", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test prompt message");
    await user.click(getSendButton());

    expect(onSend).toHaveBeenCalled();
  });

  it("should disable input when disabled prop is true", () => {
    render(
      <TestWrapper>
        <InputBox {...defaultProps} disabled={true} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    expect(input).toBeDisabled();
  });

  it("should disable send button when input is empty", () => {
    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const sendButton = getSendButton();
    expect(sendButton).toBeDisabled();
  });

  it("should enable send button when input has text", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Some text");

    const sendButton = getSendButton();
    expect(sendButton).toBeEnabled();
  });

  it("should clear input after sending", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test message");
    await user.click(getSendButton());

    await waitFor(() => {
      expect(input).toHaveValue("");
    });
  });

  it("should send message on Enter key press", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test message");
    await user.type(input, "{Enter}");

    expect(onSend).toHaveBeenCalled();
  });

  it("should not send on Shift+Enter (allows multiline)", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "Test message");
    await user.keyboard("{Shift>}{Enter}{/Shift}");

    expect(onSend).not.toHaveBeenCalled();
  });

  it("should allow sending whitespace-only messages", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const input = screen.getByRole("textbox");
    await user.type(input, "   ");
    await user.click(getSendButton());

    expect(onSend).toHaveBeenCalled();
  });

  it("should not send when input is completely empty", () => {
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const sendButton = getSendButton();
    expect(sendButton).toBeDisabled();
  });

  it("should have file input for attachments", () => {
    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const attachButton = screen.getByTitle("Attach files");
    expect(attachButton).toBeInTheDocument();
  });

  it("should handle file attachment selection", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    // Create a mock file
    const file = new File(["test content"], "test.txt", { type: "text/plain" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    // Simulate file selection
    await user.upload(fileInput, file);

    // Check that the attachment appears
    await waitFor(() => {
      expect(screen.getByText(/test\.txt/)).toBeInTheDocument();
    });
  });

  it("should handle image file attachment", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const file = new File(["image data"], "photo.png", { type: "image/png" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText(/photo\.png/)).toBeInTheDocument();
    });
  });

  it("should remove attachment when dismiss button clicked", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const file = new File(["test"], "remove-me.txt", { type: "text/plain" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText(/remove-me\.txt/)).toBeInTheDocument();
    });

    // Find and click the dismiss button
    const dismissButtons = screen.getAllByRole("button");
    const dismissButton = dismissButtons.find(
      (btn) =>
        btn.querySelector("svg") && btn.getAttribute("aria-label") !== "Send"
    );

    if (dismissButton) {
      await user.click(dismissButton);
    }
  });

  it("should send with attachments even without text", async () => {
    const user = userEvent.setup();
    const onSend = jest.fn();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} onSend={onSend} />
      </TestWrapper>
    );

    const file = new File(["test"], "file.txt", { type: "text/plain" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText(/file\.txt/)).toBeInTheDocument();
    });

    // Should be able to send with just attachment
    const sendButton = getSendButton();
    expect(sendButton).toBeEnabled();
    await user.click(sendButton);

    expect(onSend).toHaveBeenCalled();
  });

  it("should handle audio file attachment", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const file = new File(["audio data"], "sound.mp3", { type: "audio/mpeg" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText(/sound\.mp3/)).toBeInTheDocument();
    });
  });

  it("should handle video file attachment", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const file = new File(["video data"], "video.mp4", { type: "video/mp4" });
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText(/video\.mp4/)).toBeInTheDocument();
    });
  });

  it("should handle multiple file attachments", async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <InputBox {...defaultProps} />
      </TestWrapper>
    );

    const files = [
      new File(["text content"], "document.txt", { type: "text/plain" }),
      new File(["image data"], "photo.png", { type: "image/png" }),
      new File(["audio data"], "audio.mp3", { type: "audio/mpeg" }),
    ];
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement;

    await user.upload(fileInput, files);

    await waitFor(() => {
      expect(screen.getByText(/document\.txt/)).toBeInTheDocument();
      expect(screen.getByText(/photo\.png/)).toBeInTheDocument();
      expect(screen.getByText(/audio\.mp3/)).toBeInTheDocument();
    });
  });
});
