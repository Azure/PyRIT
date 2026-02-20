import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import TargetConfig from "./TargetConfig";
import { targetsApi } from "../../services/api";
import type { TargetInstance } from "../../types";

jest.mock("../../services/api", () => ({
  targetsApi: {
    listTargets: jest.fn(),
    createTarget: jest.fn(),
  },
}));

jest.mock("./CreateTargetDialog", () => {
  const MockDialog = ({
    open,
    onClose,
    onCreated,
  }: {
    open: boolean;
    onClose: () => void;
    onCreated: () => void;
  }) => {
    if (!open) return null;
    return (
      <div data-testid="create-dialog">
        <button onClick={onClose} data-testid="dialog-close">
          Cancel
        </button>
        <button onClick={onCreated} data-testid="dialog-create">
          Create
        </button>
      </div>
    );
  };
  MockDialog.displayName = "MockCreateTargetDialog";
  return {
    __esModule: true,
    default: MockDialog,
  };
});

const mockedTargetsApi = targetsApi as jest.Mocked<typeof targetsApi>;

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <FluentProvider theme={webLightTheme}>{children}</FluentProvider>;

const sampleTargets: TargetInstance[] = [
  {
    target_registry_name: "openai_chat_gpt4",
    target_type: "OpenAIChatTarget",
    endpoint: "https://api.openai.com",
    model_name: "gpt-4",
  },
  {
    target_registry_name: "openai_image_dalle",
    target_type: "OpenAIImageTarget",
    endpoint: "https://api.openai.com",
    model_name: "dall-e-3",
  },
];

describe("TargetConfig", () => {
  const defaultProps = {
    activeTarget: null as TargetInstance | null,
    onSetActiveTarget: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should show loading state initially", () => {
    mockedTargetsApi.listTargets.mockReturnValue(new Promise(() => {})); // never resolves

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText("Loading targets...")).toBeInTheDocument();
  });

  it("should render target list after loading", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("openai_chat_gpt4")).toBeInTheDocument();
      expect(screen.getByText("openai_image_dalle")).toBeInTheDocument();
    });
  });

  it("should show empty state when no targets", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: [],
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });
  });

  it(
    "should show error state on API failure",
    async () => {
      mockedTargetsApi.listTargets.mockRejectedValue(
        new Error("Connection refused")
      );

      render(
        <TestWrapper>
          <TargetConfig {...defaultProps} />
        </TestWrapper>
      );

      await waitFor(
        () => {
          expect(screen.getByText(/Connection refused/)).toBeInTheDocument();
        },
        { timeout: 15000 }
      );
    },
    20000
  );

  it("should call onSetActiveTarget when Set Active is clicked", async () => {
    const onSetActiveTarget = jest.fn();
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig
          {...defaultProps}
          onSetActiveTarget={onSetActiveTarget}
        />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("openai_chat_gpt4")).toBeInTheDocument();
    });

    const setActiveButtons = screen.getAllByText("Set Active");
    await userEvent.click(setActiveButtons[0]);

    expect(onSetActiveTarget).toHaveBeenCalledWith(sampleTargets[0]);
  });

  it("should show Active badge for active target", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig
          {...defaultProps}
          activeTarget={sampleTargets[0]}
        />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("Active")).toBeInTheDocument();
    });
  });

  it("should refresh targets when Refresh button is clicked", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("openai_chat_gpt4")).toBeInTheDocument();
    });

    expect(mockedTargetsApi.listTargets).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByText("Refresh"));

    await waitFor(() => {
      expect(mockedTargetsApi.listTargets).toHaveBeenCalledTimes(2);
    });
  });

  it("should open create dialog when New Target is clicked", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: [],
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText("New Target"));

    expect(screen.getByTestId("create-dialog")).toBeInTheDocument();
  });

  it("should refresh list after target creation", async () => {
    mockedTargetsApi.listTargets
      .mockResolvedValueOnce({ items: [], pagination: { limit: 200, has_more: false } })
      .mockResolvedValueOnce({ items: sampleTargets, pagination: { limit: 200, has_more: false } });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });

    // Open dialog and trigger create
    await userEvent.click(screen.getByText("New Target"));
    await userEvent.click(screen.getByTestId("dialog-create"));

    await waitFor(() => {
      expect(screen.getByText("openai_chat_gpt4")).toBeInTheDocument();
    });
  });

  it("should display target type, endpoint, and model", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("OpenAIChatTarget")).toBeInTheDocument();
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(
        screen.getAllByText("https://api.openai.com").length
      ).toBeGreaterThanOrEqual(1);
    });
  });
});
