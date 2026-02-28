import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import AttackHistory from './AttackHistory'
import { attacksApi, labelsApi } from '../../services/api'

jest.mock('../../services/api', () => ({
  attacksApi: {
    listAttacks: jest.fn(),
    getAttackOptions: jest.fn(),
    getConverterOptions: jest.fn(),
  },
  labelsApi: {
    getLabels: jest.fn(),
  },
}))

const mockedAttacksApi = attacksApi as jest.Mocked<typeof attacksApi>
const mockedLabelsApi = labelsApi as jest.Mocked<typeof labelsApi>

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

const sampleAttacks = [
  {
    attack_result_id: 'ar-conv-1',
    conversation_id: 'conv-1',
    attack_type: 'CrescendoAttack',
    attack_specific_params: null,
    target: { target_type: 'OpenAIChatTarget', endpoint: 'https://api.openai.com', model_name: 'gpt-4' },
    converters: ['Base64Converter'],
    outcome: 'success' as const,
    last_message_preview: 'The model responded with...',
    message_count: 10,
    related_conversation_ids: ['rel-1', 'rel-2'],
    labels: { category: 'test' },
    created_at: '2026-01-15T10:30:00Z',
    updated_at: '2026-01-15T11:00:00Z',
  },
  {
    attack_result_id: 'ar-conv-2',
    conversation_id: 'conv-2',
    attack_type: 'ManualAttack',
    attack_specific_params: null,
    target: { target_type: 'OpenAIImageTarget', endpoint: 'https://api.openai.com', model_name: 'dall-e-3' },
    converters: [],
    outcome: 'failure' as const,
    last_message_preview: null,
    message_count: 2,
    related_conversation_ids: [],
    labels: {},
    created_at: '2026-01-14T08:00:00Z',
    updated_at: '2026-01-14T08:30:00Z',
  },
]

describe('AttackHistory', () => {
  const defaultProps = {
    onOpenAttack: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
    mockedAttacksApi.getAttackOptions.mockImplementation(() => new Promise(() => {}))
    mockedAttacksApi.getConverterOptions.mockImplementation(() => new Promise(() => {}))
    mockedLabelsApi.getLabels.mockImplementation(() => new Promise(() => {}))
  })

  it('should render title and filters', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: [],
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    expect(screen.getByText('Attack History')).toBeInTheDocument()
    expect(screen.getByTestId('refresh-btn')).toBeInTheDocument()
    expect(screen.getByTestId('attack-class-filter')).toBeInTheDocument()
    expect(screen.getByTestId('outcome-filter')).toBeInTheDocument()
    expect(screen.getByTestId('converter-filter')).toBeInTheDocument()
    expect(screen.getByTestId('operator-filter')).toBeInTheDocument()
    expect(screen.getByTestId('operation-filter')).toBeInTheDocument()
    expect(screen.getByTestId('label-filter')).toBeInTheDocument()

    await waitFor(() => {
      expect(mockedAttacksApi.listAttacks).toHaveBeenCalledTimes(1)
    })
  })

  it('should show empty state when no attacks', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: [],
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toBeInTheDocument()
    })
    expect(screen.getByText('No attacks found')).toBeInTheDocument()
  })

  it('should render attack table rows', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('attacks-table')).toBeInTheDocument()
    })
    expect(screen.getByTestId('attack-row-ar-conv-1')).toBeInTheDocument()
    expect(screen.getByTestId('attack-row-ar-conv-2')).toBeInTheDocument()
    expect(screen.getByText('CrescendoAttack')).toBeInTheDocument()
    expect(screen.getByText('ManualAttack')).toBeInTheDocument()
  })

  it('should show outcome badges', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('outcome-badge-ar-conv-1')).toBeInTheDocument()
    })
    expect(screen.getByText('success')).toBeInTheDocument()
    expect(screen.getByText('failure')).toBeInTheDocument()
  })

  it('should show target info as badge', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText('gpt-4')).toBeInTheDocument()
    })
    expect(screen.getByText('dall-e-3')).toBeInTheDocument()
  })

  it('should show message counts', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText('10')).toBeInTheDocument()
    })
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('should show conversation count from related_conversation_ids + 1', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    // conv-1 has 2 related conversations + 1 main = 3
    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument()
    })
    // conv-2 has 0 related conversations + 1 main = 1
    expect(screen.getByText('1')).toBeInTheDocument()
  })

  it('should call onOpenAttack when row is clicked', async () => {
    const onOpenAttack = jest.fn()
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} onOpenAttack={onOpenAttack} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('attack-row-ar-conv-1')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('attack-row-ar-conv-1'))
    expect(onOpenAttack).toHaveBeenCalledWith('ar-conv-1')
  })

  it('should call onOpenAttack when open button is clicked', async () => {
    const onOpenAttack = jest.fn()
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} onOpenAttack={onOpenAttack} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('open-attack-ar-conv-2')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('open-attack-ar-conv-2'))
    expect(onOpenAttack).toHaveBeenCalledWith('ar-conv-2')
  })

  it('should show last message preview when available', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText('The model responded with...')).toBeInTheDocument()
    })
  })

  it('should show converter badges', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText('Base64Converter')).toBeInTheDocument()
    })
  })

  it('should show label badges', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText('category: test')).toBeInTheDocument()
    })
  })

  it('should show refresh button and call listAttacks on click', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: [],
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toBeInTheDocument()
    })

    expect(mockedAttacksApi.listAttacks).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId('refresh-btn'))
    await waitFor(() => {
      expect(mockedAttacksApi.listAttacks).toHaveBeenCalledTimes(2)
    })
  })

  it('should show pagination when there are results', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: true, next_cursor: 'cursor-2' },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('next-page-btn')).toBeInTheDocument()
    })
    expect(screen.getByTestId('prev-page-btn')).toBeInTheDocument()
    expect(screen.getByTestId('next-page-btn')).toBeEnabled()
    expect(screen.getByTestId('prev-page-btn')).toBeDisabled()
  })

  it('should disable next page button when no more results', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('next-page-btn')).toBeDisabled()
    })
  })

  it('should load filter options on mount', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: [],
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(mockedAttacksApi.getAttackOptions).toHaveBeenCalledTimes(1)
    })
    expect(mockedAttacksApi.getConverterOptions).toHaveBeenCalledTimes(1)
    expect(mockedLabelsApi.getLabels).toHaveBeenCalledTimes(1)
  })

  it('should render table header columns', async () => {
    mockedAttacksApi.listAttacks.mockResolvedValue({
      items: sampleAttacks,
      pagination: { limit: 25, has_more: false },
    })

    render(
      <TestWrapper>
        <AttackHistory {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('attacks-table')).toBeInTheDocument()
    })
    expect(screen.getByText('Status')).toBeInTheDocument()
    expect(screen.getByText('Attack Type')).toBeInTheDocument()
    expect(screen.getByText('Target')).toBeInTheDocument()
    expect(screen.getByText('Msgs')).toBeInTheDocument()
    expect(screen.getByText('Convs')).toBeInTheDocument()
    expect(screen.getByText('Converters')).toBeInTheDocument()
    expect(screen.getByText('Labels')).toBeInTheDocument()
    expect(screen.getByText('Last Message')).toBeInTheDocument()
    expect(screen.getByText('Created')).toBeInTheDocument()
    expect(screen.getByText('Updated')).toBeInTheDocument()
  })
})
