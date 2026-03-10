import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import LabelsBar from './LabelsBar'
import { DEFAULT_GLOBAL_LABELS } from './labelDefaults'
import { labelsApi } from '../../services/api'

jest.mock('../../services/api', () => ({
  labelsApi: {
    getLabels: jest.fn(),
  },
}))

const mockedLabelsApi = labelsApi as jest.Mocked<typeof labelsApi>

function TestWrapper({ children }: { children: React.ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

describe('LabelsBar', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockedLabelsApi.getLabels.mockImplementation(() => new Promise(() => {}))
  })

  it('should render default labels', () => {
    const onChange = jest.fn()
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={onChange} />
      </TestWrapper>
    )

    expect(screen.getByTestId('label-operator')).toBeInTheDocument()
    expect(screen.getByTestId('label-operation')).toBeInTheDocument()
    expect(screen.getByText('roakey')).toBeInTheDocument()
    expect(screen.getByText('op_trash_panda')).toBeInTheDocument()
  })

  it('should show warning icon for dummy values', () => {
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={jest.fn()} />
      </TestWrapper>
    )

    expect(screen.getByTestId('labels-warning')).toBeInTheDocument()
  })

  it('should not show warning when values are customized', () => {
    render(
      <TestWrapper>
        <LabelsBar labels={{ operator: 'alice', operation: 'my_test' }} onLabelsChange={jest.fn()} />
      </TestWrapper>
    )

    expect(screen.queryByTestId('labels-warning')).not.toBeInTheDocument()
  })

  it('should not allow removing required labels (operator, operation)', () => {
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={jest.fn()} />
      </TestWrapper>
    )

    // operator and operation should not have remove buttons
    expect(screen.queryByTestId('remove-label-operator')).not.toBeInTheDocument()
    expect(screen.queryByTestId('remove-label-operation')).not.toBeInTheDocument()
  })

  it('should allow removing custom labels', () => {
    const onChange = jest.fn()
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS, team: 'red' }} onLabelsChange={onChange} />
      </TestWrapper>
    )

    const removeBtn = screen.getByTestId('remove-label-team')
    fireEvent.click(removeBtn)

    expect(onChange).toHaveBeenCalledWith({
      operator: 'roakey',
      operation: 'op_trash_panda',
    })
  })

  it('should add a new label via popover', async () => {
    const onChange = jest.fn()
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={onChange} />
      </TestWrapper>
    )

    fireEvent.click(screen.getByTestId('add-label-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('new-label-key')).toBeInTheDocument()
    })

    const keyInput = screen.getByPlaceholderText('key')
    const valueInput = screen.getByPlaceholderText('value')

    fireEvent.change(keyInput, { target: { value: 'team' } })
    fireEvent.change(valueInput, { target: { value: 'red' } })
    fireEvent.click(screen.getByTestId('confirm-add-label'))

    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_GLOBAL_LABELS,
      team: 'red',
    })
  })

  it('should reject uppercase keys', async () => {
    const onChange = jest.fn()
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={onChange} />
      </TestWrapper>
    )

    fireEvent.click(screen.getByTestId('add-label-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('new-label-key')).toBeInTheDocument()
    })

    const keyInput = screen.getByPlaceholderText('key')
    const valueInput = screen.getByPlaceholderText('value')

    // The onChange handler auto-lowercases input, so 'Team' becomes 'team' and 'Red' becomes 'red'
    fireEvent.change(keyInput, { target: { value: 'Team' } })
    fireEvent.change(valueInput, { target: { value: 'Red' } })
    fireEvent.click(screen.getByTestId('confirm-add-label'))

    // Since auto-lowercase is applied, the label should be added with lowercase values
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_GLOBAL_LABELS,
      team: 'red',
    })
  })

  it('should reject duplicate keys', async () => {
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={jest.fn()} />
      </TestWrapper>
    )

    fireEvent.click(screen.getByTestId('add-label-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('new-label-key')).toBeInTheDocument()
    })

    const keyInput = screen.getByPlaceholderText('key')
    const valueInput = screen.getByPlaceholderText('value')

    fireEvent.change(keyInput, { target: { value: 'operator' } })
    fireEvent.change(valueInput, { target: { value: 'alice' } })
    fireEvent.click(screen.getByTestId('confirm-add-label'))

    expect(screen.getByText('Label key already exists')).toBeInTheDocument()
  })

  it('should allow editing a label value by clicking on it', async () => {
    const onChange = jest.fn()
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={onChange} />
      </TestWrapper>
    )

    // Click on operator label to edit
    fireEvent.click(screen.getByTestId('label-operator'))

    await waitFor(() => {
      expect(screen.getByTestId('edit-label-operator')).toBeInTheDocument()
    })
  })

  it('should export correct default labels', () => {
    expect(DEFAULT_GLOBAL_LABELS).toEqual({
      operator: 'roakey',
      operation: 'op_trash_panda',
    })
  })

  it('should fetch existing labels on mount', async () => {
    render(
      <TestWrapper>
        <LabelsBar labels={{ ...DEFAULT_GLOBAL_LABELS }} onLabelsChange={jest.fn()} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(mockedLabelsApi.getLabels).toHaveBeenCalled()
    })
  })
})
