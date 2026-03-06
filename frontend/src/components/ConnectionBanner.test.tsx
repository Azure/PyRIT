import { render, screen, act } from '@testing-library/react'
import { ConnectionBanner } from './ConnectionBanner'
import * as hookModule from '../hooks/useConnectionHealth'

// Mock the useConnectionHealth hook
jest.mock('../hooks/useConnectionHealth', () => ({
  useConnectionHealth: jest.fn(),
}))

const mockUseConnectionHealth = hookModule.useConnectionHealth as jest.Mock

describe('ConnectionBanner', () => {
  beforeEach(() => {
    jest.useFakeTimers()
    jest.clearAllMocks()
  })

  afterEach(() => {
    jest.useRealTimers()
  })

  it('renders nothing when connected with no reconnection', () => {
    mockUseConnectionHealth.mockReturnValue({ status: 'connected', reconnectCount: 0 })

    const { container } = render(<ConnectionBanner />)

    expect(container.querySelector('[data-testid="connection-banner"]')).toBeNull()
  })

  it('renders warning banner when degraded', () => {
    mockUseConnectionHealth.mockReturnValue({ status: 'degraded', reconnectCount: 0 })

    render(<ConnectionBanner />)

    const banner = screen.getByTestId('connection-banner')
    expect(banner).toBeInTheDocument()
    expect(screen.getByText(/unstable/i)).toBeInTheDocument()
  })

  it('renders error banner when disconnected', () => {
    mockUseConnectionHealth.mockReturnValue({ status: 'disconnected', reconnectCount: 0 })

    render(<ConnectionBanner />)

    const banner = screen.getByTestId('connection-banner')
    expect(banner).toBeInTheDocument()
    expect(screen.getByText(/unable to reach/i)).toBeInTheDocument()
  })

  it('renders success banner on reconnection and auto-dismisses after 5s', () => {
    mockUseConnectionHealth.mockReturnValue({ status: 'connected', reconnectCount: 1 })

    render(<ConnectionBanner />)

    expect(screen.getByTestId('connection-banner')).toBeInTheDocument()
    expect(screen.getByText(/reconnected/i)).toBeInTheDocument()

    // After 5 seconds, the banner should dismiss
    act(() => {
      jest.advanceTimersByTime(5_000)
    })

    expect(screen.queryByTestId('connection-banner')).toBeNull()
  })

  it('shows reconnect banner again on subsequent reconnections', () => {
    // First reconnection
    mockUseConnectionHealth.mockReturnValue({ status: 'connected', reconnectCount: 1 })
    const { rerender } = render(<ConnectionBanner />)
    expect(screen.getByText(/reconnected/i)).toBeInTheDocument()

    // Dismiss
    act(() => {
      jest.advanceTimersByTime(5_000)
    })
    expect(screen.queryByTestId('connection-banner')).toBeNull()

    // Second reconnection
    mockUseConnectionHealth.mockReturnValue({ status: 'connected', reconnectCount: 2 })
    rerender(<ConnectionBanner />)
    expect(screen.getByText(/reconnected/i)).toBeInTheDocument()
  })
})
