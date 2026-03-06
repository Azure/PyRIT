import { render, screen, fireEvent } from '@testing-library/react'
import { ErrorBoundary } from './ErrorBoundary'

// Component that throws on render
function ThrowingChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error('Test crash')
  }
  return <div data-testid="child-content">OK</div>
}

describe('ErrorBoundary', () => {
  let consoleSpy: jest.SpyInstance

  beforeEach(() => {
    // Suppress React's console.error for error boundary logging
    consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    consoleSpy.mockRestore()
  })

  it('renders children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={false} />
      </ErrorBoundary>
    )

    expect(screen.getByTestId('child-content')).toBeInTheDocument()
    expect(screen.queryByTestId('error-boundary-fallback')).toBeNull()
  })

  it('catches render error and shows fallback', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument()
    expect(screen.getByText(/test crash/i)).toBeInTheDocument()
    expect(screen.getByText('Try again')).toBeInTheDocument()
  })

  it('"Try again" resets error state and re-renders children', () => {
    // We use a stateful wrapper to toggle the throw
    let shouldThrow = true

    function ToggleChild() {
      if (shouldThrow) throw new Error('First crash')
      return <div data-testid="child-content">OK</div>
    }

    render(
      <ErrorBoundary>
        <ToggleChild />
      </ErrorBoundary>
    )

    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument()

    // Stop throwing and click "Try again"
    shouldThrow = false
    fireEvent.click(screen.getByText('Try again'))

    expect(screen.getByTestId('child-content')).toBeInTheDocument()
    expect(screen.queryByTestId('error-boundary-fallback')).toBeNull()
  })

  it('shows "Reload page" after repeated crash', () => {
    let crashCount = 0

    function CrashOnRender() {
      crashCount++
      // Always throw — simulates repeated crash
      throw new Error(`Crash #${crashCount}`)
    }

    render(
      <ErrorBoundary>
        <CrashOnRender />
      </ErrorBoundary>
    )

    // First crash: only "Try again" visible, no "Reload page"
    expect(screen.getByText('Try again')).toBeInTheDocument()
    expect(screen.queryByText('Reload page')).toBeNull()

    // Click "Try again" → crashes again → now shows "Reload page"
    fireEvent.click(screen.getByText('Try again'))

    expect(screen.getByText('Try again')).toBeInTheDocument()
    expect(screen.getByText('Reload page')).toBeInTheDocument()
  })

  it('logs error to console', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(consoleSpy).toHaveBeenCalledWith(
      '[ErrorBoundary] Caught render error:',
      expect.any(Error),
      expect.any(String)
    )
  })
})
