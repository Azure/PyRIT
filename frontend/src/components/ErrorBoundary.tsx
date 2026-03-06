import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Button, MessageBar, MessageBarBody, tokens } from '@fluentui/react-components'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  crashCount: number
}

/**
 * React Error Boundary that catches render errors in its subtree.
 *
 * - On first crash: shows "Try again" which resets the error state.
 * - If the app crashes again after "Try again": shows "Reload page" as a
 *   nuclear option (full page reload).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, crashCount: 0 }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary] Caught render error:', error, info.componentStack)
    this.setState((prev) => ({ crashCount: prev.crashCount + 1 }))
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    const showReload = this.state.crashCount > 1

    return (
      <div
        data-testid="error-boundary-fallback"
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          gap: tokens.spacingVerticalL,
          padding: tokens.spacingHorizontalXXL,
        }}
      >
        <MessageBar intent="error">
          <MessageBarBody>
            Something went wrong: {this.state.error?.message || 'Unknown error'}
          </MessageBarBody>
        </MessageBar>

        <div style={{ display: 'flex', gap: tokens.spacingHorizontalM }}>
          <Button appearance="primary" onClick={this.handleReset}>
            Try again
          </Button>
          {showReload && (
            <Button appearance="secondary" onClick={this.handleReload}>
              Reload page
            </Button>
          )}
        </div>
      </div>
    )
  }
}
