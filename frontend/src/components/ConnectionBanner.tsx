import { useEffect, useState } from 'react'
import { MessageBar, MessageBarBody } from '@fluentui/react-components'
import { useConnectionHealth } from '../hooks/useConnectionHealth'
import type { ConnectionStatus } from '../hooks/useConnectionHealth'

const intentMap: Record<ConnectionStatus, 'error' | 'warning' | 'success'> = {
  disconnected: 'error',
  degraded: 'warning',
  connected: 'success',
}

const messageMap: Record<ConnectionStatus, string> = {
  disconnected: 'Unable to reach the PyRIT backend. Check that the server is running. Retrying…',
  degraded: 'Connection to the backend is unstable.',
  connected: 'Reconnected to the backend.',
}

const AUTO_DISMISS_MS = 5_000

export function ConnectionBanner() {
  const { status, reconnectCount } = useConnectionHealth()
  const [showReconnected, setShowReconnected] = useState(false)

  // Show "Reconnected" briefly after recovery
  useEffect(() => {
    if (reconnectCount > 0) {
      setShowReconnected(true)
      const timer = setTimeout(() => setShowReconnected(false), AUTO_DISMISS_MS)
      return () => clearTimeout(timer)
    }
  }, [reconnectCount])

  // Nothing to show when connected and no recent reconnection
  if (status === 'connected' && !showReconnected) {
    return null
  }

  const displayStatus = status === 'connected' && showReconnected ? 'connected' : status

  return (
    <div data-testid="connection-banner" style={{ position: 'sticky', top: 0, zIndex: 1000 }}>
      <MessageBar intent={intentMap[displayStatus]}>
        <MessageBarBody>{messageMap[displayStatus]}</MessageBarBody>
      </MessageBar>
    </div>
  )
}
