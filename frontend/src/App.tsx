import { useState, useCallback } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import TargetConfig from './components/Config/TargetConfig'
import AttackHistory from './components/History/AttackHistory'
import { ConnectionBanner } from './components/ConnectionBanner'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ConnectionHealthProvider } from './hooks/useConnectionHealth'
import { DEFAULT_GLOBAL_LABELS } from './components/Labels/LabelsBar'
import type { ViewName } from './components/Sidebar/Navigation'
import type { Message, TargetInstance, TargetInfo } from './types'
import { attacksApi } from './services/api'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [currentView, setCurrentView] = useState<ViewName>('chat')
  const [activeTarget, setActiveTarget] = useState<TargetInstance | null>(null)
  const [globalLabels, setGlobalLabels] = useState<Record<string, string>>({ ...DEFAULT_GLOBAL_LABELS })
  /** True while loading a historical attack from the history view */
  const [isLoadingAttack, setIsLoadingAttack] = useState(false)

  // When the user switches to a genuinely different target, start a fresh attack.
  // Just re-selecting the same target (or viewing config without changing) keeps
  // the current conversation intact so the user can branch from it.
  const handleSetActiveTarget = useCallback((target: TargetInstance) => {
    setActiveTarget(prev => {
      const isSame = prev &&
        prev.target_registry_name === target.target_registry_name &&
        prev.target_type === target.target_type &&
        (prev.endpoint ?? '') === (target.endpoint ?? '') &&
        (prev.model_name ?? '') === (target.model_name ?? '')
      if (isSame) return prev
      // Switching targets no longer clears the loaded attack.  The cross-target
      // guard in ChatWindow prevents sending to a mismatched target, and the
      // backend enforces this server-side as well.  Clearing state here was
      // confusing because navigating to config to pick the *correct* target
      // would wipe the conversation the user was trying to continue.
      return target
    })
  }, [])
  /** The AttackResult's primary key (set on first message). */
  const [attackResultId, setAttackResultId] = useState<string | null>(null)
  /** The attack's primary conversation_id (set on first message). */
  const [conversationId, setConversationId] = useState<string | null>(null)
  /** The currently active conversation (may be main or a related conversation). */
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  /** Labels that the currently loaded attack was created with (for operator locking). */
  const [attackLabels, setAttackLabels] = useState<Record<string, string> | null>(null)
  /** Target info from the currently loaded historical attack (for cross-target guard). */
  const [attackTarget, setAttackTarget] = useState<TargetInfo | null>(null)
  /** Number of related conversations for the currently loaded attack. */
  const [relatedConversationCount, setRelatedConversationCount] = useState(0)

  const handleSendMessage = (message: Message) => {
    setMessages(prev => [...prev, message])
  }

  const handleReceiveMessage = (message: Message) => {
    setMessages(prev => {
      // If the last message is a loading indicator, replace it
      if (prev.length > 0 && prev[prev.length - 1].isLoading) {
        return [...prev.slice(0, -1), message]
      }
      return [...prev, message]
    })
  }

  const handleNewAttack = () => {
    setMessages([])
    setAttackResultId(null)
    setConversationId(null)
    setActiveConversationId(null)
    setAttackLabels(null)
    setAttackTarget(null)
    setRelatedConversationCount(0)
  }

  const handleConversationCreated = useCallback((arId: string, convId: string) => {
    setAttackResultId(arId)
    setConversationId(convId)
    setActiveConversationId(convId)
    // New attack was created by the current user — use their global labels
    setAttackLabels(null)
    setAttackTarget(null)
  }, [])

  const handleSelectConversation = useCallback((convId: string) => {
    setActiveConversationId(convId)
    // Messages will be loaded by ChatWindow's useEffect
  }, [])

  const handleOpenAttack = useCallback(async (openAttackResultId: string) => {
    setMessages([])
    setAttackResultId(openAttackResultId)
    setIsLoadingAttack(true)
    setCurrentView('chat')
    // Fetch attack info to get conversation_id and stored labels (for operator locking)
    try {
      const attack = await attacksApi.getAttack(openAttackResultId)
      setConversationId(attack.conversation_id)
      setActiveConversationId(attack.conversation_id)
      setAttackLabels(attack.labels ?? {})
      setAttackTarget(attack.target ?? null)
      setRelatedConversationCount(attack.related_conversation_ids?.length ?? 0)
    } catch {
      setConversationId(null)
      setActiveConversationId(null)
      setAttackLabels(null)
      setAttackTarget(null)
      setRelatedConversationCount(0)
    } finally {
      setIsLoadingAttack(false)
    }
  }, [])

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <ErrorBoundary>
      <ConnectionHealthProvider>
        <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
          <ConnectionBanner />
          <MainLayout
            currentView={currentView}
            onNavigate={setCurrentView}
            onToggleTheme={toggleTheme}
            isDarkMode={isDarkMode}
          >
            {currentView === 'chat' && (
              <ChatWindow
                messages={messages}
                onSendMessage={handleSendMessage}
                onReceiveMessage={handleReceiveMessage}
                onNewAttack={handleNewAttack}
                activeTarget={activeTarget}
                attackResultId={attackResultId}
                conversationId={conversationId}
                activeConversationId={activeConversationId}
                onConversationCreated={handleConversationCreated}
                onSelectConversation={handleSelectConversation}
                onSetMessages={setMessages}
                labels={globalLabels}
                onLabelsChange={setGlobalLabels}
                onNavigate={setCurrentView}
                attackLabels={attackLabels}
                attackTarget={attackTarget}
                isLoadingAttack={isLoadingAttack}
                relatedConversationCount={relatedConversationCount}
              />
            )}
            {currentView === 'config' && (
              <TargetConfig
                activeTarget={activeTarget}
                onSetActiveTarget={handleSetActiveTarget}
              />
            )}
            {currentView === 'history' && (
              <AttackHistory onOpenAttack={handleOpenAttack} />
            )}
          </MainLayout>
        </FluentProvider>
      </ConnectionHealthProvider>
    </ErrorBoundary>
  )
}

export default App
