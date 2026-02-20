import { useState } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import TargetConfig from './components/Config/TargetConfig'
import type { ViewName } from './components/Sidebar/Navigation'
import type { Message, TargetInstance } from './types'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [currentView, setCurrentView] = useState<ViewName>('chat')
  const [activeTarget, setActiveTarget] = useState<TargetInstance | null>(null)
  const [conversationId, setConversationId] = useState<string | null>(null)

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

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
  }

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
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
            onNewChat={handleNewChat}
            activeTarget={activeTarget}
            conversationId={conversationId}
            onConversationCreated={setConversationId}
          />
        )}
        {currentView === 'config' && (
          <TargetConfig
            activeTarget={activeTarget}
            onSetActiveTarget={setActiveTarget}
          />
        )}
      </MainLayout>
    </FluentProvider>
  )
}

export default App
