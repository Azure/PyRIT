import { useState } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import HistoryPage from './components/History/HistoryPage'
import { Message } from './types'
import { chatApi } from './services/api'

type View = 'chat' | 'history'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [currentView, setCurrentView] = useState<View>('chat')

  const handleSendMessage = (message: Message) => {
    setMessages(prev => [...prev, message])
  }

  const handleReceiveMessage = (message: Message, convId: string) => {
    setMessages(prev => {
      // Remove loading message (animated ellipsis) if it exists
      const filtered = prev.filter(m => m.content !== '...')
      return [...filtered, message]
    })
    setConversationId(convId)
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
    setCurrentView('chat')
  }

  const handleReturnToChat = () => {
    setCurrentView('chat')
  }

  const handleShowHistory = () => {
    setCurrentView('history')
  }

  const handleSelectConversation = async (convId: string) => {
    try {
      const conversation = await chatApi.getConversation(convId)
      // Convert backend messages to frontend Message format
      const loadedMessages: Message[] = conversation.messages.map(msg => ({
        role: msg.role as 'user' | 'assistant' | 'system',
        content: msg.content,
        timestamp: msg.timestamp,
      }))
      setMessages(loadedMessages)
      setConversationId(convId)
      setCurrentView('chat')
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
      <MainLayout 
        onToggleTheme={toggleTheme} 
        isDarkMode={isDarkMode} 
        onReturnToChat={handleReturnToChat}
        onShowHistory={handleShowHistory}
        currentView={currentView}
      >
        {currentView === 'chat' ? (
          <ChatWindow
            messages={messages}
            conversationId={conversationId}
            onSendMessage={handleSendMessage}
            onReceiveMessage={handleReceiveMessage}
            onNewChat={handleNewChat}
          />
        ) : (
          <HistoryPage onSelectConversation={handleSelectConversation} />
        )}
      </MainLayout>
    </FluentProvider>
  )
}

export default App
