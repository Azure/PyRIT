import { useState } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import { Message } from './types'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isDarkMode, setIsDarkMode] = useState(true)

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

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
      <MainLayout onToggleTheme={toggleTheme} isDarkMode={isDarkMode}>
        <ChatWindow
          messages={messages}
          conversationId={conversationId}
          onSendMessage={handleSendMessage}
          onReceiveMessage={handleReceiveMessage}
        />
      </MainLayout>
    </FluentProvider>
  )
}

export default App
