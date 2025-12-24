import { useState } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import { Message } from './types'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isDarkMode, setIsDarkMode] = useState(true)

  const handleSendMessage = (message: Message) => {
    setMessages(prev => [...prev, message])
  }

  const handleReceiveMessage = (message: Message) => {
    setMessages(prev => [...prev, message])
  }

  const handleNewChat = () => {
    setMessages([])
  }

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
      <MainLayout 
        onToggleTheme={toggleTheme} 
        isDarkMode={isDarkMode}
      >
        <ChatWindow
          messages={messages}
          onSendMessage={handleSendMessage}
          onReceiveMessage={handleReceiveMessage}
          onNewChat={handleNewChat}
        />
      </MainLayout>
    </FluentProvider>
  )
}

export default App
