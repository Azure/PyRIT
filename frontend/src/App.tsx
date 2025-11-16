import { useState } from 'react'
import MainLayout from './components/Layout/MainLayout'
import ChatWindow from './components/Chat/ChatWindow'
import { Message } from './types'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)

  const handleSendMessage = (message: Message) => {
    setMessages([...messages, message])
  }

  const handleReceiveMessage = (message: Message, convId: string) => {
    setMessages([...messages, message])
    setConversationId(convId)
  }

  return (
    <MainLayout>
      <ChatWindow
        messages={messages}
        conversationId={conversationId}
        onSendMessage={handleSendMessage}
        onReceiveMessage={handleReceiveMessage}
      />
    </MainLayout>
  )
}

export default App
