import { useState } from 'react'
import {
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import MessageList from './MessageList'
import InputBox from './InputBox'
import { Message } from '../../types'
import { chatApi } from '../../services/api'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    backgroundColor: tokens.colorNeutralBackground1,
  },
})

interface ChatWindowProps {
  messages: Message[]
  conversationId: string | null
  onSendMessage: (message: Message) => void
  onReceiveMessage: (message: Message, conversationId: string) => void
}

export default function ChatWindow({
  messages,
  conversationId,
  onSendMessage,
  onReceiveMessage,
}: ChatWindowProps) {
  const styles = useStyles()
  const [isSending, setIsSending] = useState(false)

  const handleSend = async (text: string) => {
    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }
    onSendMessage(userMessage)

    // Send to API
    setIsSending(true)
    try {
      const response = await chatApi.sendMessage({
        message: text,
        conversation_id: conversationId || undefined,
      })

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      }
      onReceiveMessage(assistantMessage, response.conversation_id)
    } catch (error) {
      console.error('Failed to send message:', error)
      // Optionally show error message to user
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className={styles.root}>
      <MessageList messages={messages} />
      <InputBox onSend={handleSend} disabled={isSending} />
    </div>
  )
}
