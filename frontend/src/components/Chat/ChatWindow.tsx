import { useState } from 'react'
import {
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import MessageList from './MessageList'
import InputBox from './InputBox'
import { Message, MessageAttachment } from '../../types'
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

  const handleSend = async (text: string, attachments: MessageAttachment[]) => {
    // Add user message with attachments for display
    const userMessage: Message = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    onSendMessage(userMessage)

    // Add temporary loading message
    const loadingMessage: Message = {
      role: 'assistant',
      content: '...',  // Will be styled as animated ellipsis
      timestamp: new Date().toISOString(),
    }
    onSendMessage(loadingMessage)

    // Send to API
    setIsSending(true)
    try {
      const response = await chatApi.sendMessage({
        message: text,
        conversation_id: conversationId || undefined,
        attachments: attachments,
      })

      // Remove loading message and add real response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      }
      // This will replace the loading message since we're using the callback
      onReceiveMessage(assistantMessage, response.conversation_id)
    } catch (error) {
      console.error('Failed to send message:', error)
      // Replace loading message with detailed error
      const errorDetails = error instanceof Error ? error.message : String(error)
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Error: ${errorDetails}\n\nPlease check the browser console for more details.`,
        timestamp: new Date().toISOString(),
      }
      onReceiveMessage(errorMessage, conversationId || '')
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
