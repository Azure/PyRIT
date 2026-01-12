import { useState } from 'react'
import {
  makeStyles,
  tokens,
  Button,
  Text,
} from '@fluentui/react-components'
import { AddRegular } from '@fluentui/react-icons'
import MessageList from './MessageList'
import InputBox from './InputBox'
import { Message, MessageAttachment } from '../../types'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    width: '100%',
    backgroundColor: tokens.colorNeutralBackground2,
    overflow: 'hidden',
  },
  ribbon: {
    height: '48px',
    minHeight: '48px',
    flexShrink: 0,
    backgroundColor: tokens.colorNeutralBackground3,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: `0 ${tokens.spacingHorizontalL}`,
    gap: tokens.spacingHorizontalM,
  },
  conversationInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase300,
  },
})

interface ChatWindowProps {
  messages: Message[]
  onSendMessage: (message: Message) => void
  onReceiveMessage: (message: Message) => void
  onNewChat: () => void
}

export default function ChatWindow({
  messages,
  onSendMessage,
  onReceiveMessage,
  onNewChat,
}: ChatWindowProps) {
  const styles = useStyles()
  const [isSending, setIsSending] = useState(false)

  const handleSend = async (originalValue: string, _convertedValue: string | undefined, attachments: MessageAttachment[]) => {
    // Add user message with attachments for display
    const userMessage: Message = {
      role: 'user',
      content: originalValue,
      timestamp: new Date().toISOString(),
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    onSendMessage(userMessage)

    // Simple echo response after a short delay
    setIsSending(true)
    setTimeout(() => {
      const assistantMessage: Message = {
        role: 'assistant',
        content: `Echo: ${originalValue}`,
        timestamp: new Date().toISOString(),
      }
      onReceiveMessage(assistantMessage)
      setIsSending(false)
    }, 500)
  }

  return (
    <div className={styles.root}>
      <div className={styles.ribbon}>
        <div className={styles.conversationInfo}>
          <Text>PyRIT Frontend</Text>
        </div>
        <Button
          appearance="primary"
          icon={<AddRegular />}
          onClick={onNewChat}
        >
          New Chat
        </Button>
      </div>
      <MessageList messages={messages} />
      <InputBox
        onSend={handleSend}
        disabled={isSending}
      />
    </div>
  )
}
