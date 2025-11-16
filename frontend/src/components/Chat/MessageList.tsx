import { useEffect, useRef } from 'react'
import {
  makeStyles,
  Text,
  Avatar,
  tokens,
} from '@fluentui/react-components'
import { Message } from '../../types'

const useStyles = makeStyles({
  root: {
    flex: 1,
    overflowY: 'auto',
    padding: tokens.spacingVerticalXXL,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  message: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    maxWidth: '800px',
    alignSelf: 'flex-start',
  },
  userMessage: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse',
  },
  messageContent: {
    backgroundColor: tokens.colorNeutralBackground3,
    padding: tokens.spacingVerticalM,
    borderRadius: tokens.borderRadiusMedium,
    flex: 1,
  },
  userMessageContent: {
    backgroundColor: tokens.colorBrandBackground,
    color: tokens.colorNeutralForegroundOnBrand,
  },
  messageText: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  timestamp: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalXS,
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    gap: tokens.spacingVerticalM,
  },
})

interface MessageListProps {
  messages: Message[]
}

export default function MessageList({ messages }: MessageListProps) {
  const styles = useStyles()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className={styles.emptyState}>
        <Text size={500} weight="semibold">Welcome to PyRIT</Text>
        <Text size={300} style={{ color: tokens.colorNeutralForeground3 }}>
          Start a conversation to test AI safety and robustness
        </Text>
      </div>
    )
  }

  return (
    <div className={styles.root}>
      {messages.map((message, index) => {
        const isUser = message.role === 'user'
        const timestamp = new Date(message.timestamp).toLocaleTimeString()

        return (
          <div
            key={index}
            className={`${styles.message} ${isUser ? styles.userMessage : ''}`}
          >
            <Avatar
              name={isUser ? 'User' : 'Assistant'}
              color={isUser ? 'colorful' : 'brand'}
            />
            <div className={`${styles.messageContent} ${isUser ? styles.userMessageContent : ''}`}>
              <Text className={styles.messageText}>{message.content}</Text>
              <Text className={styles.timestamp} block>{timestamp}</Text>
            </div>
          </div>
        )
      })}
      <div ref={messagesEndRef} />
    </div>
  )
}
