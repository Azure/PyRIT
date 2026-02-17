import { useEffect, useRef } from 'react'
import {
  makeStyles,
  Text,
  Avatar,
  tokens,
  MessageBar,
  MessageBarBody,
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
  loadingEllipsis: {
    fontSize: tokens.fontSizeBase500,
    animationName: {
      '0%': { opacity: '0.3' },
      '50%': { opacity: '1' },
      '100%': { opacity: '0.3' },
    },
    animationDuration: '1.5s',
    animationIterationCount: 'infinite',
  },
  attachmentsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
    marginTop: tokens.spacingVerticalS,
  },
  attachmentPreview: {
    maxWidth: '400px',
    maxHeight: '400px',
    borderRadius: tokens.borderRadiusMedium,
    objectFit: 'contain',
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  videoPreview: {
    maxWidth: '400px',
    maxHeight: '300px',
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  attachmentFile: {
    padding: tokens.spacingVerticalS,
    backgroundColor: tokens.colorNeutralBackground1,
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    gap: tokens.spacingVerticalM,
  },
  errorContainer: {
    marginTop: tokens.spacingVerticalS,
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
        const isSimulated = message.role === 'simulated_assistant'
        const timestamp = new Date(message.timestamp).toLocaleTimeString()
        const avatarName = isUser ? 'User' : isSimulated ? 'Simulated' : 'Assistant'

        return (
          <div
            key={index}
            className={`${styles.message} ${isUser ? styles.userMessage : ''}`}
          >
            <Avatar
              name={avatarName}
              color={isUser ? 'colorful' : isSimulated ? 'steel' : 'brand'}
            />
            <div className={`${styles.messageContent} ${isUser ? styles.userMessageContent : ''}`}>
              {/* Error rendering */}
              {message.error && (
                <div className={styles.errorContainer}>
                  <MessageBar intent="error">
                    <MessageBarBody>
                      <Text weight="semibold">{message.error.type}</Text>
                      {message.error.description && (
                        <Text>: {message.error.description}</Text>
                      )}
                    </MessageBarBody>
                  </MessageBar>
                </div>
              )}

              {/* Text content */}
              {message.content && (
                <Text className={message.isLoading ? styles.loadingEllipsis : styles.messageText}>
                  {message.content}
                </Text>
              )}

              {/* Attachments (images, audio, video, files) */}
              {message.attachments && message.attachments.length > 0 && (
                <div className={styles.attachmentsContainer}>
                  {message.attachments.map((att, attIndex) => (
                    <div key={attIndex}>
                      {att.type === 'image' && (
                        <img
                          src={att.url}
                          alt={att.name}
                          className={styles.attachmentPreview}
                        />
                      )}
                      {att.type === 'video' && (
                        <video
                          src={att.url}
                          controls
                          className={styles.videoPreview}
                        />
                      )}
                      {att.type === 'audio' && (
                        <audio src={att.url} controls />
                      )}
                      {att.type === 'file' && (
                        <div className={styles.attachmentFile}>
                          <Text size={200}>📄 {att.name}</Text>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
              <Text className={styles.timestamp} block>{timestamp}</Text>
            </div>
          </div>
        )
      })}
      <div ref={messagesEndRef} />
    </div>
  )
}
