import { useState, KeyboardEvent } from 'react'
import {
  makeStyles,
  Textarea,
  Button,
  tokens,
} from '@fluentui/react-components'
import { SendRegular } from '@fluentui/react-icons'

const useStyles = makeStyles({
  root: {
    padding: tokens.spacingVerticalXL,
    borderTop: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground1,
  },
  inputContainer: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    maxWidth: '1200px',
    margin: '0 auto',
  },
  textarea: {
    flex: 1,
  },
  sendButton: {
    alignSelf: 'flex-end',
  },
})

interface InputBoxProps {
  onSend: (message: string) => void
  disabled?: boolean
}

export default function InputBox({ onSend, disabled = false }: InputBoxProps) {
  const styles = useStyles()
  const [input, setInput] = useState('')

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={styles.root}>
      <div className={styles.inputContainer}>
        <Textarea
          className={styles.textarea}
          placeholder="Type your message... (Shift+Enter for new line)"
          value={input}
          onChange={(_, data) => setInput(data.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          resize="vertical"
          rows={3}
        />
        <Button
          className={styles.sendButton}
          appearance="primary"
          icon={<SendRegular />}
          onClick={handleSend}
          disabled={disabled || !input.trim()}
        >
          Send
        </Button>
      </div>
    </div>
  )
}
