import { useState } from 'react'
import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components'
import MainLayout from './components/Layout/MainLayout'
import PromptBuilderPage from './components/Builder/PromptBuilderPage'

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false)

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
      <MainLayout
        onToggleTheme={toggleTheme}
        isDarkMode={isDarkMode}
      >
        <PromptBuilderPage />
      </MainLayout>
    </FluentProvider>
  )
}

export default App
