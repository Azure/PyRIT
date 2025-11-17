// Client-side converter implementations for preview
// Only non-LLM converters that can run in the browser

export const clientSideConverters: Record<string, (text: string, config: any) => string> = {
  Base64Converter: (text: string) => {
    return btoa(text)
  },
  
  CaesarConverter: (text: string, config: any) => {
    const shift = config.shift || 3
    return text.split('').map(char => {
      if (char.match(/[a-z]/i)) {
        const code = char.charCodeAt(0)
        const base = code >= 65 && code <= 90 ? 65 : 97
        return String.fromCharCode(((code - base + shift) % 26) + base)
      }
      return char
    }).join('')
  },
  
  LeetspeakConverter: (text: string) => {
    const leetMap: Record<string, string> = {
      'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7',
      'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5', 'T': '7'
    }
    return text.split('').map(char => leetMap[char] || char).join('')
  },
  
  BinaryConverter: (text: string) => {
    return text.split('').map(char => 
      char.charCodeAt(0).toString(2).padStart(8, '0')
    ).join(' ')
  },
  
  MorseConverter: (text: string) => {
    const morseMap: Record<string, string> = {
      'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.',
      'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
      'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.',
      's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-',
      'y': '-.--', 'z': '--..', '0': '-----', '1': '.----', '2': '..---',
      '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
      '8': '---..', '9': '----.', ' ': '/'
    }
    return text.toLowerCase().split('').map(char => morseMap[char] || char).join(' ')
  },
  
  AtbashConverter: (text: string) => {
    return text.split('').map(char => {
      if (char.match(/[a-z]/i)) {
        const code = char.charCodeAt(0)
        const isUpper = code >= 65 && code <= 90
        const base = isUpper ? 65 : 97
        return String.fromCharCode(base + (25 - (code - base)))
      }
      return char
    }).join('')
  },
  
  FlipConverter: (text: string) => {
    return text.split('').reverse().join('')
  },
  
  CharacterSpaceConverter: (text: string) => {
    return text.split('').join(' ')
  },
}

export function canConvertClientSide(className: string): boolean {
  return className in clientSideConverters
}

export function convertClientSide(text: string, converters: Array<{class_name: string, config: any}>): string {
  let result = text
  
  for (const converter of converters) {
    const converterFn = clientSideConverters[converter.class_name]
    if (converterFn) {
      result = converterFn(result, converter.config)
    }
  }
  
  return result
}
