const DEFAULT_PRIMARY_CHARS = [
  'a',
  'c',
  'd',
  'e',
  'f',
  'i',
  'j',
  'k',
  'l',
  'm',
  'n',
  'o',
  'p',
  'q',
  'r',
  's',
  'v',
  'w',
  'x',
  'z',
]

const ALL_ALLOWED_CHARS = [
  ...DEFAULT_PRIMARY_CHARS,
  'b',
  'g',
  'h',
  't',
  'u',
  'y',
  '1',
  '2',
  '3',
  '4',
  '9',
  '0',
  '5',
  '6',
  '7',
  '8',
]

export interface CodeSet {
  short: Array<string>
  long: Array<string>
}

function combineElements(arrA: string[], arrB: string[], callback: (s: string) => void): void {
  const results: { text: string; i: number; j: number; t: number }[] = []

  for (let i = 0; i < arrA.length; ++i) {
    for (let j = 0; j < arrB.length; ++j) {
      results.push({
        i,
        j,
        text: arrA[i] + arrB[j],
        t: i + Math.floor(j / 8) * 8,
      })
    }
  }

  results
    .sort((a, b) => {
      if (a.t !== b.t) {
        return a.t - b.t
      }
      if (a.i !== b.i) {
        return a.i - b.i
      }
      return a.j - b.j
    })
    .forEach(elem => callback(elem.text))
}

export function createCharCodeSet(primaryCharacters = DEFAULT_PRIMARY_CHARS): CodeSet {
  const primaryChars = primaryCharacters.filter(char => ALL_ALLOWED_CHARS.includes(char))
  const secondaryChars = ALL_ALLOWED_CHARS.filter(char => !primaryChars.includes(char))

  const codeSet: CodeSet = { short: [], long: [] }
  const callback = (str: string): void => {
    codeSet.long.push(str)
  }

  primaryChars.forEach((str: string) => codeSet.short.push(str))

  combineElements(primaryChars, primaryChars, callback)
  combineElements(primaryChars, secondaryChars, callback)
  combineElements(secondaryChars, secondaryChars, callback)

  return codeSet
}
