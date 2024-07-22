import * as fs from 'fs'
import { homedir } from 'os'
import { TextEncoder } from 'util'
import {
  commands,
  ConfigurationChangeEvent,
  DecorationOptions,
  Disposable,
  Position,
  Range,
  Selection,
  TextEditor,
  window,
  workspace,
} from 'vscode'
import { getVisibleLines } from './get-lines'
import { Settings } from './settings'
import { ExtensionComponent, Nullable } from './typings'

const enum Command {
  Type = 'type',
  ReplacePreviousChar = 'replacePreviousChar',
  Exit = 'jump-extension.exit',
  Enter = 'jump-extension.primary-jump',
  EnterSearchJump = 'jump-extension.search-jump',
  EnterInlineJump = 'jump-extension.inline-jump',
}

const enum Event {
  ConfigChanged = 'configChanged',
  ActiveEditorChanged = 'activeEditorChanged',
  ActiveSelectionChanged = 'activeSelectionChanged',
  VisibleRangesChanged = 'visibleRangesChanged',
}

interface JumpPosition {
  line: number
  char: number
  match: string
}

interface JumpPositionMap {
  [code: string]: JumpPosition
}

interface StateJumpActive {
  isInJumpMode: true
  isInSearchMode: boolean
  editor: TextEditor
  typedCharacters: string
  current_regex_index: number
  decorations: Map<string, DecorationOptions>
  move_cursor_timer: NodeJS.Timeout | undefined
}

interface StateJumpInactive {
  isInJumpMode: false
  isInSearchMode: false
  editor: undefined
  typedCharacters: string
  current_regex_index: number
  decorations: undefined
  move_cursor_timer: undefined
}

type State = StateJumpActive | StateJumpInactive

const HANDLE_NAMES = [
  Command.Type,
  Command.ReplacePreviousChar,
  Command.Exit,
  Command.Enter,
  Event.ConfigChanged,
  Event.ActiveEditorChanged,
  Event.ActiveSelectionChanged,
  Event.VisibleRangesChanged,
] as const
const DEFAULT_STATE: State = {
  isInJumpMode: false,
  isInSearchMode: false,
  editor: undefined,
  typedCharacters: '',
  current_regex_index: 0,
  decorations: undefined,
  move_cursor_timer: undefined,
}
const TYPE_REGEX = /\w/
const TYPE_LOG_FILE_PATH = `${homedir()}/.jumpa_type_log`

interface TypeLogEntry {
  filename: string
  line_txt: string
  line_number: number
  char_index: number
}

export class Jump implements ExtensionComponent {
  private handles: Record<Command | Event, Nullable<Disposable>>
  private settings: Settings
  private positions: JumpPositionMap
  private state: State
  private timeout?: NodeJS.Timeout
  private last_type_log_entry: TypeLogEntry | undefined
  private type_log_file = -1

  public constructor() {
    this.settings = new Settings()

    this.state = DEFAULT_STATE
    this.handles = {
      [Command.Type]: null,
      [Command.ReplacePreviousChar]: null,
      [Command.Exit]: null,
      [Command.Enter]: null,
      [Command.EnterSearchJump]: null,
      [Command.EnterInlineJump]: null,
      [Event.ConfigChanged]: null,
      [Event.ActiveEditorChanged]: null,
      [Event.ActiveSelectionChanged]: null,
      [Event.VisibleRangesChanged]: null,
    }
    this.positions = {}

    fs.open(TYPE_LOG_FILE_PATH, 'a', (err, fd) => {
      this.type_log_file = fd
    })
  }

  public activate(): void {
    this.settings.activate()

    this.handles[Command.Enter] = commands.registerCommand(Command.Enter, () =>
      this.handleEnterJumpMode(),
    )

    this.handles[Command.EnterSearchJump] = commands.registerCommand(Command.EnterSearchJump, () =>
      this.handleEnterSearchJumpMode(),
    )
    this.handles[Command.EnterInlineJump] = commands.registerCommand(Command.EnterInlineJump, () =>
      this.handleEnterInlineJumpMode(),
    )
    this.handles[Command.Exit] = commands.registerCommand(Command.Exit, () =>
      this.handleExitJumpMode(),
    )
    this.handles[Event.ConfigChanged] = workspace.onDidChangeConfiguration(this.handleConfigChange)
    this.handles[Event.ActiveSelectionChanged] = window.onDidChangeTextEditorSelection(
      this.handleSelectionChange,
    )
    this.handles[Event.ActiveEditorChanged] = window.onDidChangeActiveTextEditor(
      this.handleEditorChange,
    )
    this.handles[Event.VisibleRangesChanged] = window.onDidChangeTextEditorVisibleRanges(
      this.handleVisibleRangesChange,
    )
  }

  public deactivate(): void {
    this.handleExitJumpMode()
    this.settings.deactivate()

    for (const handleName of HANDLE_NAMES) {
      this.tryDispose(handleName)
    }
  }

  private handleConfigChange = (event: ConfigurationChangeEvent): void => {
    if (this.state.isInJumpMode) {
      this.state.decorations = new Map()
      this.setDecorations(this.state.editor)
      this.settings.handleConfigurationChange(event)
      this.makeJumpAnchors()
    } else {
      this.settings.handleConfigurationChange(event)
    }
  }

  private handleVisibleRangesChange = (): void => {
    if (!this.state.isInJumpMode) {
      return
    }

    this.timeout && clearTimeout(this.timeout)

    this.timeout = setTimeout(() => this.makeJumpAnchors(), 300)
  }

  private handleSelectionChange = (): void => {
    if (!this.state.isInJumpMode || this.state.isInSearchMode) {
      return
    }

    this.makeJumpAnchors()
  }

  private handleEditorChange = (editor: TextEditor | undefined): void => {
    if (!this.state.isInJumpMode) {
      return
    }

    if (editor === undefined) {
      this.handleExitJumpMode()
    } else if (!this.state.isInSearchMode) {
      this.state.decorations = new Map()
      this.setDecorations(this.state.editor)
      this.state.editor = editor
      this.makeJumpAnchors()
    }
  }

  private tryDispose(handleName: Command | Event): void {
    const handle = this.handles[handleName]
    if (handle) {
      handle.dispose()
      this.handles[handleName] = null
    }
  }

  private handleEnterJumpMode = (): void => {
    if (this.state.isInJumpMode) {
      const next_regex_index =
        (this.state.current_regex_index + 1) % this.settings.primaryRegexes.length
      this.handleExitJumpMode()
      this.state.current_regex_index = next_regex_index
    }

    const activeEditor = window.activeTextEditor
    if (activeEditor === undefined) {
      return
    }

    this.setJumpContext(true)

    this.tryDispose(Command.Type)
    this.handles[Command.Type] = commands.registerCommand(Command.Type, this.handleTypeEvent)
    this.handles[Command.ReplacePreviousChar] = commands.registerCommand(
      Command.ReplacePreviousChar,
      () => {},
    )

    this.state.editor = activeEditor

    this.makeJumpAnchors()
  }

  private handleEnterSearchJumpMode = (): void => {
    if (this.state.isInJumpMode) {
      this.handleExitJumpMode()
    }

    const activeEditor = window.activeTextEditor
    if (activeEditor === undefined) {
      return
    }

    this.setJumpContext(true)

    this.tryDispose(Command.Type)
    this.handles[Command.Type] = commands.registerCommand(
      Command.Type,
      this.handleTypeEventSearchMode,
    )
    this.state.isInSearchMode = true

    this.state.editor = activeEditor
  }

  private handleEnterInlineJumpMode = (): void => {

  }

  private clearDecorations = (): void => {
    if (!this.state.isInJumpMode) {
      return
    }
    this.state.decorations = new Map()
    this.setDecorations(this.state.editor)
    this.setSearchDecorations(this.state.editor, [])
  }

  private handleExitJumpMode = (): void => {
    if (!this.state.isInJumpMode) {
      return
    }

    this.state.typedCharacters = ''
    this.clearDecorations()
    this.state = { ...DEFAULT_STATE }

    this.tryDispose(Command.Type)
    this.handles[Command.Type] = commands.registerCommand(Command.Type, this.handleTypeLogEvent)
    this.tryDispose(Command.ReplacePreviousChar)
    this.setJumpContext(false)
  }

  private typedTextIsUsable = (text: string): boolean => {
    for (const [code] of Object.entries(this.positions)) {
      if (code.startsWith(text)) {
        return true
      }
    }
    return false
  }

  private handleTypeEvent = ({ text }: { text: string }): void => {
    // Ignore additional characters if the typed text is not prefix of some code
    if (!this.typedTextIsUsable(this.state.typedCharacters + text.toLowerCase())) {
      return
    }

    if (!TYPE_REGEX.test(text) || !this.state.isInJumpMode) {
      this.state.typedCharacters = ''
      return
    }

    this.state.typedCharacters += text.toLowerCase()

    const code = this.state.typedCharacters
    if (this.positions[code] === undefined) {
      // Remove non-matching decorations
      this.state.decorations = new Map(
        [...this.state.decorations.entries()].filter(([k]) =>
          k.startsWith(this.state.typedCharacters),
        ),
      )
      this.setDecorations(this.state.editor)
      return
    }
    const { line, char } = this.positions[code]

    this.setSelection(line, char, text.toUpperCase() === text)

    this.handleExitJumpMode()
  }

  private setSelection = (line: number, char: number, shifted: boolean): void => {
    if (!this.state.editor) {
      console.log('Error in setSelection')
      return
    }
    if (shifted) {
      const [{ anchor }] = this.state.editor.selections.slice(-1)
      const active = new Position(line, char)
      this.state.editor.selection = new Selection(anchor, active)

      if (line && char && this.settings.cursorSurroundingLines) {
        commands.executeCommand('cursorLeftSelect')
        commands.executeCommand('cursorRightSelect')
      }
    } else {
      this.state.editor.selection = new Selection(line, char, line, char)

      if (line && char && this.settings.cursorSurroundingLines) {
        commands.executeCommand('cursorLeft')
        commands.executeCommand('cursorRight')
      }
    }
  }

  private handleTypeEventSearchMode = ({ text }: { text: string }): void => {
    this.state.typedCharacters += text.toLowerCase()

    console.log(this.state.move_cursor_timer)
    if (this.state.decorations?.has(text.toLowerCase())) {
      const range = this.state.decorations?.get(text.toLowerCase())?.range
      if (!range) {
        return
      }
      const { line, character } = range.start
      this.setSelection(line, character - 1, text.toUpperCase() === text)

      this.tryDispose(Command.Type)
      this.handles[Command.Type] = commands.registerCommand(
        Command.Type,
        this.handleTypeMovesCursor,
      )

      this.state.move_cursor_timer = setTimeout(() => {
        this.tryDispose(Command.Type)
        this.handles[Command.Type] = commands.registerCommand(Command.Type, this.handleTypeLogEvent)
        this.handleExitJumpMode()
        this.state.move_cursor_timer = undefined
      }, 1000)

      this.clearDecorations()
    } else if (this.state.move_cursor_timer === undefined) {
      const all_positions = this.getMatchPositions(
        new RegExp(`${this.state.typedCharacters}(.)`, this.settings.userRegexFlags),
        64,
      )

      if (all_positions === undefined) {
        return
      }

      this.showSearchDecorations(all_positions)
    }
  }

  private handleTypeMovesCursor = (args: { text: string }): void => {
    if (
      this.state.isInJumpMode &&
      this.state.move_cursor_timer !== undefined &&
      this.state.editor !== undefined
    ) {
      this.state.typedCharacters += args.text
      const ranges = new Array<Range>()
      for (const selection of this.state.editor.selections) {
        ranges.push(
          new Range(
            selection.end.line,
            selection.end.character - this.state.typedCharacters.length + 1,
            selection.end.line,
            selection.end.character + 1,
          ),
        )
      }
      this.setSearchDecorations(this.state.editor, ranges)

      clearTimeout(this.state.move_cursor_timer)
      this.state.move_cursor_timer = setTimeout(() => {
        this.tryDispose(Command.Type)
        this.handles[Command.Type] = commands.registerCommand(Command.Type, this.handleTypeLogEvent)
        this.handleExitJumpMode()
        this.state.move_cursor_timer = undefined
      }, this.settings.jumpCooldown)
      commands.executeCommand('cursorMove', { to: 'right', by: 'character' })
    } else {
      this.handleExitJumpMode()
    }
  }

  private handleTypeLogEvent = (args: { text: string }): void => {
    if (window.activeTextEditor !== undefined) {
      const editor = window.activeTextEditor

      for (const selection of editor.selections) {
        if (
          this.last_type_log_entry !== undefined &&
          this.last_type_log_entry.filename === editor.document.fileName &&
          selection.active.line === this.last_type_log_entry.line_number &&
          selection.active.character >= this.last_type_log_entry.char_index + args.text.length &&
          selection.active.character <= this.last_type_log_entry.char_index + args.text.length + 2
        ) {
          this.last_type_log_entry.char_index = selection.active.character
        } else {
          this.last_type_log_entry = {
            filename: editor.document.fileName,
            line_txt: editor.document.lineAt(selection.active.line).text,
            line_number: selection.active.line,
            char_index: selection.active.character,
          }
          const te = new TextEncoder()
          fs.write(
            this.type_log_file,
            te.encode(
              `${this.last_type_log_entry.filename}:${this.last_type_log_entry.line_number}:${this.last_type_log_entry.char_index}:${this.last_type_log_entry.line_txt}\n`,
            ),
            () => {},
          )
        }
      }
    }
    commands.executeCommand('default:type', args)
  }

  private setJumpContext(value: boolean): void {
    commands.executeCommand('setContext', 'jump.isInJumpMode', value)
    this.state.isInJumpMode = value
  }

  private setDecorations(editor: TextEditor): void {
    console.trace()
    editor.setDecorations(this.settings.decorationType, [
      ...(this.state.decorations?.values() ?? []),
    ])
  }

  private setSearchDecorations(editor: TextEditor, ranges: Range[]): void {
    editor.setDecorations(this.settings.textDecorationType, ranges)
  }

  private getMatchPositions(regex: RegExp, maxPositions: number): JumpPosition[] | undefined {
    const editor = this.state.editor ?? null
    const lines = editor && getVisibleLines(editor)

    if (!editor || lines === null) {
      return undefined
    }
    const linesCount = lines.length
    const all_positions: JumpPosition[] = []
    let positionCount = 0
    for (let i = 0; i < linesCount && positionCount < maxPositions; ++i) {
      const matches = [...lines[i].text.toLowerCase().matchAll(regex)]

      for (const match of matches) {
        if (positionCount >= maxPositions) {
          break
        }
        if (match.index === undefined) {
          continue
        }
        const match_char = match.indices?.findLast(m => m !== undefined)?.[0]
        all_positions.push({
          line: lines[i].lineNumber,
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          //@ts-ignore
          char: match_char ?? match.index,
          match: match[1],
        })
        ++positionCount
      }
    }
    return all_positions
  }

  private makeJumpAnchors(): void {
    const scanRegexp = this.settings.primaryRegexes[this.state.current_regex_index]
    if (scanRegexp === undefined) {
      return
    }

    this.positions = {}
    const maxDecorations = this.settings.codes.long.length

    const all_positions = this.getMatchPositions(scanRegexp, maxDecorations)
    if (all_positions === undefined) {
      return
    }
    this.showDecorations(all_positions)
  }

  private showDecorations(all_positions: JumpPosition[]): void {
    const { editor } = this.state
    if (!editor) {
      return
    }

    const [{ active }] = editor.selections.slice(-1)
    all_positions.sort((a, b) =>
      Math.abs(a.line - active.line) !== Math.abs(b.line - active.line)
        ? Math.abs(a.line - active.line) - Math.abs(b.line - active.line)
        : Math.abs(active.character - (a.char ?? Infinity)) -
          Math.abs(active.character - (b.char ?? Infinity)),
    )
    const codes =
      all_positions.length <= this.settings.codes.short.length
        ? this.settings.codes.short
        : this.settings.codes.long

    this.state.decorations = new Map<string, DecorationOptions>()
    all_positions.forEach((position, index) => {
      const code = codes[index]

      const { line } = position
      const char = position.char + this.settings.charOffset

      this.positions[code] = position
      this.state.decorations?.set(code, {
        range: new Range(line, char, line, char),
        renderOptions: this.settings.getOptions(this.state.current_regex_index, code),
      })
    })

    this.setDecorations(editor)
  }

  private showSearchDecorations(all_positions: JumpPosition[]): void {
    const { editor } = this.state
    if (!editor) {
      return
    }

    const code_counts = all_positions.reduce((map: Map<string, number>, element) => {
      map.set(element.match, (map.get(element.match) ?? 0) + 1)
      return map
    }, new Map<string, number>())

    // We can only use codes that are only the next character for 1 match or for no matches
    const usable_codes = this.settings.codes.short.filter(code => (code_counts.get(code) ?? 0) <= 1)
    if (usable_codes.length < all_positions.length) {
      return
    }

    const excess_codes = this.settings.codes.short.filter(
      code => (code_counts.get(code) ?? 0) === 0,
    )

    const text_match_ranges = new Array<Range>()

    this.state.decorations = new Map<string, DecorationOptions>()
    let excess_codes_index = 0
    all_positions.forEach(position => {
      const code =
        (code_counts.get(position.match) ?? 0) === 1
          ? position.match
          : excess_codes[excess_codes_index++]

      const { line } = position
      const char = position.char + this.settings.charOffset

      this.positions[code] = position
      this.state.decorations?.set(code, {
        range: new Range(line, char, line, char),
        renderOptions: this.settings.getOptions(this.state.current_regex_index, code),
      })

      const start_char = position.char - this.state.typedCharacters.length
      text_match_ranges.push(new Range(line, start_char, line, position.char))
    })

    this.setDecorations(editor)
    this.setSearchDecorations(editor, text_match_ranges)
  }
}
