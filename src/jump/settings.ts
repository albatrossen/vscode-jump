import {
  ConfigurationChangeEvent,
  DecorationInstanceRenderOptions,
  TextEditorDecorationType,
  ThemableDecorationAttachmentRenderOptions,
  ThemableDecorationRenderOptions,
  Uri,
  window,
  workspace
} from 'vscode'
import { CodeSet, createCharCodeSet } from './char-codes'
import { ExtensionComponent } from './typings'

export const enum SettingNamespace {
  Editor = 'editor',
  Jump = 'jump',
}

const enum Setting {
  UseIcons = 'useIcons',
  PrimaryRegexes = 'primaryRegexes',
  InlineRegexes = 'inlineRegexes',
  WordRegexpFlags = 'wordRegexpFlags',
  PrimaryCharset = 'primaryCharset',
  FontFamily = 'fontFamily',
  FontSize = 'fontSize',
  CursorSurroundingLines = 'cursorSurroundingLines',
  JumpCooldown = 'jumpCooldown',
}

const enum DisplaySetting {
  Color = 'display.color',
  BackgroundColor = 'display.backgroundColor',
  FontScale = 'display.fontScale',
}

interface DecorationOptions {
  pad: number
  colors: Array<string>
  width: number
  height: number
  fontSize: number
  fontFamily: string
  backgroundColors: Array<string>
  margin?: string
}

// Default values
const DEFAULT_USE_ICONS = true

const DATA_URI = Uri.parse('data:')

export class Settings implements ExtensionComponent {
  private decorationOptions: DecorationOptions
  private textMarkOptions: DecorationOptions
  private codeOptions: Map<number, Map<string, DecorationInstanceRenderOptions>>
  public codes: CodeSet
  public decorationType: TextEditorDecorationType
  public textDecorationType: TextEditorDecorationType
  public primaryRegexes: RegExp[]
  public inlineRegexes: RegExp[]
  public charOffset: number
  public cursorSurroundingLines: number
  public userRegexFlags: string
  public jumpCooldown: number

  public constructor() {
    this.decorationOptions = {} as unknown as DecorationOptions
    this.textMarkOptions = {} as unknown as DecorationOptions
    this.decorationType = window.createTextEditorDecorationType({})
    this.textDecorationType = window.createTextEditorDecorationType({})
    this.codeOptions = new Map()
    this.codes = { short: [], long: [] }
    this.primaryRegexes = []
    this.inlineRegexes = []
    this.charOffset = 0
    this.cursorSurroundingLines = 0

    const jumpConfig = workspace.getConfiguration(SettingNamespace.Jump)
    this.userRegexFlags = jumpConfig[Setting.WordRegexpFlags]
    this.jumpCooldown = jumpConfig[Setting.JumpCooldown]
  }

  public activate(): void {
    this.update()
  }

  public deactivate(): void {
    this.codes = { short: [], long: [] }
    this.codeOptions.clear()
  }

  public getOptions(current_regex_index: number, code: string): DecorationInstanceRenderOptions {
    return this.codeOptions.get(current_regex_index)?.get(code) as DecorationInstanceRenderOptions
  }

  public update(): void {
    this.buildDecorationType()
    this.buildTextMarkDecorationType()
    this.buildWordRegexp()
    this.buildCharset()
    this.buildCodeOptions()
  }

  public handleConfigurationChange(event: ConfigurationChangeEvent): boolean {
    if (event.affectsConfiguration(SettingNamespace.Jump)) {
      this.update()
      return true
    } else if (event.affectsConfiguration(SettingNamespace.Editor)) {
      this.buildDecorationType()
      this.buildTextMarkDecorationType()
      this.buildWordRegexp()
      this.buildCharset()
      this.buildCodeOptions()
      return true
    } else {
      return false
    }
  }

  private buildDecorationType(): void {
    const jumpConfig = workspace.getConfiguration(SettingNamespace.Jump)
    const editorConfig = workspace.getConfiguration(SettingNamespace.Editor)
    const useIcons = jumpConfig.get<boolean>(Setting.UseIcons) ?? DEFAULT_USE_ICONS

    this.charOffset = useIcons ? 2 : 0

    const fontFamily = editorConfig.get(Setting.FontFamily) as string
    const editorFontSize = editorConfig.get(Setting.FontSize) as number
    const fontSizeScale = jumpConfig.get(DisplaySetting.FontScale) as number
    const colors = jumpConfig.get<Array<string>>(DisplaySetting.Color) ?? []
    // prettier-ignore
    const backgroundColors = jumpConfig.get<Array<string>>(DisplaySetting.BackgroundColor) ?? []

    const fontSize = fontSizeScale * editorFontSize

    const pad = 2 * Math.ceil(fontSize / (10 * 2))
    const width = fontSize + pad * 2

    const options = {
      pad,
      fontSize: editorFontSize,
      fontFamily,
      colors,
      backgroundColors,
      width,
      height: fontSize,
    }

    const decorationTypeOptions:
      | ThemableDecorationAttachmentRenderOptions
      | ThemableDecorationRenderOptions = useIcons
      ? {
          width: `${width}px`,
          height: `${fontSize}px`,
          margin: `-${width}px 0 0 0`,
        }
      : {
          width: `${width}px`,
          height: `${fontSize}px`,
        }

    this.decorationOptions = options
    this.decorationType = window.createTextEditorDecorationType({
      before: decorationTypeOptions,
    })

    this.cursorSurroundingLines = editorConfig.get(Setting.CursorSurroundingLines) as number
  }

  private buildTextMarkDecorationType(): void {
    const jumpConfig = workspace.getConfiguration(SettingNamespace.Jump)
    const editorConfig = workspace.getConfiguration(SettingNamespace.Editor)
    const useIcons = jumpConfig.get<boolean>(Setting.UseIcons) ?? DEFAULT_USE_ICONS

    this.charOffset = useIcons ? 2 : 0

    const fontFamily = editorConfig.get(Setting.FontFamily) as string
    const fontSize = editorConfig.get(Setting.FontSize) as number
    const colors = jumpConfig.get<Array<string>>(DisplaySetting.Color) ?? []
    // prettier-ignore
    const backgroundColors = ["#AFBBA1"]

    const pad = 2 * Math.ceil(fontSize / (10 * 2))

    const options = {
      pad,
      fontSize,
      fontFamily,
      colors,
      backgroundColors,
      width: fontSize,
      height: fontSize,
    }

    this.textDecorationType = window.createTextEditorDecorationType({
      backgroundColor: backgroundColors[0],
    })

    this.textMarkOptions = options
  }

  private buildWordRegexp(): void {
    const jumpConfig = workspace.getConfiguration(SettingNamespace.Jump)
    const userPrimaryRegexes = jumpConfig[Setting.PrimaryRegexes]
    const userWordRegexFlags = jumpConfig[Setting.WordRegexpFlags]
    const userInlineRegexes = jumpConfig[Setting.InlineRegexes]

    this.primaryRegexes = [
      ...userPrimaryRegexes.map((regex_str: string) => new RegExp(regex_str, userWordRegexFlags)),
    ]

    this.inlineRegexes = [
      ...userInlineRegexes.map((regex_str: string) => new RegExp(regex_str, userWordRegexFlags)),
    ]
  }

  private buildCharset(): void {
    const jumpConfig = workspace.getConfiguration(SettingNamespace.Jump)
    const charsetSetting = jumpConfig.get<string>(Setting.PrimaryCharset)
    const charset = charsetSetting?.length ? charsetSetting.toLowerCase().split('') : undefined
    this.codes = createCharCodeSet(charset)
  }

  private buildCodeOptions(): void {
    const settings = workspace.getConfiguration(SettingNamespace.Jump)
    const useIcons = settings.get<boolean>(Setting.UseIcons) ?? DEFAULT_USE_ICONS

    for (let i = 0; i <= Math.max(this.primaryRegexes.length, this.inlineRegexes.length); ++i) {
      this.codeOptions.set(i, new Map())
      for (const code of [...this.codes.short, ...this.codes.long]) {
        const [codePrefix, codeSuffix] = useIcons
          ? this.createCodeAffixes(i, code.length)
          : ['', '']
        this.codeOptions
          .get(i)
          ?.set(code, this.createRenderOptions(useIcons, `${codePrefix}${code}${codeSuffix}`))
      }
    }
  }

  public createTextMarkOptions(length: number): DecorationInstanceRenderOptions {
    return this.createRenderOptions(DEFAULT_USE_ICONS, this.createTextMarker(length))
  }

  private createRenderOptions(
    useIcons: boolean,
    optionValue: string,
  ): DecorationInstanceRenderOptions {
    const key = useIcons ? 'contentIconPath' : 'contentText'
    const value = useIcons ? DATA_URI.with({ path: optionValue }) : optionValue

    return {
      dark: {
        before: {
          [key]: value,
        },
      },
      light: {
        before: {
          [key]: value,
        },
      },
    }
  }

  private createCodeAffixes(regex_index: number, code_length: number): [string, string] {
    // prettier-ignore
    const { pad, fontSize, backgroundColors, fontFamily, colors, width, height } = this.decorationOptions
    const halfOfPad = pad >> 1

    const backgroundColor = backgroundColors[regex_index]
    const color = colors[regex_index]

    // Width is for 2 characters
    const actual_width = (code_length * (width - pad)) / 2 + pad

    return [
      `image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${actual_width} ${height}" height="${height}" width="${actual_width}"><rect width="${actual_width}" height="${height}" rx="2" ry="2" fill="${backgroundColor}"></rect><text font-family="${fontFamily}" font-size="${height}px" textLength="${
        actual_width - pad
      }" fill="${color}" x="${halfOfPad}" y="${0.8 * height}">`,
      `</text></svg>`,
    ]
  }

  private createTextMarker(length: number): string {
    const { pad, fontSize, backgroundColors, fontFamily, colors, width, height } = this.textMarkOptions
    const backgroundColor = backgroundColors[0]
    const color = colors[0]
    const actual_width = width * length + 2 * pad
    return `image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${actual_width} ${fontSize}" height="${height}" width="${actual_width}"><rect width="${actual_width}" height="${height}" fill="${backgroundColor}"></rect><text font-family="${fontFamily}" font-size="${fontSize}px" textLength="${ actual_width - 2 * pad }" fill="${color}" x="${pad}" y="${fontSize * 0.8}"> </text></svg>`
  }
}
