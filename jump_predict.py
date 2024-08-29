from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
import re
from typing import Callable, cast

import torch
from torch.utils.data import Dataset, DataLoader, random_split


# a = torch.arange(1*2*3).reshape((1,2,3)) + 1
# b = torch.zeros((1,2,2))
# print(a)

# # for i in range(b.shape[0]):
# #     # for j in range(b.shape[1]):
# #     #     # for k in range(b.shape[2]):
# #     #     #     b[i,j,k] = torch.dot(a[i,j], a[i,k])
# #     #     b[i,j] = a[i,j] @ a[i].T
# #     b[i] = a[i] @ a[i].T
# b = a @ a.transpose(1, 2)

# print(b.shape)
# print(b)


TYPE_LOG_PATH = Path('~/.jumpa_type_log').expanduser().absolute()

MAX_LINE_TOKEN_COUNT = 500

class TokenType(Enum):
    OTHER = 0
    LINE_START = auto()
    ALPHA_LOWER = auto()
    ALPHA_UPPER = auto()
    NUMERICAL = auto()
    WHITESPACE = auto()
    OPERATOR = auto()
    END = auto()

@dataclass
class Token:
    type: int
    original: str


python_keywords = [ "False", "None", "True", "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield" ]
cpp_keywords = [ "alignas", "alignof", "and", "asm", "assert", "auto", "await", "bitand", "bitor", "bool", "break", "case", "cast", "catch", "char", "class", "co", "compl", "concept", "const", "consteval", "constexpr", "constinit", "continue", "decltype", "default", "delete", "do", "double", "dynamic", "else", "enum", "eq", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "local", "long", "mutable", "namespace", "new", "noexcept", "not", "nullptr", "operator", "or", "private", "protected", "public", "register", "reinterpret", "requires", "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "thread", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar", "while", "xor", "yield" ]
js_keywords = [ "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "enum", "export", "extends", "false", "finally", "for", "function", "if", "implements", "import", "in", "instanceof", "interface", "let", "new", "null", "package", "private", "protected", "public", "return", "static", "super", "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with", "yield" ]
possible_keywords = set(python_keywords + cpp_keywords + js_keywords)

operators = ['+', '-', '=', '&', '|', '!', '<', '>', '*', '/', '^', '%', '?']


def create_token_builders(lines: list[tuple[int ,str]]) -> tuple[list[Callable[[str], Token | None]], int]:
    words = [m for _, l in lines for m in re.findall(r"[A-Za-z]+", l)]
    word_count: dict[str, int] = {}
    for w in words:
        word_count[w] = 1 if w not in word_count else word_count[w] + 1

    keywords = {w: i for i, w in enumerate(
                                {w for w, c in word_count.items() if c > 100 and w in possible_keywords},
                                start=TokenType.END.value)}

    print('Actual keywords:', keywords)

    char_count: dict[str, int] = {}
    for _, l in lines:
        for c in l:
            char_count[c] = 1 if c not in char_count else char_count[c] + 1

    symbols = {c: i for i, c in enumerate({char for char, count in char_count.items()
                         if count > 100
                         and not char.isalnum()
                         and not char.isspace()
                         and char not in operators},
                         start=TokenType.END.value + len(keywords))}

    print('Actual symbols:', symbols)

    def m(type: int, match):
        return Token(type, match[0]) if match is not None else None

    def n(look: dict[str, int], match):
        return Token(look[match[0]], match[0]) if match is not None else None

    return [
        lambda s: m(TokenType.ALPHA_LOWER.value, re.match(r'^[a-z]+', s)),
        lambda s: m(TokenType.ALPHA_UPPER.value, re.match(r'^[A-Z]+', s)),
        lambda s: m(TokenType.NUMERICAL.value, re.match(r'^\d+\.?\d*', s)),
        lambda s: m(TokenType.WHITESPACE.value, re.match(r'^\s+', s)),
        lambda s: m(TokenType.OPERATOR.value, re.match('^(?:' + '|'.join('\\' + o for o in operators) + ')+', s)),
        lambda s: n(keywords, re.match('^(?:' + '|'.join(keywords) + ')', s)),
        lambda s: n(symbols, re.match('^(?:' + '|'.join('\\' + s for s in symbols) + ')', s)),
        lambda s: Token(TokenType.OTHER.value, s),
    ], TokenType.END.value + len(keywords) + len(symbols)


def tokenize(lines: list[tuple[int, str]]):
    token_builders, num_token_types = create_token_builders(lines)

    def tokenize_line(c: int, l: str):
        out: list[Token] = [Token(TokenType.LINE_START.value, '')]
        jump_token_idx = 0 if c == 0 else None
        while l != '':
            for builder in token_builders:
                t = builder(l)
                if t is not None:
                    out.append(t)
                    l = l[len(t.original):]

                    c -= len(t.original)
                    if jump_token_idx is None and c <= 0:
                        jump_token_idx = len(out) - 1
                    break
            else:
                raise Exception(f'Unmatched sequence "{l}"')
        if jump_token_idx is None:
            raise Exception(f'Jump was not in line!')

        return jump_token_idx, out

    return list(filter(lambda x: len(x[1]) < MAX_LINE_TOKEN_COUNT, [tokenize_line(c, l) for c, l in lines])), num_token_types


class JumpDataset(Dataset):
    def __init__(self, log_path: Path):
        raw_lines: list[tuple[int, str]] = cast(list[tuple[int, str]], list(
            filter(
                lambda x: x is not None,
                map(
                    lambda x: (int(m[1]), m[2]) if (m := re.match(r'[^:]*:\d*:(\d*):(.*)', x)) else None,
                    log_path.open('r').readlines()
                )
            )
        ))

        self.tokenized_lines, self.num_token_types = tokenize(raw_lines)

        self.encoded_lines = [
            torch.nn.functional.one_hot(torch.tensor([t.type for t in tl]), self.num_token_types).float()
            for _, tl in self.tokenized_lines
        ]

        self.jump_target = [
            torch.nn.functional.one_hot(torch.tensor(c), len(tl)).float()
            for c, tl in self.tokenized_lines
        ]

    def __len__(self):
        return len(self.encoded_lines)

    def __getitem__(self, index: int):
        return self.encoded_lines[index], self.jump_target[index]


dataset = JumpDataset(TYPE_LOG_PATH)
TRAIN_DATA_FRACTION = 0.001
train_dataset, test_dataset = random_split(dataset, [TRAIN_DATA_FRACTION, 1 - TRAIN_DATA_FRACTION])

def pad_collate(data: list[tuple[torch.Tensor, torch.Tensor]]):
    tensors, targets = zip(*data)
    features = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return features, targets

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)

# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = MAX_LINE_TOKEN_COUNT):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)


class OuterModel(torch.nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int, hidden_count: int):
        super().__init__()
        self.num_tokens = num_tokens

        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Linear(num_tokens, hidden_size),
            torch.nn.ReLU(),
        )

        self.positional_encoder = PositionalEncoding(hidden_size, dropout=0.0)

        self.attention = torch.nn.MultiheadAttention(hidden_size, 1, batch_first=True)

        self.query, self.key, self.value = (
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size)
        )

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.final = torch.nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BATCH x LENGTH x TOKENS

        x = self.embedding_layer(x) # BATCH x LENGTH x HIDDEN

        x = self.positional_encoder(x) # BATCH x LENGTH x HIDDEN

        q, k, v = self.query(x), self.key(x), self.value(x)
        x, _ = self.attention(q, k, v, need_weights=False) # BATCH x LENGTH x HIDDEN

        x = self.classify(x) # BATCH x LENGTH x 1

        x = torch.squeeze(x, dim=2) # BATCH x LENGTH

        x = self.final(x) # BATCH x LENGTH

        return x

model = OuterModel(dataset.num_token_types, 128, 4)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=10e-5)

for epoch in range(1000):
    epoch_loss = torch.zeros((1))
    for x, y in train_dataloader: # Batch x Length x Tokens, Batch x Length
        # torch.set_printoptions(threshold=10_000)
        # print(x)
        # print(y)
        x: torch.Tensor
        y: torch.Tensor

        optim.zero_grad()

        p: torch.Tensor = model(x)
        # print(p)

        loss = loss_fn(p, y)
        loss.backward()

        optim.step()

        epoch_loss += loss
    print(epoch_loss.item())
