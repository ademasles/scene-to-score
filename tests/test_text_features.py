"""B1 tests: text building, metadata reading and the cache, with a tiny random CLAP stand-in.

No download and no real data: the tokenizer is a whitespace toy and the model a small random
module with the two entry points the extraction uses (``get_text_features``, ``text_model``).
"""

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from scene_to_score.features.text import (
    VARIANTS,
    Film,
    build_text,
    encode_batch,
    extract_variant,
    read_films,
    resolve_max_length,
    text_dir,
    token_lengths,
)

COLUMNS = {
    "movie_id": "movie_id",
    "genres": "genres",
    "year": "year",
    "synopsis": "imdb_description",
}
HIDDEN, PROJ, VOCAB = 16, 8, 64


class ToyTokenizer:
    """Words to ids, with <s> and </s>; right padding; keeps </s> when truncating."""

    model_max_length = 16
    padding_side = "right"

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        ids = []
        for text in texts:
            seq = [0] + [2 + sum(map(ord, word)) % 60 for word in text.split()] + [1]
            if truncation and max_length is not None and len(seq) > max_length:
                seq = [*seq[: max_length - 1], 1]
            ids.append(seq)
        if return_tensors is None:
            return {"input_ids": ids}
        width = max(map(len, ids))
        padded = torch.full((len(ids), width), 63, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for row, seq in enumerate(ids):
            padded[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": padded, "attention_mask": mask}


class ToyTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        keep = attention_mask.unsqueeze(-1).to(torch.float32)
        hidden = self.emb(input_ids) * keep
        pooled = hidden.sum(1) / keep.sum(1)
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=pooled)


class ToyClap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = ToyTower()
        self.proj = nn.Linear(HIDDEN, PROJ, bias=False)

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = nn.functional.normalize(self.proj(out.pooler_output), dim=-1)
        return BaseModelOutputWithPooling(
            last_hidden_state=out.last_hidden_state,
            pooler_output=cast(torch.FloatTensor, pooled),
        )


@pytest.fixture
def model() -> ToyClap:
    torch.manual_seed(0)
    return ToyClap().eval()


FILM = Film("7", "Drama, War", "1977", "A farm boy leaves home.")


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("full", "Genres: Drama, War. Year: 1977. Synopsis: A farm boy leaves home."),
        ("noyear", "Genres: Drama, War. Synopsis: A farm boy leaves home."),
        ("genres", "Genres: Drama, War."),
        ("synopsis", "Synopsis: A farm boy leaves home."),
    ],
)
def test_build_text_variants(variant: str, expected: str) -> None:
    assert build_text(FILM, variant) == expected


@pytest.mark.parametrize(
    ("film", "expected"),
    [
        (Film("1", None, "1977", "S."), "Year: 1977. Synopsis: S."),
        (Film("1", "Drama", None, "S."), "Genres: Drama. Synopsis: S."),
        (Film("1", "Drama", "1977", None), "Genres: Drama. Year: 1977."),
        (Film("1", None, None, None), ""),
    ],
)
def test_missing_field_disappears(film: Film, expected: str) -> None:
    text = build_text(film, "full")
    assert text == expected
    assert "None" not in text


def test_title_never_in_text() -> None:
    assert not any("title" in build_text(FILM, variant).lower() for variant in VARIANTS)


def test_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        build_text(FILM, "title")


def _write_csv(path: Path, rows: list[dict[str, str]], header: list[str] | None = None) -> None:
    header = header or ["movie_id", "title", "genres", "year", "imdb_description"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _row(movie_id: str, genres: str, year: str, synopsis: str) -> dict[str, str]:
    return {
        "movie_id": movie_id,
        "title": "Some title",
        "genres": genres,
        "year": year,
        "imdb_description": synopsis,
    }


def test_read_films_cleans_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_csv(
        path,
        [
            _row("10", "['Drama', 'War']", "1944.0", "  Two\nlines. "),
            _row("2", "Comedy", "", "None"),
            _row("3", "nan", "1931", "x"),
        ],
    )
    films = read_films(path, COLUMNS)
    assert [film.movie_id for film in films] == ["2", "3", "10"]
    assert films[0] == Film("2", "Comedy", None, None)
    assert films[1].genres is None
    assert films[2] == Film("10", "Drama, War", "1944", "Two lines.")


def test_read_films_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_csv(path, [{"movie_id": "1", "genres": "a"}], header=["movie_id", "genres"])
    with pytest.raises(KeyError, match="year"):
        read_films(path, COLUMNS)


def test_read_films_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    row = _row("1", "a", "1", "s")
    _write_csv(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        read_films(path, COLUMNS)


def test_resolve_max_length() -> None:
    assert resolve_max_length(ToyTokenizer(), None) == 16
    assert resolve_max_length(ToyTokenizer(), 8) == 8
    tokenizer = ToyTokenizer()
    tokenizer.model_max_length = int(1e30)
    with pytest.raises(ValueError, match="max_length"):
        resolve_max_length(tokenizer, None)


def test_token_lengths() -> None:
    lengths = token_lengths(ToyTokenizer(), ["", "a b c"])
    assert lengths.tolist() == [2, 5]


def test_encode_batch_shapes_and_unit_norm(model: ToyClap) -> None:
    pooled, hidden, mask = encode_batch(model, ToyTokenizer(), ["a b", "a b c d e"], 16)
    assert pooled.shape == (2, PROJ)
    assert hidden.shape == (2, 7, HIDDEN)
    assert mask.shape == (2, 7)
    assert mask.sum(1).tolist() == [4, 7]
    torch.testing.assert_close(pooled.norm(dim=-1), torch.ones(2))


def test_padding_does_not_change_the_result(model: ToyClap) -> None:
    tokenizer = ToyTokenizer()
    pooled_b, hidden_b, mask_b = encode_batch(model, tokenizer, ["a b", "a b c d e f g"], 16)
    pooled_s, hidden_s, mask_s = encode_batch(model, tokenizer, ["a b"], 16)
    torch.testing.assert_close(pooled_b[0], pooled_s[0])
    torch.testing.assert_close(hidden_b[0, : hidden_s.shape[1]], hidden_s[0])
    assert mask_b[0].sum() == mask_s[0].sum()
    assert not hidden_b[0, hidden_s.shape[1] :].any()


def test_extract_variant_writes_the_cache(model: ToyClap, tmp_path: Path) -> None:
    films = [
        Film("3", "Drama", "1931", "word " * 30),  # longer than max_length: truncated
        Film("5", "Comedy", None, "short one"),
        Film("9", None, None, None),  # nothing to say: empty text
    ]
    out_dir = extract_variant(
        model,
        ToyTokenizer(),
        films,
        "full",
        tmp_path,
        split="all",
        max_length=12,
        batch_size=2,
        model_id="toy",
    )
    assert out_dir == text_dir(tmp_path, "full") == tmp_path / "text" / "clap-full"

    assert (out_dir / "all_ids.txt").read_text().split() == ["3", "5", "9"]
    pooled = np.load(out_dir / "all_pooled.npy", mmap_mode="r")
    tokens = np.load(out_dir / "all_tokens.npy", mmap_mode="r")
    mask = np.load(out_dir / "all_mask.npy", mmap_mode="r")
    assert (pooled.dtype, pooled.shape) == (np.float16, (3, PROJ))
    assert (tokens.dtype, tokens.shape) == (np.float16, (3, 12, HIDDEN))
    assert (mask.dtype, mask.shape) == (np.bool_, (3, 12))
    np.testing.assert_allclose(np.linalg.norm(pooled.astype(np.float32), axis=1), 1, atol=1e-2)
    assert mask.sum(axis=1).tolist() == [12, 7, 2]
    assert not tokens[~mask].any()  # padding is zero

    records = [json.loads(line) for line in (out_dir / "texts.jsonl").read_text().splitlines()]
    assert [r["movie_id"] for r in records] == ["3", "5", "9"]
    assert records[1]["text"] == "Genres: Comedy. Synopsis: short one"
    assert records[2]["text"] == ""
    assert [r["truncated"] for r in records] == [True, False, False]

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_truncated"] == 1
    assert meta["share_truncated"] == pytest.approx(1 / 3)
    assert meta["n_empty_texts"] == 1
    assert meta["n_missing"] == {"genres": 1, "year": 2, "synopsis": 1}
    assert meta["L"] == 12
    assert meta["hidden_size"] == HIDDEN
    assert meta["pooled_dim"] == PROJ
    assert "last_hidden_state" in meta["hidden_states_access"]
    assert meta["fields"] == ["genres", "year", "synopsis"]


def test_variants_go_to_their_own_folder(model: ToyClap, tmp_path: Path) -> None:
    for variant in VARIANTS:
        extract_variant(model, ToyTokenizer(), [FILM], variant, tmp_path, model_id="toy")
    names = sorted(path.name for path in (tmp_path / "text").iterdir())
    assert names == ["clap-full", "clap-genres", "clap-noyear", "clap-synopsis"]
    short = np.load(tmp_path / "text" / "clap-genres" / "all_tokens.npy")
    long = np.load(tmp_path / "text" / "clap-full" / "all_tokens.npy")
    assert short.shape[1] < long.shape[1]


def test_rerun_is_reproducible(model: ToyClap, tmp_path: Path) -> None:
    films = [FILM, Film("8", "Comedy", "1920", "Another story")]
    for name in ("a", "b"):
        extract_variant(model, ToyTokenizer(), films, "full", tmp_path / name, model_id="toy")
    first = np.load(tmp_path / "a" / "text" / "clap-full" / "all_pooled.npy")
    second = np.load(tmp_path / "b" / "text" / "clap-full" / "all_pooled.npy")
    np.testing.assert_array_equal(first, second)
