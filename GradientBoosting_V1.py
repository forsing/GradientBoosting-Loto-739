#!/usr/bin/env python3

"""
GradientBoosting_V1
Deterministicki boosting nad celim kombinacijama.

Skor sluzi rangiranju kombinacija.
Nije kalibrisana verovatnoca pogotka.
"""

import csv
import sys
from pathlib import Path

import numpy as np


DEFAULTS = (
    "/data/loto7_4702_k82.csv",
    "/data/loto7_4702_k82_loto_2973.csv",
    "/data/loto7_4702_k82_loto_plus_1729.csv",
)

WINDOW = 96
TRAIN = 192
VALIDATION = 48
TEST = 64

# Bira se samo na validaciji; zatim ostaje zamrznuto.
PARAMS = (
    (24, 0.08),
    (40, 0.08),
    (40, 0.14),
)

PAIR_I, PAIR_J = np.triu_indices(7, 1)


def load(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or all(not s.strip() for s in row):
                continue

            try:
                c = tuple(int(s.strip()) for s in row)
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: ocekujem sedam celih brojeva"
                ) from exc

            if (
                len(c) != 7
                or len(set(c)) != 7
                or min(c) < 1
                or max(c) > 39
            ):
                raise ValueError(
                    f"{path}, red {line}: neispravna kombinacija {c}"
                )

            rows.append(tuple(sorted(c)))

    minimum = WINDOW + TRAIN + VALIDATION + TEST
    if len(rows) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} redova"
        )

    # Redosled CSV-a ostaje redosled izvlačenja.
    return np.asarray(rows, dtype=np.int16)


def permitted(c):
    """Samo pravila igre. Nema zabrane nizova, grupa ili parnosti."""
    c = tuple(sorted(map(int, c)))

    return (
        len(c) == 7
        and len(set(c)) == 7
        and c[0] >= 1
        and c[-1] <= 39
    )


def describe(c):
    """Opis distribucije cele kombinacije, bez osam razmaka."""
    c = np.atleast_2d(c).astype(float)

    distances = c[:, PAIR_J] - c[:, PAIR_I]

    # Empirijska distribucija svih 21 rastojanja unutar kombinacije.
    bins = np.column_stack([
        (distances <= b).mean(axis=1)
        for b in (2, 5, 10, 20)
    ])

    # Osobine rasporeda, bez propisanog broja po oblasti.
    occupancy = np.column_stack([
        ((c >= a) & (c <= b)).sum(axis=1)
        for a, b in (
            (1, 10),
            (11, 20),
            (21, 30),
            (31, 39),
        )
    ])

    angle = 2 * np.pi * c / 39

    spectral = np.column_stack([
        (
            np.cos(k * angle).sum(axis=1) ** 2
            + np.sin(k * angle).sum(axis=1) ** 2
        ) / 49
        for k in (1, 2, 3)
    ])

    return np.column_stack((
        c.sum(axis=1) / 273,
        (c[:, -1] - c[:, 0]) / 38,
        bins,
        occupancy / 7,
        spectral,
    ))


class Context:
    """Distribucija zavrsenih kombinacija i njihovih prelaza."""

    def __init__(self, history):
        self.history = np.asarray(history[-WINDOW:])
        self.desc = describe(self.history)

        # Robustna skala iz podataka; nema pretpostavke normalnosti.
        self.scale = np.maximum(
            np.quantile(self.desc, 0.75, axis=0)
            - np.quantile(self.desc, 0.25, axis=0),
            0.08,
        )

        self.incidence = np.zeros((len(self.history), 39))

        self.incidence[
            np.arange(len(self.history))[:, None],
            self.history - 1,
        ] = 1

        self.previous = self.incidence[-1]

        # Prethodnik -> naslednik, samo zavrsena izvlačenja.
        delta = (
            self.desc[:-1] - self.desc[-1]
        ) / self.scale

        self.weights = (
            1 / (1 + np.mean(delta * delta, axis=1)) ** 2
        )
        self.weights /= self.weights.sum()

        # Odnosi parova. Dijagonala pojedinacnih pojavljivanja se uklanja.
        self.graph = (
            self.incidence.T @ self.incidence
            / len(self.history)
        )
        np.fill_diagonal(self.graph, 0)

    def features(self, candidates):
        c = np.atleast_2d(candidates).astype(int)
        d = describe(c)

        incidence = np.zeros((len(c), 39))
        incidence[
            np.arange(len(c))[:, None],
            c - 1,
        ] = 1

        matches = incidence @ self.incidence.T

        # Zajednicka podrska parova i trojki cele kombinacije.
        pair_support = matches * (matches - 1) / 42

        triples = (
            matches * (matches - 1) * (matches - 2) / 210
        )

        overlap = incidence @ self.previous / 7

        delta = (
            d[:, None, :] - self.desc[None, :, :]
        ) / self.scale

        kernel = (
            1 / (1 + np.mean(delta * delta, axis=2)) ** 2
        )

        affinities = self.graph[
            (c - 1)[:, PAIR_I],
            (c - 1)[:, PAIR_J],
        ]

        extra = np.column_stack((
            kernel.mean(axis=1),
            kernel[:, 1:] @ self.weights,
            pair_support.mean(axis=1),
            pair_support[:, 1:] @ self.weights,
            triples.mean(axis=1),
            triples[:, 1:] @ self.weights,
            affinities.mean(axis=1),
            affinities.min(axis=1),
            affinities.max(axis=1),
            overlap,
            np.mean(
                np.abs(d - self.desc[-1]),
                axis=1,
            ),
        ))

        return np.column_stack((d, extra))

    def seeds(self):
        order = np.argsort(
            -self.weights,
            kind="stable",
        )[:12]

        raw = [
            tuple(c)
            for c in self.history[-8:]
        ]

        raw += [
            tuple(self.history[i + 1])
            for i in order
        ]

        return sorted(set(
            c for c in raw if permitted(c)
        ))


class Boost:
    """Newton boosting: deterministicka stabla dubine dva."""

    def __init__(self, trees, rate):
        self.count = trees
        self.rate = rate

    @staticmethod
    def sigmoid(x):
        return (
            1 / (1 + np.exp(-np.clip(x, -30, 30)))
        )

    def fit(self, x, y, sample_weights):
        self.edges = [
            np.unique(
                np.quantile(
                    x[:, j],
                    np.linspace(0, 1, 17)[1:-1],
                )
            )
            for j in range(x.shape[1])
        ]

        binned = np.column_stack([
            np.searchsorted(
                edges,
                x[:, j],
                side="left",
            )
            for j, edges in enumerate(self.edges)
        ])

        self.trees = []
        score = np.zeros(len(y))
        indices = np.arange(len(y))

        for _ in range(self.count):
            p = self.sigmoid(score)
            g = sample_weights * (y - p)
            h = sample_weights * p * (1 - p)

            def build(ids, depth):
                total_g = g[ids].sum()
                total_h = h[ids].sum()

                leaf = float(np.clip(
                    total_g / (total_h + 2),
                    -2,
                    2,
                ))

                if depth == 0 or len(ids) < 24:
                    return leaf

                best_gain = 1e-9
                best = None

                for j, edges in enumerate(self.edges):
                    if not len(edges):
                        continue

                    bins = binned[ids, j]
                    length = len(edges) + 1

                    gs = np.cumsum(
                        np.bincount(
                            bins,
                            weights=g[ids],
                            minlength=length,
                        )
                    )[:-1]

                    hs = np.cumsum(
                        np.bincount(
                            bins,
                            weights=h[ids],
                            minlength=length,
                        )
                    )[:-1]

                    ns = np.cumsum(
                        np.bincount(
                            bins,
                            minlength=length,
                        )
                    )[:-1]

                    gains = (
                        gs ** 2 / (hs + 2)
                        + (total_g - gs) ** 2
                        / (total_h - hs + 2)
                        - total_g ** 2 / (total_h + 2)
                    )

                    gains[
                        (ns < 8)
                        | (len(ids) - ns < 8)
                    ] = -np.inf

                    k = int(np.argmax(gains))

                    if gains[k] > best_gain:
                        best_gain = float(gains[k])
                        best = (j, k)

                if best is None:
                    return leaf

                j, k = best
                mask = binned[ids, j] <= k

                return (
                    j,
                    float(self.edges[j][k]),
                    build(ids[mask], depth - 1),
                    build(ids[~mask], depth - 1),
                )

            tree = build(indices, 2)
            self.trees.append(tree)

            score += (
                self.rate * self.apply(tree, x)
            )

        return self

    @staticmethod
    def apply(tree, x):
        if isinstance(tree, float):
            return np.full(len(x), tree)

        j, threshold, left, right = tree
        mask = x[:, j] <= threshold

        out = np.empty(len(x))
        out[mask] = Boost.apply(left, x[mask])
        out[~mask] = Boost.apply(right, x[~mask])

        return out

    def score(self, x):
        result = np.zeros(len(x))

        for tree in self.trees:
            result += (
                self.rate * self.apply(tree, x)
            )

        return result


def training_candidates(actual):
    actual = tuple(map(int, actual))
    candidates = {actual}

    # Deterministicke alternativne kombinacije za trening rangiranja.
    # Ne koriste se kao stvarno izvuceni ishodi.
    for position in range(7):
        for shift in (1, 7):
            c = list(actual)
            c[position] = (
                (c[position] - 1 + shift) % 39 + 1
            )

            if len(set(c)) == 7:
                candidates.add(tuple(sorted(c)))

    return sorted(candidates)


def make_example(data, t):
    # Ulaz modela vidi iskljucivo redove pre t.
    ctx = Context(data[:t])

    # Ishod t koristi se samo za oznaku trening primera.
    actual = tuple(map(int, data[t]))
    candidates = training_candidates(actual)

    x = ctx.features(candidates)

    y = np.array(
        [c == actual for c in candidates],
        dtype=float,
    )

    # Jednaka ukupna tezina stvarnog ishoda i alternativa.
    # Ovo je tezina trening klasa, ne raspodela loto brojeva.
    w = np.where(
        y == 1,
        1.0,
        1.0 / (len(candidates) - 1),
    )

    return x, y, w


def train_model(data, t, params, cache):
    indices = range(
        max(WINDOW, t - TRAIN),
        t,
    )

    parts = []

    for i in indices:
        if i not in cache:
            cache[i] = make_example(data, i)

        parts.append(cache[i])

    x, y, w = (
        np.concatenate([
            part[j] for part in parts
        ])
        for j in range(3)
    )

    return Boost(*params).fit(x, y, w)


def search(model, ctx):
    states = ctx.seeds()

    if not states:
        raw = tuple(map(int, ctx.history[-1]))

        for i in range(7):
            for n in range(1, 40):
                c = tuple(sorted(
                    raw[:i] + (n,) + raw[i + 1:]
                ))

                if permitted(c):
                    states.append(c)

    if not states:
        raise RuntimeError(
            "Nije pronadjen pocetni kandidat"
        )

    states = sorted(set(states))
    scores = model.score(ctx.features(states))

    order = sorted(
        range(len(states)),
        key=lambda i: (-scores[i], states[i]),
    )

    states = [
        states[i] for i in order[:6]
    ]

    best = states[0]
    best_score = float(
        model.score(ctx.features([best]))[0]
    )

    # Deterministicka pretraga; ne tvrdi globalni maksimum
    # svih mogucih kombinacija.
    for _ in range(5):
        candidates = set(states)

        for c in states:
            for i in range(7):
                remainder = c[:i] + c[i + 1:]

                for n in range(1, 40):
                    if n in remainder:
                        continue

                    new = tuple(sorted(
                        remainder + (n,)
                    ))

                    if permitted(new):
                        candidates.add(new)

        candidates = sorted(candidates)

        scores = model.score(
            ctx.features(candidates)
        )

        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (
                -scores[i],
                candidates[i],
            ),
        )

        new_states = [
            candidates[i]
            for i in ranked[:6]
        ]

        new_score = float(scores[ranked[0]])

        if new_score > best_score + 1e-12:
            best = new_states[0]
            best_score = new_score

        if new_states == states:
            break

        states = new_states

    assert permitted(best)
    return best


def hits(a, b):
    return len(set(a) & set(b))


def evaluate(data, start, end, params, cache):
    predictions = []
    model = None

    for t in range(start, end):
        if (
            model is None
            or (t - start) % 16 == 0
        ):
            model = train_model(
                data,
                t,
                params,
                cache,
            )

        predictions.append(
            search(model, Context(data[:t]))
        )

    found = [
        hits(c, data[t])
        for t, c in zip(
            range(start, end),
            predictions,
        )
    ]

    control = [
        hits(data[t - 1], data[t])
        for t in range(start, end)
    ]

    return (
        np.asarray(found),
        np.asarray(control),
        predictions,
    )


def run(path):
    data = load(path)
    n = len(data)

    test_start = n - TEST
    val_start = test_start - VALIDATION

    # Svaki CSV ima sopstveni cache, model i izbor parametara.
    cache = {}

    print("\n" + str(Path(path)), flush=True)

    print(
        f"Redova: {n}; poslednja: "
        + ",".join(map(str, data[-1])),
        flush=True,
    )

    trials = []

    for params in PARAMS:
        result, _, _ = evaluate(
            data,
            val_start,
            test_start,
            params,
            cache,
        )

        trials.append(float(result.mean()))

        print(
            f"Validacija {params}: "
            f"{result.mean():.4f}",
            flush=True,
        )

    chosen = max(
        range(len(PARAMS)),
        key=lambda i: (trials[i], -i),
    )

    params = PARAMS[chosen]

    result, control, predictions = evaluate(
        data,
        test_start,
        n,
        params,
        cache,
    )

    print(
        f"Zamrznuti parametri: {params}; "
        f"test: {TEST} kola"
    )

    print(
        f"Model: {result.mean():.4f}; "
        f"3+: {int((result >= 3).sum())}/{TEST}; "
        f"4+: {int((result >= 4).sum())}/{TEST}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo: "
        f"{control.mean():.4f}"
    )

    print(
        f"Razlika: "
        f"{(result - control).mean():+.4f}; "
        f"razlicitih predikcija: "
        f"{len(set(predictions))}/{TEST}"
    )

    print(
        "Distribucija pogodaka 0..7:",
        np.bincount(
            result,
            minlength=8,
        ).tolist(),
    )

    historical_repeats = sum(
        any(
            tuple(row) == c
            for row in data[:t]
        )
        for t, c in zip(
            range(test_start, n),
            predictions,
        )
    )

    print(
        "Ponavljanja istorijske kombinacije u testu:",
        historical_repeats,
    )

    final_model = train_model(
        data,
        n,
        params,
        cache,
    )

    next_combination = search(
        final_model,
        Context(data),
    )

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, next_combination))
    )

    print(
        "NEXT vec postoji u istoriji:",
        any(
            tuple(row) == next_combination
            for row in data
        ),
    )

    if result.mean() <= control.mean():
        print(
            "Test nije pokazao bolji prosek od kontrole."
        )

    return next_combination


def self_check():
    # Nizovi i grupe nisu rucno zabranjeni.
    assert permitted((1, 2, 3, 4, 5, 22, 35))
    assert permitted((1, 2, 4, 29, 30, 33, 35))
    assert permitted((2, 6, 13, 19, 24, 31, 38))

    x = (
        np.arange(240, dtype=float)
        .reshape(80, 3) / 240
    )

    y = (x[:, 0] > 0.5).astype(float)
    w = np.ones(80)

    a = Boost(12, 0.1).fit(x, y, w).score(x)
    b = Boost(12, 0.1).fit(x, y, w).score(x)

    assert np.array_equal(a, b)
    assert a[y == 1].mean() > a[y == 0].mean()

    # Buduci redovi ne menjaju trening primer za trenutak t.
    rows = np.array([
        sorted({
            (3 * i + 5 * j) % 39 + 1
            for j in range(7)
        })
        for i in range(110)
    ], dtype=np.int16)

    first = make_example(rows, 100)

    changed = rows.copy()
    changed[101:] = rows[0]

    second = make_example(changed, 100)

    assert all(
        np.array_equal(a, b)
        for a, b in zip(first, second)
    )


def main():
    self_check()

    print(
        "Deterministicki boosting celih kombinacija; "
        "bez random generisanja."
    )

    print(
        "Nema rucnih filtera oblika kombinacije; "
        "samo pravila 7/39."
    )

    outputs = []

    for path in (sys.argv[1:] or DEFAULTS):
        try:
            outputs.append((path, run(path)))
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    # Iste rezultate prijavljuje; ne menja ih vestacki.
    for i, (left, a) in enumerate(outputs):
        for right, b in outputs[i + 1:]:
            if a == b:
                print(
                    f"ISTA NEXT za {Path(left).name} "
                    f"i {Path(right).name}: {a}"
                )


if __name__ == "__main__":
    main()



"""
Deterministicki boosting celih kombinacija; bez random generisanja.
Nema rucnih filtera oblika kombinacije; samo pravila 7/39.


/data/loto7_4702_k82.csv
Redova: 4702; poslednja: 8,10,13,21,22,23,32
Validacija (24, 0.08): 0.8542
Validacija (40, 0.08): 1.0625
Validacija (40, 0.14): 1.1875
Zamrznuti parametri: (40, 0.14); test: 64 kola
Model: 1.2031; 3+: 1/64; 4+: 0/64
Kontrola: prethodno izvuceno kolo: 1.1719
Razlika: +0.0312; razlicitih predikcija: 63/64
Distribucija pogodaka 0..7: [15, 22, 26, 1, 0, 0, 0, 0]
Ponavljanja istorijske kombinacije u testu: 0
NEXT red 4703: 6,x,8,y,18,z,38
NEXT vec postoji u istoriji: False



/data/loto7_4702_k82_loto_2973.csv
Redova: 2973; poslednja: 1,2,8,26,29,35,39
Validacija (24, 0.08): 1.1875
Validacija (40, 0.08): 1.0625
Validacija (40, 0.14): 1.3750
Zamrznuti parametri: (40, 0.14); test: 64 kola
Model: 1.2969; 3+: 7/64; 4+: 0/64
Kontrola: prethodno izvuceno kolo: 1.3281
Razlika: -0.0312; razlicitih predikcija: 56/64
Distribucija pogodaka 0..7: [13, 26, 18, 7, 0, 0, 0, 0]
Ponavljanja istorijske kombinacije u testu: 0
NEXT red 2974: 7,x,17,y,32,z,39
NEXT vec postoji u istoriji: False
Test nije pokazao bolji prosek od kontrole.



/data/loto7_4702_k82_loto_plus_1729.csv
Redova: 1729; poslednja: 8,10,13,21,22,23,32
Validacija (24, 0.08): 1.1042
Validacija (40, 0.08): 1.2500
Validacija (40, 0.14): 1.1458
Zamrznuti parametri: (40, 0.08); test: 64 kola
Model: 1.2344; 3+: 6/64; 4+: 0/64
Kontrola: prethodno izvuceno kolo: 1.2656
Razlika: -0.0312; razlicitih predikcija: 47/64
Distribucija pogodaka 0..7: [15, 25, 18, 6, 0, 0, 0, 0]
Ponavljanja istorijske kombinacije u testu: 7
NEXT red 1730: 3,x,11,y,13,z,18
NEXT vec postoji u istoriji: False
Test nije pokazao bolji prosek od kontrole.
"""



"""
Koristim implementaciju determinističkog gradient boostinga: 
niz stabala dubine 2, treniranih Newton postupkom.
Model ocenjuje celu kombinaciju preko opisa njene distribucije, 
odnosa parova i trojki i sličnosti sa istorijskim prelazima. 
Zatim deterministička pretraga bira kombinaciju sa najvećom pronađenom ocenom.


Uklonio sam ručne zabrane oblika kombinacije. 
Model ocenjuje distribuciju cele kombinacije; jedina obavezna provera je sedam različitih brojeva od 1 do 39. 
Nema random generisanja, nametnute uniformne ili normalne raspodele, niti rangiranja po frekvenciji pojedinačnih brojeva.
Provereno na sva tri CSV-a, poslednja 64 kola:

CSV	        Model: prosečno pogodaka	    Kontrola: prethodno izvučeno kolo
4702	    1,2031	                        1,1719
2973	    1,2969	                        1,3281
1729	    1,2344	                        1,2656

Ovi rezultati ne potvrđuju da je ovo najbolji prediktor. 
Kod prolazi proveru ponovljivosti i proveru da budući redovi ne utiču na istorijske trening primere. 
Ovo je jedna proverena implementacija determinističkog boostinga, sa potpuno prikazanim rezultatima.
"""
