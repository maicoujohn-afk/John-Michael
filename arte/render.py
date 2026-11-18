"""Filosoficamente — "A Caverna"
Renderizador procedural da arte de perfil do canal.

Conceito: um jovem contemporâneo, sentado diante da fogueira no fundo da
caverna de Platão, projeta na parede de pedra não a própria sombra, mas a de
Platão, que aponta para cima, para a saída e a luz azul da noite lá fora.
O passado e o presente se unem através da sombra.

Uso: python3 render.py   (gera os PNGs em ./saida)
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.environ.get("FONTS_DIR", "/tmp/claude-0/fonts")
OUT = os.path.join(HERE, "saida")
os.makedirs(OUT, exist_ok=True)

CINZEL = f"{FONTS}/fontsource-cinzel-5.3.0/files/cinzel-latin-{{w}}-normal.woff"
MONT = f"{FONTS}/montserrat/files/montserrat-latin-{{w}}-normal.woff"
CORM = f"{FONTS}/fontsource-cormorant-garamond-5.3.0/files/cormorant-garamond-latin-{{w}}-italic.woff"
IVORY = (238, 228, 208)


# ----------------------------------------------------------------- utilidades
def blur(a, sigma):
    """Gaussian blur por FFT com borda refletida."""
    if sigma <= 0.3:
        return a
    p = int(3 * sigma) + 1
    b = np.pad(a, p, mode="reflect")
    h, w = b.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    r = np.fft.irfft2(np.fft.rfft2(b) * g, s=b.shape)
    return r[p:-p, p:-p].astype(np.float32)


def vnoise(h, w, scale, rng):
    gh, gw = max(2, int(h / scale) + 3), max(2, int(w / scale) + 3)
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray(g, "F").resize((w, h), Image.BICUBIC)
    return np.asarray(im)


def fbm(h, w, scale, rng, octaves=6, gain=0.5):
    tot, amp, norm = np.zeros((h, w), np.float32), 1.0, 0.0
    for i in range(octaves):
        s = scale / (2 ** i)
        if s < 1.5:
            break
        tot += amp * vnoise(h, w, s, rng)
        norm += amp
        amp *= gain
    return tot / norm


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class Canvas:
    """Sistema de coordenadas da cena: um quadrado de lado S com origem (ox, oy)."""

    def __init__(self, W, H, S, ox, oy, ss=2):
        self.W, self.H, self.S, self.ox, self.oy, self.ss = W, H, S, ox, oy, ss
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.x = (xx - ox) / S  # coordenadas de cena
        self.y = (yy - oy) / S

    def px(self, x, y, ss=1):
        return ((self.ox + x * self.S) * ss, (self.oy + y * self.S) * ss)

    def mask(self, draw_fn):
        """Desenha com supersampling e devolve máscara float 0..1."""
        ss = self.ss
        im = Image.new("L", (self.W * ss, self.H * ss), 0)
        d = ImageDraw.Draw(im)
        draw_fn(d, lambda x, y: self.px(x, y, ss), ss)
        im = im.resize((self.W, self.H), Image.LANCZOS)
        return np.asarray(im, np.float32) / 255.0


def catmull(pts, n=10, closed=True):
    P = list(pts)
    out = []
    m = len(P)
    rng_ = range(m) if closed else range(m - 1)
    for i in rng_:
        p0, p1 = P[(i - 1) % m], P[i]
        p2, p3 = P[(i + 1) % m], P[(i + 2) % m]
        if not closed:
            p0 = P[max(i - 1, 0)]
            p3 = P[min(i + 2, m - 1)]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                       + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3)
                for j in range(2)))
    if not closed:
        out.append(P[-1])
    return out


# ------------------------------------------------------------------- formas
PLATO = [  # sombra de Platão (perfil voltado à direita, braço erguido)
    (0.468, 0.314), (0.493, 0.318), (0.503, 0.329), (0.503, 0.338),
    (0.519, 0.356), (0.506, 0.361), (0.509, 0.368), (0.506, 0.372),
    (0.513, 0.392), (0.514, 0.418), (0.505, 0.404), (0.522, 0.415), (0.540, 0.408), (0.560, 0.398),
    (0.571, 0.388), (0.574, 0.350), (0.577, 0.312), (0.575, 0.296),
    (0.583, 0.282), (0.586, 0.263), (0.590, 0.249), (0.594, 0.262),
    (0.598, 0.281), (0.602, 0.294), (0.599, 0.312), (0.604, 0.347),
    (0.608, 0.398), (0.601, 0.428), (0.607, 0.468), (0.598, 0.503),
    (0.592, 0.525), (0.599, 0.600), (0.611, 0.700), (0.622, 0.810),
    (0.348, 0.810), (0.352, 0.740), (0.362, 0.680), (0.368, 0.620),
    (0.372, 0.596), (0.333, 0.590), (0.336, 0.513), (0.372, 0.520),
    (0.377, 0.480), (0.386, 0.445), (0.399, 0.424), (0.425, 0.409),
    (0.446, 0.393), (0.444, 0.372), (0.441, 0.348), (0.449, 0.328),
]

MAN = [  # jovem de casaco, sentado, visto de costas, cabeça baixa
    (0.500, 0.614), (0.519, 0.618), (0.531, 0.632), (0.534, 0.648),
    (0.539, 0.647), (0.541, 0.661), (0.533, 0.674), (0.527, 0.683),
    (0.548, 0.688), (0.575, 0.693), (0.598, 0.703), (0.611, 0.722),
    (0.615, 0.760), (0.617, 0.810), (0.616, 0.842), (0.606, 0.856),
    (0.394, 0.856), (0.384, 0.842), (0.383, 0.810), (0.385, 0.760), (0.389, 0.722),
    (0.402, 0.703), (0.425, 0.693), (0.452, 0.688), (0.473, 0.683),
    (0.467, 0.674), (0.459, 0.661), (0.461, 0.647), (0.466, 0.648),
    (0.469, 0.632), (0.481, 0.618),
]


def render(W, H, S, ox, oy, seed=11):
    rng = np.random.default_rng(seed)
    C = Canvas(W, H, S, ox, oy)
    x, y = C.x, C.y
    u = S / 1000.0  # 1 "unidade" ~ 1/1000 da cena, para tamanhos em px

    # ---------------------------------------------------------- pedra / parede
    stone = fbm(H, W, 260 * u, rng, 7, 0.55)
    fine = fbm(H, W, 18 * u, rng, 3)
    ridge = 1 - np.abs(fbm(H, W, 120 * u, rng, 5) - 0.5) * 2
    cracks = smooth(0.955, 0.995, ridge) * (0.5 + 0.5 * fine)
    strata = 0.5 + 0.5 * np.sin((y * 38 + stone * 9) * 1.0)
    albedo = 0.20 + 0.36 * stone + 0.08 * (fine - 0.5) + 0.03 * strata
    albedo *= 1 - 0.15 * cracks

    # chão
    floor_line = 0.795 + 0.012 * (vnoise(H, W, 90 * u, rng) - 0.5)
    floor = smooth(floor_line - 0.004, floor_line + 0.012, y)
    pebbles = fbm(H, W, 10 * u, rng, 3)
    albedo = albedo * (1 - floor) + floor * (0.16 + 0.22 * stone + 0.10 * pebbles)

    # ---------------------------------------------------------- boca da caverna
    mx, my, rx, ry = 0.5, 0.168, 0.120, 0.062
    mn = fbm(H, W, 60 * u, rng, 5)
    md = np.sqrt(((x - mx) / rx) ** 2 + ((y - my) / ry) ** 2)
    md = md + 0.55 * (mn - 0.5) + 0.25 * np.clip(y - my, 0, 1) / ry * 0.15
    mouth = smooth(1.05, 0.90, md)
    mouth_soft = blur(mouth, 40 * u)

    # céu noturno: azul-petróleo, lua velada, neblina, colunata grega distante
    sky_g = np.clip((y - (my - ry)) / (2 * ry), 0, 1)
    sky = np.stack([0.05 + 0.10 * sky_g, 0.17 + 0.20 * sky_g, 0.22 + 0.22 * sky_g], -1)
    moon_d = np.sqrt((x - mx - 0.045) ** 2 + (y - my + 0.030) ** 2)
    moon = smooth(0.0115, 0.0095, moon_d)
    halo = np.exp(-(moon_d / 0.06) ** 2) * 0.55 + np.exp(-(moon_d / 0.022) ** 2) * 0.4
    mist = fbm(H, W, 80 * u, rng, 5)
    sky = sky + (halo + 0.95 * moon)[..., None] * np.array([0.55, 0.80, 0.85])
    # colunata (Partenon) na névoa
    def temple(d, P, ss):
        base_y, top_y = my + 0.036, my + 0.010
        x0, x1 = mx - 0.060, mx + 0.032
        n = 8
        for i in range(n):
            cx = x0 + (x1 - x0) * i / (n - 1)
            w = 0.0028
            d.rectangle([P(cx - w, top_y), P(cx + w, base_y)], fill=255)
        d.rectangle([P(x0 - 0.006, top_y - 0.005), P(x1 + 0.006, top_y)], fill=255)
        d.polygon([P(x0 - 0.007, top_y - 0.005), P(x1 + 0.007, top_y - 0.005),
                   P((x0 + x1) / 2, top_y - 0.016)], fill=255)
        d.rectangle([P(x0 - 0.008, base_y), P(x1 + 0.008, base_y + 0.004)], fill=255)
        # colina
        hill = [(mx - 0.20, my + 0.09), (mx - 0.13, my + 0.045), (mx - 0.07, my + 0.040), (mx + 0.02, my + 0.039),
                (mx + 0.10, my + 0.050), (mx + 0.20, my + 0.09)]
        d.polygon([P(*p) for p in hill], fill=255)
    tm = C.mask(temple)
    sky = sky * (1 - 0.62 * tm[..., None] * (0.7 + 0.3 * mist[..., None]))
    sky = sky + (mist[..., None] - 0.35) * np.array([0.10, 0.17, 0.19]) * (0.6 + sky_g[..., None])
    sky = np.clip(sky, 0, None)

    # ---------------------------------------------------------- iluminação
    fx, fy = 0.5, 0.905  # fogueira
    d_f = np.sqrt(((x - fx) * 1.0) ** 2 + ((y - fy) * 1.15) ** 2)
    warm = 1.5 / (1 + (d_f / 0.10) ** 2) ** 1.2 + 2.6 / (1 + (d_f / 0.42) ** 2) ** 1.7
    warm_col = np.array([1.00, 0.30, 0.13])

    # sombra de Platão (projeção da fogueira) — bordas que vibram com a pedra
    plato = C.mask(lambda d, P, ss: d.polygon([P(*p) for p in PLATO], fill=255))
    dn = (vnoise(H, W, 30 * u, rng) - 0.5) * 6 * u
    yi = np.clip((np.arange(H)[:, None] + dn).astype(int), 0, H - 1)
    dn2 = (vnoise(H, W, 30 * u, rng) - 0.5) * 6 * u
    xi = np.clip((np.arange(W)[None, :] + dn2).astype(int), 0, W - 1)
    plato = plato[yi, xi]
    pen = blur(plato, 2.2 * u) * 0.75 + blur(plato, 9 * u) * 0.25
    # dobras da toga gravadas (linhas de luz finas dentro da sombra)
    def folds(d, P, ss):
        lw = max(1, int(1.6 * u * ss))
        strokes = [
            [(0.405, 0.430), (0.455, 0.505), (0.530, 0.548), (0.592, 0.585)],
            [(0.392, 0.470), (0.430, 0.560), (0.500, 0.618), (0.600, 0.650)],
            [(0.422, 0.424), (0.490, 0.462), (0.548, 0.482), (0.595, 0.515)],
            [(0.385, 0.620), (0.410, 0.700), (0.425, 0.780)],
            [(0.560, 0.620), (0.575, 0.700), (0.585, 0.790)],
            [(0.470, 0.640), (0.480, 0.720), (0.482, 0.800)],
            [(0.582, 0.405), (0.586, 0.455), (0.590, 0.500)],
        ]
        for s in strokes:
            pts = catmull(s, 12, closed=False)
            d.line([P(*p) for p in pts], fill=255, width=lw)
    fold = C.mask(folds) * plato * 0.6
    shadow = np.clip(pen - 0.55 * blur(fold, 1.2 * u), 0, 1)

    # gravura: hachuras cruzadas dentro da sombra
    ang1, ang2 = math.radians(58), math.radians(-32)
    per = 7.5 * u / S
    h1 = 0.5 + 0.5 * np.cos(2 * math.pi * (x * math.cos(ang1) + y * math.sin(ang1)) / per
                           + 2.0 * fine)
    h2 = 0.5 + 0.5 * np.cos(2 * math.pi * (x * math.cos(ang2) + y * math.sin(ang2)) / (per * 1.15))
    hatch = smooth(0.55, 0.95, h1) * 0.8 + smooth(0.62, 0.97, h2) * 0.55 * smooth(0.6, 0.95, shadow)
    hatch = np.clip(hatch, 0, 1)

    warm_vis = 1 - shadow * 0.86
    warm_vis = warm_vis * (1 - 0.45 * hatch * shadow)

    # luz fria vinda da boca + facho volumétrico
    d_m = np.sqrt((x - mx) ** 2 + ((y - my) * 0.8) ** 2)
    cool = 0.55 / (1 + (d_m / 0.17) ** 2) ** 1.4 + 0.02
    cool_col = np.array([0.30, 0.62, 0.72])
    t = np.clip((y - (my + ry)) / 0.6, 0, 1)
    shaft_c = 0.5 + 0.06 * t
    shaft_w = 0.06 + 0.10 * t
    shaft = smooth(1.0, 0.0, np.abs(x - shaft_c - 0.03 * t) / shaft_w) * (y > my + ry * 0.6)
    shaft = blur(shaft.astype(np.float32), 25 * u) * (1 - t) ** 1.2
    fog = fbm(H, W, 140 * u, rng, 5)
    shaft *= 0.35 + 0.9 * fog

    light = warm[..., None] * warm_col * warm_vis[..., None] + (cool * (1 - 0.65 * shadow))[..., None] * cool_col
    img = albedo[..., None] * light

    # rocha em volta da boca recebe contraluz azul
    rim = np.clip(mouth_soft * 2.2 - mouth, 0, 1) * (1 - mouth)
    img += albedo[..., None] * rim[..., None] * np.array([0.25, 0.55, 0.62]) * 1.1
    img = img * (1 - mouth[..., None]) + sky * mouth[..., None]
    img += (shaft * (1 - 0.7 * shadow))[..., None] * np.array([0.10, 0.22, 0.26])

    # massa de rocha que emoldura (primeiro plano, quase negra)
    edge = np.sqrt(((x - 0.5) / 0.62) ** 2 + ((y - 0.47) / 0.66) ** 2)
    edge = edge + 0.35 * (fbm(H, W, 150 * u, rng, 6) - 0.5)
    frame = smooth(0.92, 1.05, edge)
    frame_lit = albedo * (warm * 0.10) * (1 - smooth(0.98, 1.2, edge))
    img = img * (1 - frame[..., None]) + frame[..., None] * (
        frame_lit[..., None] * warm_col + albedo[..., None] * 0.012 * np.array([0.5, 0.8, 1.0]))

    # ---------------------------------------------------------- fumaça
    sm_t = np.clip((0.86 - y) / 0.6, 0, 1)
    sm_w = 0.04 + 0.22 * sm_t
    sway = 0.04 * np.sin(y * 9) * sm_t
    col = smooth(1, 0, np.abs(x - 0.5 - sway) / sm_w) * (y < 0.86)
    wisp = fbm(H, W, 70 * u, rng, 6)
    wisp2 = vnoise(H, W, 25 * u, rng)
    smoke = col * smooth(0.45, 0.85, wisp * 0.8 + 0.2 * wisp2) * (1 - sm_t) ** 0.6 * 0.55
    smoke = blur(smoke.astype(np.float32), 2 * u)
    sm_col = (warm[..., None] * 0.30 * warm_col + cool[..., None] * 0.5 * cool_col + 0.03)
    img = img * (1 - 0.35 * smoke[..., None]) + smoke[..., None] * sm_col * 0.55

    # ---------------------------------------------------------- jovem
    man = C.mask(lambda d, P, ss: d.polygon([P(*p) for p in catmull(MAN, 4)], fill=255))
    inner = blur(man, 22 * u)
    coat = fbm(H, W, 40 * u, rng, 4)
    folds_v = 0.5 + 0.5 * np.sin((x - 0.5) * 210 + coat * 4)
    lit = smooth(0.65, 0.88, y) * 0.9 + 0.12
    man_col = (0.012 + 0.11 * inner * lit * (0.75 + 0.25 * folds_v) * (0.8 + 0.4 * coat))[..., None] \
        * np.array([1.0, 0.30, 0.16])
    hair = smooth(0.684, 0.672, y) * man
    man_col = man_col * (1 - 0.75 * hair[..., None])
    # brilho carmim na borda da orelha / pescoço
    ear = np.exp(-(((x - 0.531) / 0.006) ** 2 + ((y - 0.660) / 0.012) ** 2))
    neck = np.exp(-(((x - 0.500) / 0.022) ** 2 + ((y - 0.682) / 0.006) ** 2))
    man_col += ((ear * 0.55 + neck * 0.20) * man)[..., None] * np.array([0.95, 0.12, 0.08])
    k = max(2, int(5 * u))
    top_edge = np.clip(man - np.roll(man, k, axis=0), 0, 1)
    top_edge = blur(top_edge, 1.5 * u) * smooth(0.76, 0.66, y)
    man_col += (top_edge * 0.9)[..., None] * np.array([0.22, 0.48, 0.56])
    def arms(d, P, ss):
        lw = max(1, int(3 * u * ss))
        for pts in ([(0.410, 0.712), (0.416, 0.770), (0.420, 0.850)],
                    [(0.592, 0.712), (0.585, 0.770), (0.581, 0.850)],
                    [(0.500, 0.700), (0.501, 0.780), (0.500, 0.850)]):
            d.line([P(*p) for p in catmull(pts, 10, closed=False)], fill=255, width=lw)
    am = blur(C.mask(arms), 2.0 * u) * man
    man_col *= (1 - 0.75 * am)[..., None]
    img = img * (1 - man[..., None]) + man_col * man[..., None]

    # ---------------------------------------------------------- coluna dórica quebrada
    def column(d, P, ss):
        top = [(0.073, 0.615), (0.092, 0.604), (0.104, 0.611), (0.121, 0.590),
               (0.137, 0.594), (0.152, 0.575), (0.166, 0.580), (0.182, 0.562)]
        poly = top + [(0.182, 0.905), (0.073, 0.905)]
        d.polygon([P(*p) for p in poly], fill=255)
    colm = C.mask(column)
    cu = np.clip((x - 0.073) / 0.109, 0, 1)
    flutes = 0.5 + 0.5 * np.cos(cu * math.pi * 2 * 7.5)
    shade = np.clip(np.sin(cu * math.pi), 0, 1) ** 0.6 * (0.25 + 0.75 * cu)
    marble = 0.55 + 0.25 * fbm(H, W, 30 * u, rng, 4) - 0.25 * cracks
    vein = smooth(0.93, 0.99, 1 - np.abs(fbm(H, W, 50 * u, rng, 4) - 0.5) * 2)
    marble = marble * (1 - 0.4 * vein)
    cwarm = (warm * 1.05 + 0.10) * shade ** 1.6 * smooth(0.55, 0.70, y) * (0.7 + 0.3 * flutes)
    ccool = smooth(0.25, 0.0, cu) * 0.22
    col_rgb = marble[..., None] * (cwarm[..., None] * warm_col * 1.2 + ccool[..., None] * cool_col)
    img = img * (1 - colm[..., None]) + col_rgb * colm[..., None]

    # ---------------------------------------------------------- manuscrito e corrente
    def scroll(d, P, ss):
        d.polygon([P(0.700, 0.868), P(0.858, 0.858), P(0.872, 0.902), P(0.708, 0.914)], fill=255)
        d.ellipse([P(0.690, 0.864), P(0.712, 0.918)], fill=255)
        d.ellipse([P(0.855, 0.852), P(0.879, 0.906)], fill=255)
    scm = C.mask(scroll)
    def script(d, P, ss):
        lw = max(1, int(1.2 * u * ss))
        r = np.random.default_rng(3)
        for row in range(6):
            yy = 0.874 + row * 0.0058
            xx = 0.722
            while xx < 0.848:
                L = r.uniform(0.004, 0.012)
                d.line([P(xx, yy - (xx - 0.70) * 0.075), P(xx + L, yy - (xx + L - 0.70) * 0.075)],
                       fill=255, width=lw)
                xx += L + 0.003
    txt = C.mask(script) * scm
    paper = (0.62 + 0.2 * fine)[..., None] * np.array([1.0, 0.86, 0.66])
    sc_rgb = paper * (warm[..., None] * warm_col * 1.1 + 0.03) * (1 - 0.7 * txt[..., None])
    img = img * (1 - scm[..., None]) + sc_rgb * scm[..., None]

    def chain(d, P, ss):
        lw = max(1, int(3.2 * u * ss))
        r = np.random.default_rng(5)
        cx, cy = 0.300, 0.872
        for i in range(9):
            a = 0.25 * math.sin(i * 0.9)
            w, h = (0.016, 0.008) if i % 2 == 0 else (0.012, 0.004)
            box = [P(cx - w, cy - h), P(cx + w, cy + h)]
            d.ellipse(box, outline=255, width=lw)
            cx += 0.019
            cy += 0.004 * math.sin(i * 1.3) + 0.0012
    chm = C.mask(chain)
    ch_rgb = (warm[..., None] * np.array([1.0, 0.40, 0.20]) * 0.55 + 0.012) * 0.6
    img = img * (1 - chm[..., None]) + ch_rgb * chm[..., None]

    # ---------------------------------------------------------- fogueira
    def logs(d, P, ss):
        for (a, b, w) in [((0.43, 0.925), (0.565, 0.900), 0.010), ((0.445, 0.897), (0.575, 0.928), 0.010),
                          ((0.47, 0.930), (0.53, 0.888), 0.008)]:
            d.line([P(*a), P(*b)], fill=255, width=int(w * S * ss))
    lg = C.mask(logs)
    emb = smooth(0.62, 0.8, fbm(H, W, 8 * u, rng, 3))
    log_rgb = (0.015 + emb * 1.3)[..., None] * np.array([1.0, 0.18, 0.05]) * lg[..., None]
    img = img * (1 - lg[..., None]) + log_rgb

    fl_n = fbm(H, W, 22 * u, rng, 5)
    fl_n2 = vnoise(H, W, 9 * u, rng)
    v = np.clip((0.905 - y) / 0.135, 0, 1.2)
    disp = (fl_n - 0.5) * 0.06 * (0.3 + v) + 0.012 * np.sin(v * 7 + 1.0) * v
    width = 0.046 * np.clip(1 - v, 0, 1) ** 0.75 * (1 + 0.6 * np.exp(-v * 6))
    fd = np.abs(x - 0.5 - disp)
    flame = smooth(width + 0.002, width * 0.25, fd) * (y < 0.915)
    flame *= smooth(0.0, 0.25, 0.98 - v + 0.35 * (fl_n2 - 0.5)) * smooth(0.925, 0.900, y)
    # línguas extras
    for off, hscale, wsc in [(-0.025, 0.75, 0.55), (0.028, 0.65, 0.5), (0.008, 1.12, 0.35)]:
        vv = np.clip((0.905 - y) / (0.135 * hscale), 0, 1.2)
        ww = 0.046 * wsc * np.clip(1 - vv, 0, 1) ** 0.8
        dd = np.abs(x - 0.5 - off - disp * 1.3)
        flame = np.maximum(flame, smooth(ww + 0.002, ww * 0.2, dd) * smooth(0.0, 0.3, 0.95 - vv) *
                           smooth(0.925, 0.900, y))
    flame = np.clip(blur(flame.astype(np.float32), 1.2 * u), 0, 1)
    core = flame ** 2.2
    f_rgb = (flame[..., None] * np.array([0.85, 0.07, 0.03]) * 1.7
             + (flame ** 1.8)[..., None] * np.array([1.0, 0.30, 0.05]) * 0.75
             + (flame ** 3.5)[..., None] * np.array([1.0, 0.74, 0.40]) * 0.45)
    img += f_rgb
    glow = blur(flame, 30 * u) * 1.2 + blur(flame, 90 * u) * 0.9
    img += glow[..., None] * np.array([0.9, 0.14, 0.05]) * 0.55

    # brasas subindo
    img = np.nan_to_num(img)
    E = np.zeros((H, W), np.float32)
    n_emb = 60
    ex = 0.5 + rng.normal(0, 1, n_emb) * 0.05
    ey = 0.88 - rng.power(0.6, n_emb) * 0.55
    ex += (0.88 - ey) * rng.normal(0, 0.18, n_emb)
    for X, Y in zip(ex, ey):
        px, py = C.px(X, Y)
        if 0 <= int(py) < H and 0 <= int(px) < W:
            r = max(1, int(rng.uniform(0.5, 1.4) * u))
            E[max(0, int(py) - r):int(py) + r + 1, max(0, int(px) - r):int(px) + r + 1] += \
                rng.uniform(0.4, 1.0) * (1 - (0.88 - Y) / 0.7)
    Eg = blur(E, 1.0 * u) * 2.2 + blur(E, 5 * u) * 1.2
    img += Eg[..., None] * np.array([1.0, 0.32, 0.08])

    # poeira na luz azul
    D = (rng.random((H, W)) > 0.9993).astype(np.float32) * shaft * 6
    img += blur(D, 1.0 * u)[..., None] * np.array([0.45, 0.65, 0.70])

    return img, C


# ----------------------------------------------------------------- acabamento
def finish(img, C, seed=4, vign=0.55):
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    u = C.S / 1000.0
    # tone map filmico
    img = 1 - np.exp(-img * 1.55)
    # split-tone: sombras para o azul-petróleo, preto texturizado
    lum = img.mean(-1, keepdims=True)
    tint = np.array([0.018, 0.034, 0.040])
    img = img + tint * (1 - smooth(0.0, 0.25, lum))
    # contraste em S
    img = np.clip(img, 0, 1)
    img = img * img * (3 - 2 * img) * 0.55 + img * 0.45
    # vinheta
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - vign * smooth(0.55, 1.45, r))[..., None]
    # papel envelhecido / carvão
    paper = fbm(H, W, 400 * u, rng, 6)
    fibers = vnoise(H, W, 2.5 * u, rng)
    img *= (0.90 + 0.14 * paper + 0.04 * (fibers - 0.5))[..., None]
    # grão de filme forte (luminância + leve cor)
    g1 = rng.normal(0, 1, (H, W)).astype(np.float32)
    g2 = np.asarray(Image.fromarray(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), "F")
                    .resize((W, H), Image.BILINEAR))
    grain = (g1 * 0.6 + g2 * 0.7)
    amt = 0.045 * (0.35 + 0.65 * np.sqrt(np.clip(lum, 0, 1)))
    img = img + (grain * amt[..., 0])[..., None] + rng.normal(0, 0.008, img.shape).astype(np.float32)
    # arranhões finos
    for _ in range(int(6 * W / 2000)):
        x0 = rng.uniform(0, W)
        img[:, int(x0):int(x0) + 1] += rng.uniform(0.01, 0.03)
    return np.clip(img, 0, 1)


def to_image(img):
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGB")


# ----------------------------------------------------------------- tipografia
def tracked(draw, xy, text, font, fill, tracking, anchor_center=True):
    widths = [font.getlength(ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x0, y0 = xy
    x = x0 - total / 2 if anchor_center else x0
    for ch, w in zip(text, widths):
        draw.text((x, y0), ch, font=font, fill=fill, anchor="ls")
        x += w + tracking
    return total


def text_shadow_layer(size, fn, radius):
    """Cria uma camada de sombra escura suave sob o texto para legibilidade."""
    m = Image.new("L", size, 0)
    fn(ImageDraw.Draw(m), 255)
    a = blur(np.asarray(m, np.float32) / 255.0, radius)
    return np.clip(a * 1.6, 0, 1)


def apply_text(base, fn, shadow_r, shadow_amt=0.75):
    W, H = base.size
    sh = text_shadow_layer((W, H), fn, shadow_r)
    arr = np.asarray(base, np.float32) / 255
    arr *= (1 - shadow_amt * sh)[..., None]
    out = to_image(arr)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fn(ImageDraw.Draw(layer), IVORY + (255,))
    out = out.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def arc_text(draw, cx, cy, radius, text, font, fill, tracking, top=True):
    """Texto em arco: no topo lê-se por fora do círculo; embaixo, por dentro."""
    widths = [font.getlength(ch) + tracking for ch in text]
    total = sum(widths) - tracking
    ang_total = total / radius
    if top:
        a = -math.pi / 2 - ang_total / 2
    else:
        a = math.pi / 2 + ang_total / 2
    for ch, w in zip(text, widths):
        cw = w - tracking
        mid = a + (cw / 2) / radius * (1 if top else -1)
        size = int(font.size * 2.2)
        g = Image.new("L", (size, size), 0)
        gd = ImageDraw.Draw(g)
        gd.text((size / 2, size / 2), ch, font=font, fill=255, anchor="mm")
        rot = -(math.degrees(mid) + 90) if top else -(math.degrees(mid) - 90)
        g = g.rotate(rot, resample=Image.BICUBIC)
        px = cx + radius * math.cos(mid)
        py = cy + radius * math.sin(mid)
        draw(g, int(px - size / 2), int(py - size / 2), fill)
        a += (w / radius) * (1 if top else -1)


# ----------------------------------------------------------------- formatos
def make_avatar(N=2048):
    """Avatar 1:1 seguro para corte circular, com nomes em arco."""
    S = int(N * 0.90)
    img, C = render(N, N, S, (N - S) / 2, N * 0.045)
    base = to_image(finish(img, C, vign=0.75))
    base.save(f"{OUT}/filosoficamente_avatar_sem_texto.png")

    f_title = ImageFont.truetype(CINZEL.format(w=700), int(N * 0.060))
    f_name = ImageFont.truetype(MONT.format(w=500), int(N * 0.030))
    cx = cy = N / 2

    def paint(target_draw_fn):
        pass

    W = H = N
    mask_txt = Image.new("L", (W, H), 0)

    def put(g, X, Y, fill):
        mask_txt.paste(255, (X, Y), g)

    arc_text(put, cx, cy, N * 0.415, "FILOSOFICAMENTE", f_title, IVORY, N * 0.0065, top=True)
    arc_text(put, cx, cy, N * 0.448, "JOHN MICHAEL", f_name, IVORY, N * 0.012, top=False)
    m = np.asarray(mask_txt, np.float32) / 255
    # anel escuro para legibilidade + filete marfim
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / N
    band = smooth(0.33, 0.40, rr)
    arr = np.asarray(base, np.float32) / 255
    arr *= (1 - 0.55 * band)[..., None]
    arr *= (1 - 0.6 * blur(m, N * 0.006))[..., None]
    ring = np.exp(-((rr - 0.4775) / 0.0011) ** 2) * 0.55
    # pequenos losangos separadores nas laterais
    for side in (-1, 1):
        dx = cx + side * N * 0.445
        d = (np.abs(xx - dx) + np.abs(yy - cy)) / N
        ring += smooth(0.0085, 0.0065, d)
    ivory = np.array(IVORY, np.float32) / 255
    alpha = np.clip(m + ring, 0, 1)[..., None]
    arr = arr * (1 - alpha) + ivory * alpha
    # fora do círculo: preto (para quem não recorta)
    outside = smooth(0.497, 0.502, rr)[..., None]
    arr = arr * (1 - outside) + np.array([0.02, 0.025, 0.03]) * outside
    out = to_image(np.clip(arr, 0, 1))
    out.save(f"{OUT}/filosoficamente_avatar_perfil.png")
    # prévia circular
    circ = Image.new("L", (N, N), 0)
    ImageDraw.Draw(circ).ellipse([0, 0, N - 1, N - 1], fill=255)
    prev = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    prev.paste(out, (0, 0), circ)
    prev.save(f"{OUT}/filosoficamente_avatar_previa_circular.png")


def make_post(N=2160):
    """Post quadrado 1:1 com identificação completa."""
    S = int(N * 0.86)
    img, C = render(N, N, S, (N - S) / 2, N * 0.125)
    base = to_image(finish(img, C))
    f_title = ImageFont.truetype(CINZEL.format(w=600), int(N * 0.072))
    f_name = ImageFont.truetype(MONT.format(w=500), int(N * 0.024))

    def fn(d, fill):
        tracked(d, (N / 2, N * 0.105), "FILOSOFICAMENTE", f_title, fill, N * 0.010)
        d.line([(N * 0.45, N * 0.132), (N * 0.55, N * 0.132)], fill=fill, width=max(2, N // 900))
        tracked(d, (N / 2, N * 0.962), "JOHN MICHAEL", f_name, fill, N * 0.012)
    apply_text(base, fn, N * 0.01).save(f"{OUT}/filosoficamente_post_1x1.png")


def make_vertical(W=1440, H=2560):
    """Cartaz vertical 9:16: nome do canal no alto, cena ao centro, assinatura embaixo."""
    S = int(W * 1.12)
    img, C = render(W, H, S, (W - S) / 2, H * 0.20)
    base = to_image(finish(img, C, vign=0.6))
    f_title = ImageFont.truetype(CINZEL.format(w=600), int(W * 0.070))
    f_sub = ImageFont.truetype(CORM.format(w=500), int(W * 0.045))
    f_name = ImageFont.truetype(MONT.format(w=500), int(W * 0.032))

    def fn(d, fill):
        tracked(d, (W / 2, H * 0.105), "FILOSOFICAMENTE", f_title, fill, W * 0.008)
        d.line([(W * 0.44, H * 0.123), (W * 0.56, H * 0.123)], fill=fill, width=3)
        tracked(d, (W / 2, H * 0.955), "JOHN MICHAEL", f_name, fill, W * 0.016)
    apply_text(base, fn, W * 0.012).save(f"{OUT}/filosoficamente_vertical_9x16.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["avatar", "post", "vertical"]
    if "avatar" in which:
        make_avatar()
    if "post" in which:
        make_post()
    if "vertical" in which:
        make_vertical()
    print("ok")
