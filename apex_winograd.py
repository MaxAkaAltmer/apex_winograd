from fractions import Fraction
from itertools import product
import math
import warnings
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def calculate_winograd_geometry(out_shape, kernel_shape, stride=1, dilation=1):
    # Приведение к первичным кортежам
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,)

    n_out = len(out_shape)
    n_kernel = len(kernel_shape)

    # Итоговая размерность пространства N-D
    ndim = max(n_out, n_kernel)

    # Если ядро меньше по размерности — дополняем 1 и выводим Warning
    if n_kernel < ndim:
        old_kernel = kernel_shape
        kernel_shape = kernel_shape + (1,) * (ndim - n_kernel)
        warnings.warn(
            f"\n[Winograd Geometry Warning]: Kernel {old_kernel} ({n_kernel}D) "
            f"is smaller than task dimension ({ndim}D). Kernel padded to {kernel_shape}. "
            f"Note: task is physically redundant along spatial axes!",
            UserWarning,
        )

    # out_shape всегда дополняется единицами справа до ndim
    if n_out < ndim:
        out_shape = out_shape + (1,) * (ndim - n_out)

    # Нормализация stride и dilation (дополнение 1 или срезка лишнего)
    def normalize_param(param):
        if isinstance(param, int):
            return (param,) * ndim
        param_tuple = tuple(param)
        if len(param_tuple) < ndim:
            # Дополняем единицами справа
            return param_tuple + (1,) * (ndim - len(param_tuple))
        elif len(param_tuple) > ndim:
            # Обрезаем лишнее до ndim
            return param_tuple[:ndim]
        return param_tuple

    stride = normalize_param(stride)
    dilation = normalize_param(dilation)

    # Расчет точной геометрии входа
    in_shape = []
    for o, k, s, d in zip(out_shape, kernel_shape, stride, dilation):
        k_eff = (k - 1) * d + 1
        i_dim = (o - 1) * s + k_eff
        in_shape.append(i_dim)

    inter_dim = math.prod(in_shape)

    return (
        tuple(in_shape),
        out_shape,
        kernel_shape,
        stride,
        dilation,
        inter_dim,
    )

class WinogradFromSketch(nn.Module):

    def __init__(
        self,
        in_shape: tuple,
        out_shape: tuple,
        kernel_shape: tuple,
        inter_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel_shape = kernel_shape
        self.scale = scale

        d_in = math.prod(in_shape)
        f_in = math.prod(kernel_shape)
        c_out = math.prod(out_shape)

        self.fc_D = nn.Linear(d_in, inter_dim, bias=False)
        self.fc_F = nn.Linear(f_in, inter_dim, bias=False)
        self.fc_C = nn.Linear(inter_dim, c_out, bias=False)

    @property
    def effective_WC(self) -> torch.Tensor:
        return self.fc_C.weight * self.scale

    def forward(self, D: torch.Tensor, F_tensor: torch.Tensor) -> torch.Tensor:
        B = D.size(0)
        out_D = self.fc_D(D.view(B, -1))
        out_F = self.fc_F(F_tensor.view(B, -1))

        # Применяем масштабирование прямо во forward
        out_C = F.linear(out_D * out_F, self.effective_WC)
        return out_C.view(B, 1, *self.out_shape)

    def apply_pruning(self, threshold: float = 0.02):
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.abs(param) >= threshold
                param.mul_(mask)
            

# --- Эталонная свертка ---
def batched_conv_nd(
    D: torch.Tensor, F_tensor: torch.Tensor, stride=1, dilation=1
) -> torch.Tensor:

    ndim = D.ndim - 2
    conv_fn = (F.conv1d, F.conv2d, F.conv3d)[ndim - 1]

    B = D.size(0)
    inp = D.view(1, B, *D.shape[2:])
    w = F_tensor.view(B, 1, *F_tensor.shape[2:])

    out = conv_fn(inp, w, stride=stride, dilation=dilation, groups=B)
    return out.view(B, 1, *out.shape[2:])

def get_1d_winograd_scale(dim: int) -> int:
    if dim <= 2:
        return 1
    nodes = []
    for i in range(dim - 1):
        val = (i + 1) // 2
        nodes.append(val if i % 2 == 1 else -val)
    
    denominators = []
    for j, xj in enumerate(nodes):
        prod = 1
        for k, xk in enumerate(nodes):
            if j != k:
                prod *= abs(xj - xk)
        denominators.append(prod)
    
    return math.lcm(*denominators)

def get_max_scaler(in_shape: tuple) -> int:

    res = 1
    for dim in in_shape:
        res *= get_1d_winograd_scale(dim)
    return res
    
def get_max_denominator(in_shape: tuple) -> int:

    res = 1
    for dim in in_shape:
        res *= math.factorial(dim - 1)
    return res

# --- Квантование тензора к ближайшим дробям ---
def snap_tensor_to_fractions(
    tensor: torch.Tensor, max_denominator: int, threshold: float
) -> torch.Tensor:
    arr = tensor.detach().cpu().numpy()
    snapped_arr = np.zeros_like(arr)

    for idx, val in np.ndenumerate(arr):
        if abs(val) < threshold:
            snapped_arr[idx] = 0.0
        else:
            frac = Fraction(float(val)).limit_denominator(max_denominator)
            snapped_arr[idx] = float(frac)

    return torch.tensor(snapped_arr, dtype=tensor.dtype, device=tensor.device)


def normalize_and_snap_matrices(
    model: nn.Module, prune_threshold: float, max_denom: int
):
    with torch.no_grad():
        WD_snap = snap_tensor_to_fractions(
            model.fc_D.weight,
            max_denominator=max_denom,
            threshold=prune_threshold,
        )
        WF_snap = snap_tensor_to_fractions(
            model.fc_F.weight,
            max_denominator=max_denom,
            threshold=prune_threshold,
        )
        # Берем эффективную матрицу с учетом scale!
        WC_snap = snap_tensor_to_fractions(
            model.effective_WC,
            max_denominator=max_denom,
            threshold=prune_threshold,
        )

        return WD_snap, WF_snap, WC_snap


# --- Оценка точности матриц ---
def evaluate_custom_matrices(
    WD: torch.Tensor,
    WF: torch.Tensor,
    WC: torch.Tensor,
    in_shape: tuple,
    kernel_shape: tuple,
    out_shape: tuple,
    stride=1,
    dilation=1,
    batch_size=2048,
):
    with torch.no_grad():
        D_rand = torch.randn(batch_size, 1, *in_shape, device=device)
        F_rand = torch.randn(batch_size, 1, *kernel_shape, device=device)

        target = batched_conv_nd(
            D_rand, F_rand, stride=stride, dilation=dilation
        )

        d_flat = D_rand.view(batch_size, -1)
        f_flat = F_rand.view(batch_size, -1)

        out_D = F.linear(d_flat, WD)
        out_F = F.linear(f_flat, WF)
        out_C = F.linear(out_D * out_F, WC)

        pred = out_C.view(batch_size, 1, *out_shape)
        abs_diff = torch.abs(pred - target)

        mse = F.mse_loss(pred, target).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(abs_diff).item()
        max_err = torch.max(abs_diff).item()

        return mse, rmse, mae, max_err


# --- Вывод матриц ---

def print_matrix(name: str, tensor: torch.Tensor, threshold: float):
    arr = tensor.detach().cpu().numpy()
    arr_clean = np.where(np.abs(arr) < threshold, 0.0, arr)
    zeros_pct = (arr_clean == 0.0).mean() * 100

    print(f"\n--- Matrix {name} (shape: {arr.shape}) ---")
    print(f"Zero percentage (< {threshold}): {zeros_pct:.1f}%")

    with np.printoptions(
        precision=2,
        suppress=True,
        linewidth=9999,
        threshold=np.inf,
        formatter={"float": "{:6.2f}".format},
    ):
        print(arr_clean)

def print_matrix_as_fractions(
    name: str, tensor: torch.Tensor, max_denom: int = 12, threshold: float = 0.01
):
    arr = tensor.detach().cpu().numpy()
    rows, cols = arr.shape
    zeros_count = 0

    print(
        f"\n--- Matrix {name} (fractions with q <= {max_denom}, shape: {arr.shape}) ---"
    )
    for r in range(rows):
        row_str = []
        for c in range(cols):
            val = arr[r, c]
            if abs(val) < threshold:
                row_str.append(f"{'0':>7}")
                zeros_count += 1
            else:
                frac = Fraction(float(val)).limit_denominator(max_denom)
                row_str.append(f"{str(frac):>7}")
        print("[" + " ".join(row_str) + " ]")

    zeros_pct = (zeros_count / (rows * cols)) * 100
    print(f"Zero percentage: {zeros_pct:.1f}%")
    

def flat_index(coords: tuple, shape: tuple) -> int:
    idx = 0
    acc = 1
    for c, s in zip(reversed(coords), reversed(shape)):
        idx += c * acc
        acc *= s
    return idx
    
def generate_symbolic_target(
    in_shape: tuple,
    out_shape: tuple,
    kernel_shape: tuple,
    stride: tuple,
    dilation: tuple,
):
    num_d = math.prod(in_shape)
    num_f = math.prod(kernel_shape)
    d_sym = sp.symbols(f"d0:{num_d}")
    f_sym = sp.symbols(f"f0:{num_f}")

    out_coords_list = list(product(*[range(o) for o in out_shape]))
    kernel_coords_list = list(product(*[range(k) for k in kernel_shape]))

    target_sym = []
    for out_coords in out_coords_list:
        expr = sp.S.Zero
        for k_coords in kernel_coords_list:
            d_coords = [
                o * s + k * d
                for o, k, s, d in zip(out_coords, k_coords, stride, dilation)
            ]
            d_idx = flat_index(d_coords, in_shape)
            f_idx = flat_index(k_coords, kernel_shape)
            expr += d_sym[d_idx] * f_sym[f_idx]
        target_sym.append(expr)

    return target_sym, d_sym, f_sym


def get_convolution_monomial_map(in_shape, out_shape, kernel_shape, stride, dilation):
    out_coords_list = list(product(*[range(o) for o in out_shape]))
    kernel_coords_list = list(product(*[range(k) for k in kernel_shape]))
    monomials = []

    for out_coords in out_coords_list:
        for k_coords in kernel_coords_list:
            d_coords = [o * s + k * d for o, k, s, d in zip(out_coords, k_coords, stride, dilation)]
            d_idx = flat_index(d_coords, in_shape)
            f_idx = flat_index(k_coords, kernel_shape)
            pair = (d_idx, f_idx)
            if pair not in monomials:
                monomials.append(pair)
    return monomials


def build_target_matrix(in_shape, out_shape, kernel_shape, stride, dilation, monomials):
    out_coords_list = list(product(*[range(o) for o in out_shape]))
    kernel_coords_list = list(product(*[range(k) for k in kernel_shape]))
    target_mat = np.zeros((len(out_coords_list), len(monomials)), dtype=np.float64)

    for out_idx, out_coords in enumerate(out_coords_list):
        for k_coords in kernel_coords_list:
            d_coords = [o * s + k * d for o, k, s, d in zip(out_coords, k_coords, stride, dilation)]
            d_idx = flat_index(d_coords, in_shape)
            f_idx = flat_index(k_coords, kernel_shape)
            mono_idx = monomials.index((d_idx, f_idx))
            target_mat[out_idx, mono_idx] += 1.0

    return target_mat


# --- Аналитический символьный сольвер ---
def solve_winograd_exact_symbolic(
    model,
    in_shape: tuple,
    out_shape: tuple,
    kernel_shape: tuple,
    stride: tuple,
    dilation: tuple,
    threshold: float = 0.02,
    max_denom: int = 36,
    verbose: bool = True,
):
    if verbose:
        print("\n" + "=" * 60)
        print("Launching analytical symbolic solver (SymPy + LSTSQ)")
        print("=" * 60)

    # Извлекаем обученные веса
    WD_float = model.fc_D.weight.detach().cpu().numpy()
    WF_float = model.fc_F.weight.detach().cpu().numpy()

    inter_dim = WD_float.shape[0]
    d_num = math.prod(in_shape)
    f_num = math.prod(kernel_shape)

    monomials = get_convolution_monomial_map(
        in_shape, out_shape, kernel_shape, stride, dilation
    )
    num_monomials = len(monomials)

    Target_num = build_target_matrix(
        in_shape, out_shape, kernel_shape, stride, dilation, monomials
    )
    Target_sym = sp.Matrix(Target_num)

    best_WD_exact, best_WF_exact, best_WC_exact = None, None, None
    best_diff_norm = float("inf")

    # Непрерывный перебор знаменателей WD и WF от 1 до max_denom
    for q_try in range(1, max_denom + 1):
        WD_exact = np.zeros_like(WD_float, dtype=object)
        WF_exact = np.zeros_like(WF_float, dtype=object)
        WD_num = np.zeros_like(WD_float, dtype=np.float64)
        WF_num = np.zeros_like(WF_float, dtype=np.float64)

        for i in range(inter_dim):
            max_d = np.max(np.abs(WD_float[i]))
            max_f = np.max(np.abs(WF_float[i]))

            for j in range(d_num):
                if abs(WD_float[i, j]) >= threshold and max_d > 0:
                    frac = Fraction(float(WD_float[i, j] / max_d)).limit_denominator(q_try)
                    WD_exact[i, j] = sp.Rational(frac.numerator, frac.denominator)
                    WD_num[i, j] = float(frac)

            for j in range(f_num):
                if abs(WF_float[i, j]) >= threshold and max_f > 0:
                    frac = Fraction(float(WF_float[i, j] / max_f)).limit_denominator(q_try)
                    WF_exact[i, j] = sp.Rational(frac.numerator, frac.denominator)
                    WF_num[i, j] = float(frac)

        # Строим матрицу M_num размера (num_monomials x inter_dim)
        M_num = np.zeros((num_monomials, inter_dim), dtype=np.float64)
        for k in range(inter_dim):
            for m_idx, (d_i, f_j) in enumerate(monomials):
                M_num[m_idx, k] = WD_num[k, d_i] * WF_num[k, f_j]

        # Решение МНК: M_num * WC_num^T = Target_num.T
        X_num, residuals, rank, s = np.linalg.lstsq(M_num, Target_num.T, rcond=None)
        WC_num = X_num.T

        # Квантуем пересчитанную WC к простым дробям (до max_denom)
        WC_exact = np.zeros_like(WC_num, dtype=object)
        WC_num_q = np.zeros_like(WC_num, dtype=np.float64)
        for i in range(WC_num.shape[0]):
            for k in range(WC_num.shape[1]):
                val = WC_num[i, k]
                if abs(val) >= threshold:
                    frac = Fraction(float(val)).limit_denominator(max_denom)
                    WC_exact[i, k] = sp.Rational(frac.numerator, frac.denominator)
                    WC_num_q[i, k] = float(frac)
                else:
                    WC_exact[i, k] = sp.S.Zero
                    WC_num_q[i, k] = 0.0

        # Быстрый численный фильтр для отсечения заведомо неверных q
        diff_max_num = np.max(np.abs(WC_num_q @ M_num.T - Target_num))

        if diff_max_num < best_diff_norm:
            best_diff_norm = diff_max_num
            best_WD_exact, best_WF_exact, best_WC_exact = WD_exact, WF_exact, WC_exact

        # Полная символьная верификация в SymPy (запускается только для перспективных q)
        if diff_max_num < 1e-3:
            M_sym = sp.Matrix(
                [
                    [WD_exact[k, d_i] * WF_exact[k, f_j] for k in range(inter_dim)]
                    for d_i, f_j in monomials
                ]
            )
            C_sym = sp.Matrix(WC_exact)
            Diff_sym = sp.expand(C_sym * M_sym.T - Target_sym)

            if Diff_sym == sp.zeros(Diff_sym.shape[0], Diff_sym.shape[1]):
                if verbose:
                    print(f"Success: Exact symbolic identity proved at q = {q_try} (residual = 0)!")
                return WD_exact, WF_exact, WC_exact, True

    if verbose:
        print(f"Symbolic evaluation finished with best numerical error: {best_diff_norm:.6f}.")
    return best_WD_exact, best_WF_exact, best_WC_exact, False


# --- Вывод явных выражений для каждого пикселя ---
def print_symbolic_pixel_expressions(
    WD,
    WF,
    WC,
    in_shape: tuple,
    out_shape: tuple,
    kernel_shape: tuple,
    stride: tuple,
    dilation: tuple,
    max_denom: int = 72,
):
    out_dim = math.prod(out_shape)

    # Приведение тензоров к numpy при необходимости
    if isinstance(WD, torch.Tensor):
        WD = WD.detach().cpu().numpy()
    if isinstance(WF, torch.Tensor):
        WF = WF.detach().cpu().numpy()
    if isinstance(WC, torch.Tensor):
        WC = WC.detach().cpu().numpy()

    inter_dim = WD.shape[0]
    d_num = math.prod(in_shape)
    f_num = math.prod(kernel_shape)

    # Создание символов d0..dN и f0..fM
    d_sym = sp.symbols(f"d0:{d_num}")
    f_sym = sp.symbols(f"f0:{f_num}")

    # Генерация символьного эталона для произвольного N-D
    target_sym, _, _ = generate_symbolic_target(
        in_shape, out_shape, kernel_shape, stride, dilation
    )

    def to_sympy_val(val):
        if isinstance(val, (sp.Basic, sp.Atom)):
            return val
        if abs(val) < 1e-6:
            return sp.S.Zero
        frac = Fraction(float(val)).limit_denominator(max_denom)
        return sp.Rational(frac.numerator, frac.denominator)

    # Сборка промежуточных элементов M_k = (sum WD_k,i * d_i) * (sum WF_k,j * f_j)
    M_sym = []
    for k in range(inter_dim):
        term_d = sp.S.Zero
        for i in range(d_num):
            s_val = to_sympy_val(WD[k, i])
            if s_val != 0:
                term_d += s_val * d_sym[i]

        term_f = sp.S.Zero
        for j in range(f_num):
            s_val = to_sympy_val(WF[k, j])
            if s_val != 0:
                term_f += s_val * f_sym[j]

        M_sym.append(sp.expand(term_d * term_f))

    print("\n" + "=" * 60)
    print("Analytical breakdown of expressions for each pixel")
    print("=" * 60)

    # Итерация по всем выходам
    out_coords_list = list(product(*[range(o) for o in out_shape]))

    for out_idx, out_coords in enumerate(out_coords_list):
        pred_expr = sp.S.Zero
        for k in range(inter_dim):
            s_val = to_sympy_val(WC[out_idx, k])
            if s_val != 0:
                pred_expr += s_val * M_sym[k]

        pred_expr = sp.expand(pred_expr)
        ideal_expr = target_sym[out_idx]
        diff_expr = sp.expand(pred_expr - ideal_expr)

        print(f"\n--- Pixel Y{out_idx} (coords: {out_coords}) ---")
        print(f"Computed  : {pred_expr}")
        print(f"Target    : {ideal_expr}")

        if diff_expr == 0:
            print("Residual  : 0 (Exact identity proved)")
        else:
            print(f"Residual  : {diff_expr}")


def print_sympy_matrix(name: str, matrix_exact: np.ndarray):
    rows, cols = matrix_exact.shape
    zeros_count = 0

    print(f"\n--- Symbolic Matrix {name} (shape: {rows}x{cols}) ---")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            val = matrix_exact[r, c]
            if val == 0:
                row_str.append(f"{'0':>8}")
                zeros_count += 1
            else:
                row_str.append(f"{str(val):>8}")
        print("[" + " ".join(row_str) + " ]")

    zeros_pct = (zeros_count / (rows * cols)) * 100
    print(f"Zero percentage: {zeros_pct:.1f}%")
    
class HardExampleBuffer:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.D_buffer = None
        self.F_buffer = None

    def add_examples(self, D: torch.Tensor, F_tensor: torch.Tensor, errors: torch.Tensor, top_k: int = 64):
    
        # errors имеет форму (batch_size,) - MSE для каждого элемента батча
        _, top_idx = torch.topk(errors, k=min(top_k, D.size(0)))
        
        hard_D = D[top_idx].detach()
        hard_F = F_tensor[top_idx].detach()

        if self.D_buffer is None:
            self.D_buffer = hard_D
            self.F_buffer = hard_F
        else:
            self.D_buffer = torch.cat([self.D_buffer, hard_D], dim=0)
            self.F_buffer = torch.cat([self.F_buffer, hard_F], dim=0)

        # Если буфер переполнен — сохраняем только самые свежие/тяжелые
        if self.D_buffer.size(0) > self.capacity:
            self.D_buffer = self.D_buffer[-self.capacity:]
            self.F_buffer = self.F_buffer[-self.capacity:]

    def sample(self, batch_size: int):
    
        if self.D_buffer is None or self.D_buffer.size(0) == 0:
            return None, None
        
        curr_size = self.D_buffer.size(0)
        indices = torch.randint(0, curr_size, (min(batch_size, curr_size),))
        return self.D_buffer[indices], self.F_buffer[indices]
        
def get_current_scale(step: int, total_steps: int, max_scale: float) -> float:
    
    warmup_start = int(total_steps * 0.15)  # Начинаем рост с 15% шагов
    warmup_end = int(total_steps * 0.70)    # Заканчиваем на 70% шагов

    if step < warmup_start:
        return 1.0
    if step > warmup_end:
        return max_scale

    # Косинусный сглаженный переход [0.0 ... 1.0]
    progress = (step - warmup_start) / (warmup_end - warmup_start)
    smooth_p = 0.5 * (1.0 - math.cos(math.pi * progress))
    
    return 1.0 + (max_scale - 1.0) * smooth_p
    
# --- Основной эксперимент ---
def run_stable_experiment(
    out_shape=(2, 2),
    kernel_shape=(3, 3),
    stride=1,
    dilation=1,
    steps=100000,
    batch_size=2048,
    lr=0.002,
    l1_lambda=0.00015,
    print_delta: bool = False,
):
    # Рассчитываем геометрию и приводящие кортежи
    (
        in_shape,
        out_shape,
        kernel_shape,
        stride,
        dilation,
        inter_dim,
    ) = calculate_winograd_geometry(out_shape, kernel_shape, stride, dilation)
    
    max_denominator = get_max_denominator(in_shape)
    prune_threshold = 0.5 / max_denominator
    kernel_scale = float(math.prod(kernel_shape))
    winograd_scale = get_max_scaler(in_shape)

    # Расчет экономии умножений (FLOPs)
    naive_flops = math.prod(kernel_shape) * math.prod(out_shape)
    winograd_flops = inter_dim
    flop_savings_pct = (1.0 - (winograd_flops / naive_flops)) * 100.0

    print("=" * 65)
    print(" Experiment Parameters (Winograd Algorithm Mining)")
    print("=" * 65)
    print(f"  Device                    : {device}")
    print(f"  Data Type (dtype)         : {torch.get_default_dtype()}")
    print(f"  Input Shape (in_shape)    : {in_shape} ({math.prod(in_shape)} elems)")
    print(f"  Kernel Shape (kernel)     : {kernel_shape} ({math.prod(kernel_shape)} elems)")
    print(f"  Output Shape (out_shape)  : {out_shape} ({math.prod(out_shape)} elems)")
    print(f"  Stride & Dilation         : stride={stride}, dilation={dilation}")
    print("  " + "-" * 61)
    print(f"  Naive Multiplications     : {naive_flops}")
    print(f"  Winograd Multiplications  : {winograd_flops}")
    print(f"  FLOPs Reduction           : {flop_savings_pct:+.1f}%")
    print("  " + "-" * 61)
    print(f"  Total Steps               : {steps}")
    print(f"  Batch Size                : {batch_size}")
    print(f"  Initial LR / L1 Lambda    : {lr} / {l1_lambda}")
    print(f"  Pruning Threshold         : {prune_threshold:.6f}")
    print(f"  Max Denominator (q_max)   : {max_denominator}")
    print(f"  Target Scale (max scale)  : {winograd_scale}")
    print("=" * 65 + "\n")

    torch.manual_seed(42)

    # Инициализация модели
    model = WinogradFromSketch(
        in_shape, out_shape, kernel_shape, inter_dim, scale=float(1.0)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-08)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    start_pruning_step = int(steps * 0.20)
    stop_pruning_step = int(steps * 0.80)
    
    hard_buffer = HardExampleBuffer(capacity=2048 * 4)

    for step in range(1, steps + 1):
    
        new_scale = get_current_scale(step, steps, float(winograd_scale))

        if new_scale != model.scale:
            with torch.no_grad():
                model.fc_C.weight.mul_(model.scale / new_scale)
            model.scale = new_scale
            
        # Генерируем случайный батч
        D_rand = torch.randn(batch_size, 1, *in_shape, device=device)
        F_rand = torch.randn(batch_size, 1, *kernel_shape, device=device)

        target = batched_conv_nd(D_rand, F_rand, stride=stride, dilation=dilation)
        pred = model(D_rand, F_rand)

        per_sample_mse = (pred - target).pow(2).view(batch_size, -1).mean(dim=1)
        mse_loss = per_sample_mse.mean()

        # Буфер тяжелых примеров
        hard_buffer.add_examples(D_rand, F_rand, per_sample_mse, top_k=32)

        l1_penalty = (
            model.fc_D.weight.abs().sum()
            + model.fc_F.weight.abs().sum()
            + model.fc_C.weight.abs().sum()
        )

        normalized_mse = mse_loss / kernel_scale
        total_loss = normalized_mse + (l1_lambda * l1_penalty)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Дообучение на сложных примерах из буфера
        if step % 2 == 0 and hard_buffer.D_buffer is not None:
            D_hard, F_hard = hard_buffer.sample(batch_size=2048)
            
            target_hard = batched_conv_nd(D_hard, F_hard, stride=stride, dilation=dilation)
            pred_hard = model(D_hard, F_hard)
            
            loss_hard = criterion(pred_hard, target_hard)
            
            optimizer.zero_grad()
            loss_hard.backward()
            optimizer.step()

        scheduler.step()

        if step % 500 == 0 and start_pruning_step <= step <= stop_pruning_step:
            model.apply_pruning(threshold=prune_threshold)

        # Информативный прогресс-лог
        if step % (steps // 10) == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step {step:7d}/{steps} | lr: {current_lr:.2e} | "
                f"scale: {model.scale:5.2f} | mse: {mse_loss.item():.3e} | l1 sum: {l1_penalty.item():.2f}"
            )
         
        # Периодическая «тихая» проверка Символьным Сольвером
        check_interval = max(1000, steps // 100)
        if step >= start_pruning_step and step % check_interval == 0 and mse_loss.item() < 1e-3:
            
            WD_exact, WF_exact, WC_exact, success = solve_winograd_exact_symbolic(
                model,
                in_shape,
                out_shape,
                kernel_shape,
                stride,
                dilation,
                threshold=prune_threshold,
                max_denom=max_denominator,
                verbose=False,
            )

            if success:
                print(f"\n" + "!" * 65)
                print(f" [SUCCESS]: Early stopping triggered at step {step}/{steps}!")
                print(" Exact Winograd algorithm proved via SymPy (Residual = 0)!")
                print("!" * 65 + "\n")
                break

    # --- Оценка обученных матриц ---
    orig_mse, orig_rmse, orig_mae, orig_max = evaluate_custom_matrices(
        model.fc_D.weight,
        model.fc_F.weight,
        model.effective_WC,
        in_shape,
        kernel_shape,
        out_shape,
        stride=stride,
        dilation=dilation,
        batch_size=batch_size,
    )

    print("\n" + "#" * 65)
    print(" Original Trained Matrices (Float)")
    print(f"--> MSE       : {orig_mse:.3e}")
    print(f"--> RMSE      : {orig_rmse:.6f}")
    print(f"--> MAE       : {orig_mae:.6f}")
    print(f"--> Max Error : {orig_max:.6f}")
    print("#" * 65)

    print_matrix("fc_D (data)", model.fc_D.weight, threshold=prune_threshold)
    print_matrix("fc_F (kernel)", model.fc_F.weight, threshold=prune_threshold)
    print_matrix("fc_C (output)", model.effective_WC, threshold=prune_threshold)

    # --- Квантованные матрицы ---
    WD_snap, WF_snap, WC_snap = normalize_and_snap_matrices(
        model, prune_threshold=prune_threshold, max_denom=max_denominator
    )
    snap_mse, snap_rmse, snap_mae, snap_max = evaluate_custom_matrices(
        WD_snap,
        WF_snap,
        WC_snap,
        in_shape,
        kernel_shape,
        out_shape,
        stride=stride,
        dilation=dilation,
        batch_size=batch_size,
    )

    print("\n" + "#" * 65)
    print(f" Quantized Fractional Matrices (q <= {max_denominator})")
    print(f"--> MSE       : {snap_mse:.3e}")
    print(f"--> RMSE      : {snap_rmse:.6f}")
    print(f"--> MAE       : {snap_mae:.6f}")
    print(f"--> Max Error : {snap_max:.6f}")
    print("#" * 65)

    print_matrix_as_fractions(
        "fc_D (data - snapped)",
        WD_snap,
        threshold=prune_threshold,
        max_denom=max_denominator,
    )
    print_matrix_as_fractions(
        "fc_F (kernel - snapped)",
        WF_snap,
        threshold=prune_threshold,
        max_denom=max_denominator,
    )
    print_matrix_as_fractions(
        "fc_C (output - snapped)",
        WC_snap,
        threshold=prune_threshold,
        max_denom=max_denominator,
    )

    if print_delta:
        print_symbolic_pixel_expressions(
            WD_snap, WF_snap, WC_snap, in_shape, out_shape, kernel_shape, stride, dilation
        )

    # --- Финальный Символьный Сольвер SymPy ---
    WD_exact, WF_exact, WC_exact, _ = solve_winograd_exact_symbolic(
        model,
        in_shape,
        out_shape,
        kernel_shape,
        stride,
        dilation,
        threshold=prune_threshold,
        max_denom=max_denominator,
    )
    
    # Конвертация символьных матриц SymPy в тензоры с текущим глобальным dtype
    def sympy_to_tensor(mat):
        return torch.tensor(
            np.vectorize(float)(mat), device=device, dtype=torch.get_default_dtype()
        )

    WD_exact_t = sympy_to_tensor(WD_exact)
    WF_exact_t = sympy_to_tensor(WF_exact)
    WC_exact_t = sympy_to_tensor(WC_exact)

    exact_mse, exact_rmse, exact_mae, exact_max = evaluate_custom_matrices(
        WD_exact_t,
        WF_exact_t,
        WC_exact_t,
        in_shape,
        kernel_shape,
        out_shape,
        stride=stride,
        dilation=dilation,
        batch_size=batch_size,
    )

    print("\n" + "#" * 65)
    print(" Exact Solver Matrices (SymPy Proved)")
    print(f"--> MSE       : {exact_mse:.3e}")
    print(f"--> RMSE      : {exact_rmse:.6f}")
    print(f"--> MAE       : {exact_mae:.6f}")
    print(f"--> Max Error : {exact_max:.6f}")
    print("#" * 65)

    print_sympy_matrix("fc_D (data)", WD_exact)
    print_sympy_matrix("fc_F (kernel)", WF_exact)
    print_sympy_matrix("fc_C (output)", WC_exact)

    if print_delta:
        print_symbolic_pixel_expressions(
            WD_exact, WF_exact, WC_exact, in_shape, out_shape, kernel_shape, stride, dilation
        )

# --- Запуск ---
if __name__ == "__main__":
    run_stable_experiment(
        out_shape=(2), 
        kernel_shape=(3),
        stride=1,
        dilation=(1,1),
        steps=100000,
        batch_size=2048,
        lr=0.001,
        l1_lambda=0.00015,
    )
