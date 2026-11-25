"""The CPU reductions of reproduce/experiments/seedcov.py: covariances from per-batch sums, and the comparisons."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "seedcov", Path(__file__).resolve().parents[1] / "reproduce" / "experiments" / "seedcov.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_cov_from_sums_matches_direct_and_merges_by_addition():
    rng = np.random.default_rng(0)
    N, D, p = 40, 5, 7
    mu = rng.normal(size=(D, p)) * 3                       # a per-position mean the centring must remove
    u = mu + rng.normal(size=(N, D, p)) * rng.uniform(0.5, 2, size=(D, 1))
    direct = np.einsum("ncp,ndp->cd", u - u.mean(0), u - u.mean(0)) / ((N - 1) * p)
    parts = [u[:13], u[13:30], u[30:]]                     # uneven batches, summed as analyze() does
    G = sum(np.einsum("bcp,bdp->cd", b, b) for b in parts)
    m1 = sum(b.sum(0) for b in parts)
    np.testing.assert_allclose(sc.cov_from_sums(G, m1, N, p), direct, rtol=1e-12, atol=1e-12)


def test_compare_on_exactly_proportional_covariances():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 16))
    Cu = X @ X.T + 0.1 * np.eye(16)
    r = sc.compare(3.0 * Cu, Cu)
    for nm in sc.NORMS:
        assert abs(r[f"a_{nm}"] - 3.0) < 1e-6 and r[f"err_{nm}"] < 1e-6, nm
    np.testing.assert_allclose(r["cos"], 1.0, atol=1e-8)
    np.testing.assert_allclose(r["gen"], 3.0, rtol=1e-8)
    assert r["white"] < 1e-8 and abs(r["a_tr"] - 3.0) < 1e-12


def test_compare_detects_a_rotation_and_scores_the_norms():
    th = 0.3
    R = np.eye(16)
    R[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    Cu = np.diag(np.linspace(16, 1, 16))
    Cc = 2.0 * R @ Cu @ R.T                                 # same spectrum up to scale, top plane rotated
    r = sc.compare(Cc, Cu)
    np.testing.assert_allclose(r["cos"][:2], np.cos(th), atol=1e-8)
    np.testing.assert_allclose(r["cos"][2:], 1.0, atol=1e-8)
    assert r["err_fro"] > 1e-3 and r["white"] > 1e-3       # zero only for exact proportionality
    # each norm's own fit is at least as good in that norm as the Frobenius least-squares scale
    for nm in (2, "nuc"):
        e_at_fro = np.linalg.norm(Cc - r["a_fro"] * Cu, nm) / np.linalg.norm(Cu, nm)
        assert r[f"err_{nm}"] <= e_at_fro + 1e-9


def test_full_pca_from_grams_matches_explicit_eigenvectors():
    rng = np.random.default_rng(2)
    n, d, K = 40, 120, 6                                    # fewer seeds than dimensions, as in the real run
    scale = np.geomspace(5, 0.2, d)
    Xc = rng.normal(size=(n, d)) * scale
    Xu = 0.7 * Xc + 0.5 * rng.normal(size=(n, d)) * scale[::-1]
    Xc, Xu = Xc - Xc.mean(0), Xu - Xu.mean(0)
    r = sc.full_pca(Xc @ Xc.T, Xu @ Xu.T, Xc @ Xu.T, K)

    def top(X, K):
        w, V = np.linalg.eigh(X.T @ X / (len(X) - 1))
        return w[::-1], V[:, ::-1][:, :K]
    lc, Vc = top(Xc, K)
    lu, Vu = top(Xu, K)
    np.testing.assert_allclose(r["lam_c"][:n - 1], lc[:n - 1], rtol=1e-9)
    np.testing.assert_allclose(r["overlap"], np.abs(Vc.T @ Vu), atol=1e-9)
    np.testing.assert_allclose(np.abs(Xc.T @ r["A_c"]), np.abs(Vc), atol=1e-9)   # the stored seed weights give the PCs


def test_full_pca_halves_recentre_on_their_own_mean():
    rng = np.random.default_rng(3)
    n, d, K = 40, 90, 4
    X = rng.normal(size=(n, d)) * np.geomspace(4, 0.3, d) + 3.0       # a common offset the centring must remove
    X = X - X.mean(0)
    a, b = np.arange(0, n, 2), np.arange(1, n, 2)
    r = sc.full_pca(X @ X.T, X @ X.T, X @ X.T, K, halves=(a, b))

    def top(Y):
        Y = Y - Y.mean(0)
        return np.linalg.eigh(Y.T @ Y)[1][:, ::-1][:, :K]
    np.testing.assert_allclose(r["selfcos_c"], np.abs(np.sum(top(X[a]) * top(X[b]), 0)), atol=1e-9)


def test_full_pca_takes_every_pc_and_stops_the_halves_at_theirs():
    rng = np.random.default_rng(4)
    n, d = 30, 200
    X = rng.normal(size=(n, d)) * np.geomspace(3, 0.5, d)
    X = X - X.mean(0)
    G = X @ X.T
    r = sc.full_pca(G, G, G, n - 1, halves=(np.arange(0, n, 2), np.arange(1, n, 2)))
    np.testing.assert_allclose(r["cos"], 1.0, atol=1e-6)            # a branch against itself, every PC
    assert np.isfinite(r["selfcos_c"][:13]).all() and np.isnan(r["selfcos_c"][13:]).all()   # halves of 15: 14 PCs, 13 kept


def test_resid_grams_equal_the_grams_of_x0_minus_du():
    rng = np.random.default_rng(5)
    n, d = 25, 60
    X0, Xc, Xu = (rng.normal(size=(n, d)) for _ in range(3))
    X0, Xc, Xu = X0 - X0.mean(0), Xc - Xc.mean(0), Xu - Xu.mean(0)
    Xd, R = Xc - Xu, X0 - Xu
    rr, dr = sc.resid_grams(X0 @ X0.T, Xu @ X0.T, Xu @ Xu.T, Xd @ X0.T, Xc @ Xu.T)
    np.testing.assert_allclose(rr, R @ R.T, atol=1e-10)
    np.testing.assert_allclose(dr, Xd @ R.T, atol=1e-10)


def test_cov_cos_matches_the_explicit_covariances_and_gives_the_residual():
    rng = np.random.default_rng(6)
    d = 12
    Xa, Xb = rng.normal(size=(30, d)) * np.geomspace(4, 0.5, d), rng.normal(size=(20, d)) * np.geomspace(3, 1, d)
    Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)                 # different seeds and counts, as across seed halves
    Sa, Sb = Xa.T @ Xa / 29, Xb.T @ Xb / 19
    scov, a = sc.cov_cos(Xa @ Xa.T, Xb @ Xb.T, Xa @ Xb.T)
    np.testing.assert_allclose(scov, np.sum(Sa * Sb) / (np.linalg.norm(Sa) * np.linalg.norm(Sb)), rtol=1e-12)
    np.testing.assert_allclose(a, np.sum(Sa * Sb) / np.sum(Sb ** 2), rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(Sa - a * Sb) / np.linalg.norm(Sa), np.sqrt(1 - scov ** 2), rtol=1e-10)
    np.testing.assert_allclose(sc.cov_cos(Xa @ Xa.T, 4 * Xa @ Xa.T, 2 * Xa @ Xa.T), (1.0, 0.25), rtol=1e-12)


def test_subspace_overlap_matches_principal_angles_and_ignores_order_within():
    rng = np.random.default_rng(7)
    n, d, K = 40, 120, 6
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    Z = rng.normal(size=(n, d))
    Xa = Z * np.r_[30, 10, np.geomspace(1, 0.05, d - 2)] @ Q.T
    Xb = Z * np.r_[10, 30, np.geomspace(1, 0.05, d - 2)] @ Q.T  # the top two PCs trade places
    Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)
    Aa, Ab = sc.gram_eig(Xa @ Xa.T, K)[1], sc.gram_eig(Xb @ Xb.T, K)[1]
    Sk = sc.subspace_overlap(Aa, Ab, Xa @ Xb.T)
    Va, Vb = Xa.T @ Aa, Xb.T @ Ab                             # the unit PCs
    np.testing.assert_allclose(Sk, [np.sum((Va[:, :k].T @ Vb[:, :k]) ** 2) / k for k in range(1, K + 1)], atol=1e-10)
    assert abs(Va[:, 0] @ Vb[:, 0]) < 0.3 and Sk[1] > 0.95     # same rank: no match; top-2 subspaces: nearly equal


def test_guided_grams_are_the_grams_of_the_combined_field():
    rng = np.random.default_rng(8)
    n, d, w = 20, 40, 4.5
    X0, Xc, Xu = (rng.normal(size=(n, d)) for _ in range(3))
    X0, Xc, Xu = X0 - X0.mean(0), Xc - Xc.mean(0), Xu - Xu.mean(0)
    Xw = w * Xc + (1 - w) * Xu
    ww, wx = sc.guided_grams(w, Xc @ Xc.T, Xu @ Xu.T, Xc @ Xu.T, Xc @ X0.T, Xu @ X0.T)
    np.testing.assert_allclose(ww, Xw @ Xw.T, atol=1e-10)
    np.testing.assert_allclose(wx, Xw @ X0.T, atol=1e-10)


def test_principal_plane_matches_explicit_vectors_and_puts_each_long_axis_on_its_own_direction():
    rng = np.random.default_rng(9)
    n, d, K = 40, 120, 6
    scale = np.geomspace(5, 0.2, d)
    Xa = rng.normal(size=(n, d)) * scale
    Xb = 0.6 * Xa + rng.normal(size=(n, d)) * scale[::-1]
    Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)
    Ga, Gb, Gab = Xa @ Xa.T, Xb @ Xb.T, Xa @ Xb.T
    Aa, Ab = sc.gram_eig(Ga, K)[1], sc.gram_eig(Gb, K)[1]
    cs, i, Sa, Sb = sc.principal_plane(Aa, Ab, Ga, Gb, Gab)
    Va, Vb = Xa.T @ Aa, Xb.T @ Ab                             # the unit PCs
    Y, c_ex, Zt = np.linalg.svd(Va.T @ Vb)
    np.testing.assert_allclose(cs, c_ex, atol=1e-10)
    np.testing.assert_allclose(np.mean(cs ** 2), sc.subspace_overlap(Aa, Ab, Gab)[-1], atol=1e-10)
    assert i == np.argmin(np.abs(cs ** 2 - np.mean(cs ** 2)))
    p, q = Va @ Y[:, i], Vb @ Zt[i]
    e2 = q - (p @ q) * p
    E = np.stack([p, e2 / np.linalg.norm(e2)], 1)              # the plane's orthonormal axes
    np.testing.assert_allclose(Sa, E.T @ (Xa.T @ Xa / (n - 1)) @ E, atol=1e-9)
    np.testing.assert_allclose(Sb, E.T @ (Xb.T @ Xb / (n - 1)) @ E, atol=1e-9)
    c, s = cs[i], np.sqrt(1 - cs[i] ** 2)
    assert abs(Sa[0, 1]) < 1e-9 and Sa[0, 0] > Sa[1, 1]         # a's long axis on p
    along, across = np.array([c, s]), np.array([-s, c])        # q in the plane, and its normal
    assert abs(along @ Sb @ across) < 1e-9 and along @ Sb @ along > across @ Sb @ across   # b's long axis on q


def test_principal_plane_of_a_field_against_itself_is_one_ellipse():
    rng = np.random.default_rng(10)
    n, d, K = 30, 80, 5
    X = rng.normal(size=(n, d)) * np.geomspace(4, 0.3, d)
    X = X - X.mean(0)
    G = X @ X.T
    lam, A = sc.gram_eig(G, K + 1)
    cs, i, Sa, Sb = sc.principal_plane(A[:, :K], A[:, :K], G, G, G)   # x_0 against D_c at the last step: the same field
    np.testing.assert_allclose(cs, 1.0, atol=1e-9)
    assert np.isfinite(Sa).all() and Sa[0, 0] > 0 and Sa[1, 1] > 0
    np.testing.assert_allclose(Sb, Sa, rtol=1e-9, atol=1e-9)
    cs, i, Sa, Sb = sc.principal_plane(A[:, :K], A[:, :K], G, G, G, out_a=A[:, K])   # second axis: PC K + 1
    np.testing.assert_allclose(Sb, Sa, rtol=1e-9, atol=1e-9)
    assert abs(Sa[0, 1]) < 1e-9 and Sa[0, 0] >= lam[K - 1] - 1e-9
    np.testing.assert_allclose(Sa[1, 1], lam[K], rtol=1e-9)


def test_vgap_seeds_matches_the_direct_mean_and_splits_it():
    rng = np.random.default_rng(11)
    n, S, c, p = 12, 3, 4, 9
    x, Dc, Du = (rng.normal(size=(n, S, c, p)) + rng.normal(size=(1, S, c, p)) for _ in range(3))
    s = np.array([1.0, 0.6, 0.2])
    u = np.concatenate([x, Dc, Du], axis=2)                    # [n, S, 3c, p], as run_batch stacks them
    G = np.einsum("nkap,nkbp->kab", u, u)
    m1 = u.sum(0)
    tot, mean, spread = sc.vgap_seeds(G, m1, n, s, c=c)
    dv = (Du - Dc) / s[None, :, None, None]                    # v_c - v_u
    np.testing.assert_allclose(tot, np.mean(np.sum(dv ** 2, axis=(2, 3)), 0) / (c * p), rtol=1e-12)
    np.testing.assert_allclose(mean, np.sum(dv.mean(0) ** 2, axis=(1, 2)) / (c * p), rtol=1e-12)
    np.testing.assert_allclose(spread, np.mean(np.sum((dv - dv.mean(0)) ** 2, axis=(2, 3)), 0) / (c * p), rtol=1e-12)
