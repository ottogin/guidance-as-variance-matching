"""The interface every model adapter implements, and the registry that finds them.

An adapter wraps one released checkpoint together with its released sampler. compute_w / compute_wmse only
need `levels`, `rows`, `data`, `noise` and `denoise_pair`; sample.py adds `sample`; `smoke` proves that
`denoise_pair` is exactly what the released pipeline computes.
"""
import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Split:
    """One dataset as a model uses it (paper tab:impl-data)."""

    calib: str               # what the calibration rows are
    n_calib: int
    eval: str = ""           # what the evaluation rows are
    n_eval: int = 0
    reference: str = ""      # the real reference set of the metrics
    seed: str = ""           # how initial noise is shared across arms


@dataclass(frozen=True)
class Spec:
    """A model as the paper reports it (tab:impl-models). tests/test_specs.py checks these against the paper."""

    key: str                 # registry name, e.g. "sd35m"
    name: str                # as printed, e.g. "SD 3.5 Medium"
    modality: str            # image | video | audio | motion | world | toy
    checkpoint: str
    reference: str           # what D0 is
    sampler: str
    steps: int
    output: str
    published_w: float       # in the paper's convention D0 + w (D1 - D0); NaN when not a constant
    published_note: str = ""
    datasets: dict = field(default_factory=dict)    # name -> Split; the first is the default
    calib_levels: int = 0    # L in tab:impl-data; 0 means every sampler level

    @property
    def default_dataset(self):
        return next(iter(self.datasets))


class Model:
    """Base class. Index 1 is the primary prediction D1, index 0 the reference D0."""

    spec: Spec
    batch = 8                # calibration rows per forward batch
    row_align = 1            # shard boundaries are multiples of this (keeps batching identical across shards)
    seed = 0

    def with_spec(self, **changes):
        """This instance with its Spec changed (a released checkpoint loaded through another one's adapter: see
        vmg/sources.py). The class's own Spec is left alone."""
        import dataclasses
        self.spec = dataclasses.replace(self.spec, **changes)
        return self

    def load(self):
        """Load weights onto the device. Idempotent."""

    def levels(self):
        """dict(sigma=[S], t=[S]): every level the released sampler evaluates, in sampler order (noisiest
        first). sigma is the EDM-equivalent noise level, t the sampler's own timestep value."""
        raise NotImplementedError

    def calib_levels(self):
        """Indices into levels() at which the moments are taken (default: all of them)."""
        return list(range(len(self.levels()["sigma"])))

    def rows(self, dataset, split="calib"):
        """Row ids of `dataset` used for calibration ("calib") or evaluation ("eval")."""
        raise NotImplementedError

    def data(self, dataset, rows):
        """(x0, cond): x0 [B, ...] float32 on the device, in the space the model denoises, and whatever
        conditioning `denoise_pair` needs for those rows."""
        raise NotImplementedError

    def noise(self, x0, k, eps):
        """x_t at sampler level k from x0 and standard-normal `eps`, in the model's own forward process."""
        raise NotImplementedError

    def denoise_pair(self, x, k, cond):
        """(D1, D0): the two clean-data predictions the released sampler combines at level k."""
        raise NotImplementedError

    def flat(self, x, cond):
        """(X [B, n, d], mask [B, n] or None): each row as n samples of d coordinates. The default is one
        sample per row; models with padded or variable-length rows (motion, audio) override it."""
        return x.reshape(x.shape[0], 1, -1), None

    def smoke(self):
        """Check that denoise_pair matches the released pipeline's own network calls. Raises on mismatch."""
        raise NotImplementedError

    def sample(self, dataset, rows, w, outdir):
        """Generate the evaluation rows `rows` into `outdir`. `w` is None (the published guidance), a float
        (a constant), or an array with one weight per sampler level (a schedule)."""
        raise NotImplementedError


# registry name -> "module:Class"
REGISTRY = {
    "toy": "vmg.models.toy:Toy",
    # text-to-image, the paper's main models (sd.py)
    "sd15": "vmg.models.sd:SD15",
    "sd35m": "vmg.models.sd:SD35Medium",
    "sd35l": "vmg.models.sd:SD35Large",
    "flux2-klein": "vmg.models.sd:Flux2Klein",
    # text-to-image, the other Fig 2 checkpoints (t2i.py)
    "sd14": "vmg.models.t2i:SD14",
    "sd13": "vmg.models.t2i:SD13",
    "commoncanvas": "vmg.models.t2i:CommonCanvas",
    "sd21": "vmg.models.t2i:SD21",
    "bksdm": "vmg.models.t2i:BKSDM",
    "sd3m": "vmg.models.t2i:SD3Medium",
    "chroma": "vmg.models.t2i:Chroma",
    "ovis": "vmg.models.t2i:Ovis",
    "zimage": "vmg.models.t2i:ZImage",
    "cogview4": "vmg.models.t2i:CogView4",
    "sana": "vmg.models.t2i:Sana",
    "cross-sd15-cc": "vmg.models.t2i:CrossSD15CC",
    "cross-sd15-sd21": "vmg.models.t2i:CrossSD15SD21",
    "cross-sd21-cc": "vmg.models.t2i:CrossSD21CC",
    "cross-cc-sd21": "vmg.models.t2i:CrossCCSD21",
    "cross-sd21-sd15": "vmg.models.t2i:CrossSD21SD15",
    # video and world models (video.py, cosmos.py)
    "svd": "vmg.models.video:SVD",
    "svd-frame1": "vmg.models.video:SVDFrame1",
    "wan": "vmg.models.video:Wan",
    "cogvideox": "vmg.models.video:CogVideoX",
    "latte": "vmg.models.video:Latte",
    "cosmos": "vmg.models.cosmos:Cosmos",
    # audio and motion (audio.py, mdm.py)
    "audioldm2": "vmg.models.audio:AudioLDM2",
    "sao": "vmg.models.audio:StableAudioOpen",
    "mdm": "vmg.models.mdm:MDM",
    # ImageNet and CIFAR-10: autoguidance and class CFG (edm2.py, cfg256.py); class = key with "-" -> "_"
    "edm2-512": "vmg.models.edm2:edm2_512",
    **{k: f"vmg.models.edm2:{k.replace('-', '_')}" for k in (
        "edm2-512-ag-s-xs-fid", "edm2-512-ag-s-xs-dino", "edm2-512-ag-su-xsu-fid", "edm2-512-ag-su-xsu-dino",
        "edm2-512-ag-xxl-m-fid", "edm2-512-ag-xxl-m-dino", "edm2-512-cfg-s-xsu-dino", "edm2-512-cfg-xxl-xsu-fid", "edm2-64-ag-s-xs-fid", "edm2-64-ag-s-xs-dino",
        "edm2-64-ag-s-xs-0033554", "edm2-64-ag-s-xs-0067108", "edm2-64-ag-s-xs-0134217", "edm2-64-ag-s-xs-0268435",
        "edm2-64-ag-s-xs-0536870", "edm2-64-ag-s-xs-2147483", "edm2-64-cfg-s-xsu-fid", "edm2-64-xs-cfg",
        "cifar10-edm-vp")},
    **{k: f"vmg.models.cfg256:{k.replace('-', '_')}" for k in (
        "dit-xl2", "sit-xl2", "maskdit", "repa-sit-xl2", "uvit-l2", "uvit-h2")},
}


def get(name, **kw):
    """An adapter instance by registry name."""
    assert name in REGISTRY, f"unknown model {name!r}; known: {', '.join(REGISTRY)}"
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kw)
