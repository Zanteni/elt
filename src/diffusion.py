import torch
import math
pi = math.pi
# ============================================================================
# Schedules — registry pattern, same shape as build_model/build_loss
# ============================================================================
#-----------------------------------------------------------------------------
# betas helper
#-----------------------------------------------------------------------------
def validate_betas(betas):

    assert betas.ndim == 1

    assert torch.all(betas > 0)

    assert torch.all(betas < 1)

    assert torch.all(
        betas[1:] >= betas[:-1]
    )

def build_linear_schedule(cfg):   # cfg: cfg["diffusion"]
    T = int(cfg["diffusion"]["timestep"])
    beta_start = float(cfg["diffusion"]["beta_start"])
    beta_end = float(cfg["diffusion"]["beta_end"])
    assert beta_start<beta_end,f"beta_end must be greater than beta_start,got beta_start:{beta_start},beta_end:{beta_end}"
    betas = torch.linspace(beta_start,beta_end,T)
    validate_betas(betas)
    return betas

def build_cosine_schedule(cfg):
    T = int(cfg["diffusion"]["timestep"])
    s = float(cfg["diffusion"]["cosine_s"])
    steps = torch.arange(T+1,dtype=torch.float32)
    x = (steps/T+s)/(1+s)
    alphas_cumprod  = torch.cos(x*pi/2)**2
    alphas_cumprod  = alphas_cumprod /alphas_cumprod[0]
    betas = 1-alphas_cumprod[1:]/alphas_cumprod[:-1]
    betas = torch.clamp(betas,1e-5,0.999)
    validate_betas(betas)
    return betas

SCHEDULE_BUILDERS = {
    "linear": build_linear_schedule,
    "cosine": build_cosine_schedule
}

def build_schedule(cfg):
    name = cfg["diffusion"]["schedule"]
    if name not in SCHEDULE_BUILDERS:
        raise ValueError(f"Unknown schedule '{name}', expected one of {list(SCHEDULE_BUILDERS)}")
    return SCHEDULE_BUILDERS[name](cfg)


# ============================================================================
# Gaussian Diffusion
# ============================================================================

class GaussianDiffusion:
    def __init__(self, cfg, device=None):
        # calls build_schedule(cfg) once, precomputes betas/alphas/alphas_cumprod/
        # sqrt variants/posterior variance as tensors on device -- same "build
        # once at init, cache as buffer" pattern as RoPE2D's rotation cache
        ...

    # -------------------------
    # Forward process
    # -------------------------
    def q_sample(self, x0, t, noise):
        # raw formula: sqrt(alphas_cumprod[t])*x0 + sqrt(1-alphas_cumprod[t])*noise
        ...

    def forward_process(self, x0):
        # convenience wrapper for training: samples random t, random noise,
        # calls q_sample, returns (x_t, t, noise) -- this is what train_step_dit calls
        ...

    # -------------------------
    # Reverse process
    # -------------------------
    def predict_x0(self, x_t, t, eps_pred):
        ...

    def p_mean_variance(self, x_t, t, eps_pred):
        # posterior q(x_{t-1}|x_t,x0) mean/variance, feeds ddpm_step
        ...

    # -------------------------
    # Samplers — two distinct recipes, never combined in one call
    # -------------------------
    def ddpm_step(self, model, x_t, t, y):
        # ancestral, stochastic, one full step -- debug/ablation path
        ...

    def ddim_step(self, x_t, t, t_prev, eps):
        # deterministic, CFG-agnostic -- takes a single already-blended eps
        ...

    def sample(self, model, shape, y=None, guidance_scale=1.0,
               sampler="ddim", num_steps=50, device=None):
        # outer loop, x_T -> x_0. CFG blending lives HERE (not in ddim_step):
        # calls model twice per step (y, null) when guidance_scale != 1.0,
        # blends eps_uncond + w*(eps_cond - eps_uncond), then dispatches to
        # ddim_step or ddpm_step per the `sampler` arg
        ...


# ============================================================================
# Utilities
# ============================================================================

def extract(schedule_buffer, t, x_shape):
    # index a (T,) buffer at batch of timesteps t: (B,), reshape to
    # broadcast against x_shape, e.g. (B,1,1,...)
    ...