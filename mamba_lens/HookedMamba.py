# some of the code is modified from https://github.com/johnma2006/mamba-minimal

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange, einsum, repeat
from typing import List, Optional, Tuple, Union
from jaxtyping import Float, Int
from typing_extensions import Literal
from functools import partial


import re
import logging
from dataclasses import dataclass
import json
import math
import itertools
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from transformer_lens.HookedTransformer import HookedTransformer, Loss, Output
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from transformer_lens.utils import USE_DEFAULT_VALUE

from .input_dependent_hooks import InputDependentHookPoint, InputDependentHookedRootModule

MAMBA_TOKENIZER = 'EleutherAI/gpt-neox-20b'

# modified from https://github.com/johnma2006/mamba-minimal/blob/master/model.py#L95
def get_converted_model_from_hf(pretrained_model_name, device='cuda'):
    """
    Downloads the given model from huggingface and caches it locally
    Args:
        pretrained_model_name str: The model path on huggingface to download (for example "state-spaces/mamba-370m")
        device: the device to load the model on
    Returns:
        (cfg, state_dict)
        cfg is the MambaCfg model config
        state_dict is the model state dict, in hooked state dict format
    """
    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))

    def load_state_dict_hf(model_name, device=None, dtype=None):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True, map_location=device)
        
    config_data = load_config_hf(pretrained_model_name)
    cfg = convert_original_config_to_hooked_mamba_config(config_data, device=device)
    
    state_dict = load_state_dict_hf(pretrained_model_name, device=device)
    
    converted_state_dict = convert_original_state_dict_to_hooked_state_dict(cfg, state_dict=state_dict)
        
    return cfg, converted_state_dict


def convert_original_state_dict_to_hooked_state_dict(cfg, state_dict):
    """
    Convert the original mamba state dict format into the hooked state dict format
    This format is to make interp nicer/to make things look more like HookedTransformer
    Args:
        cfg MambaCfg: Model config
        state_dict dict: State dict in original format
    Returns:
        new_state_dict dict: The state dict in hooked mamba format
    """
    D = cfg.d_model
    E = cfg.d_inner
    N = cfg.d_state
    D_delta = cfg.dt_rank
    D_conv = cfg.d_conv
    V = cfg.vocab_size
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("backbone.", "").replace("mixer.", "")
        key = key.replace("layers.", "blocks.")
        # we split in_proj into two seperate things
        if 'in_proj' in key:
            new_state_dict[key] = value[:E]
            new_state_dict[key.replace("in_proj", "skip_proj")] = value[E:]
        # we renamed these
        elif 'dt_proj' in key:
            new_state_dict[key.replace("dt_proj", "W_delta_2")] = value
        elif 'norm_f' in key:
            new_state_dict[key.replace("norm_f", "norm")] = value
        # we split this into three seperate things
        elif 'x_proj' in key:
            W = value
            # pull them out
            new_state_dict[key.replace("x_proj", "W_delta_1")] = W[:D_delta]
            new_state_dict[key.replace("x_proj", "W_B")] = W[D_delta:D_delta+N]
            new_state_dict[key.replace("x_proj", "W_C")] = W[D_delta+N:]
        # we call this W_D
        elif '.D' in key:
            new_state_dict[key.replace(".D", ".W_D")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def convert_hooked_state_dict_to_original_state_dict(cfg, state_dict):
    """
    Convert the HookedMamba state dict format back to the original 
    format the pretrained models are stored in
    Args:
        cfg MambaCfg: Model config
        state_dict dict: State dict in hooked mamba format
    Returns:
        new_state_dict dict: State dict in original mamba format

    Note:
        See the code for the things we change, the purpose of this different
        format is to make mamba look like HookedTransformer
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        original_key = key
        if not 'lm_head' in key: # everything except lm_head has backbone in front
            key = 'backbone.' + key
        # replace blocks.*.stuff with layers.*.mixer.stuff
        key = re.sub(r"blocks\.(\d+)\.", r"layers.\1.mixer.", key)

        # they don't put the norm in mixer
        if 'mixer.norm' in key:
            key = key.replace("mixer.norm", "norm")
        
        if "in_proj" in key:
            # we split this into in_proj and skip_proj
            in_proj = value
            skip_proj = state_dict[original_key.replace("in_proj", "skip_proj")]
            new_state_dict[key] = torch.cat([in_proj, skip_proj], dim=0)
        elif "skip_proj" in key:
            pass # we do this above
        elif "W_delta_2" in key:
            new_state_dict[key.replace("W_delta_2", "dt_proj")] = value
        # we renamed these
        elif 'dt_proj' in key:
            new_state_dict[key.replace("dt_proj", "W_delta_2")] = value
        elif 'norm' in key and not 'layers' in key: # the base norm is called norm_f
            new_state_dict[key.replace("norm", "norm_f")] = value
        # we split x_proj into three seperate things
        elif 'W_delta_1' in key:
            W_delta_1 = value
            W_B = state_dict[original_key.replace("W_delta_1", "W_B")]
            W_C = state_dict[original_key.replace("W_delta_1", "W_C")]
            new_state_dict[key.replace("W_delta_1", "x_proj")] = torch.cat([W_delta_1, W_B, W_C], dim=0)
        elif 'W_B' in key or 'W_C' in key:
            pass # we do this above
        # we call this W_D
        elif "W_D" in key:
            new_state_dict[key.replace("W_D", "D")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict



SSM_CONFIG = 'ssm_config'
INITIALIZER_CFG = 'initializer_cfg'

def convert_original_config_to_hooked_mamba_config(cfg: dict, device='cuda'):
    '''
    Convert the cfg dict used in the mamba ssm github
    into a MambaCfg object
    '''
    # it's already converted, what are you doing
    if type(cfg) is MambaCfg:
        return cfg
    

    # extract the minimal stuff needed for a cfg first
    default_cfg = MambaCfg(
        d_model=cfg['d_model'],
        n_layers=cfg['n_layer'],
        vocab_size=cfg['vocab_size'],
        device=device
    )


    # grab any extra params we missed
    all_params = {}
    for param, default_value in vars(default_cfg).items():
        # skip this, its handled below
        if param == INITIALIZER_CFG:
            continue
        all_params[param] = default_value
        if param in cfg:
            all_params[param] = cfg[param]
        # some things are stored in 'ssm_config'
        if SSM_CONFIG in cfg and param in cfg[SSM_CONFIG]:
            all_params[param] = cfg[SSM_CONFIG][param]
        
    # grab any extra params that are initializer config params
    init_cfg_params = {}
    for param, default_value in vars(default_cfg.initializer_cfg).items():
        init_cfg_params[param] = default_value
        if param in cfg:
            init_cfg_params[param] = cfg[param]
        if INITIALIZER_CFG in cfg and param in cfg[INITIALIZER_CFG]:
            init_cfg_params[param] = cfg[INITIALIZER_CFG][param]
    
    return MambaCfg(initializer_cfg=MambaInitCfg(**init_cfg_params), **all_params)
        

def convert_hooked_mamba_config_to_original_config(hooked_mamba_cfg : MambaCfg):
    '''
    Convert MambaCfg used here into a dict that's compatible
    with the config format used in mamba ssm
    '''
    
    names_to_convert = {
        'n_layers': 'n_layer', # this was done to conform with hooked transformer
    }

    # anything else just return the same name
    for k in vars(hooked_mamba_cfg).keys():
        if not k in names_to_convert:
            names_to_convert[k] = k
    
    for k in vars(hooked_mamba_cfg.initializer_cfg).keys():
        if not k in names_to_convert:
            names_to_convert[k] = k

    terms_to_go_inside_ssm_config = set([
        'd_state',
        'd_conv',
        'expand',
    ])

    ignore_terms = {
        'device' # this doesn't belong in the config dict
    }

    result_dict = {
        SSM_CONFIG: {},
        INITIALIZER_CFG: {}
    }

    # initializer_cfg
    for k,v in vars(hooked_mamba_cfg.initializer_cfg).items():
        if not k in ignore_terms:
           result_dict[INITIALIZER_CFG][names_to_convert[k]] = v
    
    # ssm_config
    for k in terms_to_go_inside_ssm_config:
        if not k in ignore_terms:
            result_dict[SSM_CONFIG][names_to_convert[k]] = getattr(hooked_mamba_cfg, k)
    
    # everything else
    for k,v in vars(hooked_mamba_cfg).items():
        if not k in ignore_terms:
            if not k in set([INITIALIZER_CFG]) | terms_to_go_inside_ssm_config:
                result_dict[names_to_convert[k]] = v

    return result_dict


@dataclass
class MambaInitCfg:
    initializer_range: float = 0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,  # Change to 2 if we have MLP
    dt_init: str = 'random', # other option is "constant"
    dt_scale: float = 1.0,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor : float = 1e-4


@dataclass
class MambaCfg:
    d_model: int
    n_layers: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    default_prepend_bos: bool = True
    tokenizer_prepends_bos: bool = False
    n_ctx: int = 2048
    device: Union[torch.device,str] = 'cuda'
    initializer_cfg: MambaInitCfg = MambaInitCfg()
    d_inner: int = -1

    def __post_init__(self):
        if self.d_inner == -1:
            self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
    
    @property
    def D(self):
        return self.d_model
    @property
    def E(self):
        return self.d_inner
    @property
    def N(self):
        return self.d_state
    @property
    def D_delta(self):
        return self.dt_rank
    @property
    def D_conv(self):
        return self.d_conv
    @property
    def V(self):
        return self.vocab_size
        
# from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self,
                 d: int,
                 eps: float = 1e-5,
                 device: Union[torch.device,str] = 'cpu'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

def make_postfix(l):
    return f".{l}"

def hook_has_hooks(hook):
    return len(hook.fwd_hooks) > 0 or len(hook.bwd_hooks)

class HookedMamba(HookedTransformer, InputDependentHookedRootModule):
    def __init__(self,
                 cfg: MambaCfg,
                 tokenizer: AutoTokenizer,
                 initialize_params: Optional[bool] = False):
        """
        Args:
            cfg MambaCfg: Model config
            initialize_params Optional[bool]: If True, will do proper initialization of parameters
        """
        super(InputDependentHookedRootModule, self).__init__()

        if not type(cfg) is MambaCfg:
            print("converting cfg dict into MambaCfg")
            cfg = convert_original_config_to_hooked_mamba_config(cfg=cfg, device=device)
        self.cfg = cfg

        device : Union[torch.device,str] = self.cfg.device
        
        self.tokenizer = tokenizer

        if not self.tokenizer is None:
            # patch to make some of the HookedTransformer stuff work correctly
            if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        D = cfg.D
        V = cfg.V
        
        self.embedding = nn.Embedding(V, D, device=device)
        self.hook_embed = HookPoint()
        
        self.blocks = nn.ModuleList([HookedMambaBlock(cfg=cfg, initialize_params=initialize_params) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(D, device=device)
        self.hook_norm = HookPoint() # [B,L,D]
        self.lm_head  = nn.Linear(D, V, bias=False, device=device)
        self.hook_logits = HookPoint() # [B,L,V]
        
        # this does proper initialization of parameters
        if initialize_params:
            # self.apply runs the given funtion on all submodules
            self.apply(
                partial(
                    _init_weights,
                    n_layer=cfg.n_layer,
                    initializer_range=cfg.initializer_cfg.initializer_range,
                    rescale_prenorm_residual=cfg.initializer_cfg.rescale_prenorm_residual,
                    n_residuals_per_layer=cfg.initializer_cfg.n_residuals_per_layer
                )
            )
        
        # setup hook points
        super().setup()

    def forward(self, 
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "B L"],
            Float[torch.Tensor, "B L E"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: Optional[bool] = False,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[
            Union[Literal["left", "right"],None]
        ] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "B L"]] = None,
        stop_at_layer: Optional[int] = None,
        only_use_these_layers: Optional[List[int]] = None,
        fast_conv: Optional[bool] = False,
        fast_ssm: Optional[bool] = False,
        warn_disabled_hooks: Optional[bool] = True,
    ) -> Union[
        None,
        Float[torch.Tensor, "B L V"],
        Loss,
        Tuple[Float[torch.Tensor, "B L V"], Loss],
    ]:
        """Forward Pass.

        Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically
        tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a
        text string.

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style
        language models - if you want a custom loss function, the recommended behaviour is returning
        the logits and then applying your custom loss function.

        Args:
            return_type Optional[str]: The type of output to return. Can be one of: None (return
                nothing, don't calculate logits), 'logits' (return logits), 'loss' (return
                cross-entropy loss), 'both' (return logits and loss).
            loss_per_token bool: Whether to return the (next token prediction) loss per token (True)
                or average (False). Average loss is a scalar (averaged over position *and* batch),
                per-token loss is a tensor ([batch, position-1]) - position-1 because we're
                predicting the next token, and there's no specified next token for the final token.
                Defaults to False.
            prepend_bos Optional[bool]: Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. (Even for models not explicitly trained with a prepended BOS token, heads
                often use the first position as a resting position and accordingly lose information
                from the first token, so this empirically seems to give better results.) Pass True
                or False to locally override the default.
            padding_side Optional[Literal["left", "right"]]: Overrides self.tokenizer.padding_side.
                Specifies which side to pad on when tokenizing multiple strings of different
                lengths.
            start_at_layer Optional[int]: If not None, start the forward pass at the specified
                layer. Requires input to be the residual stream before the specified layer with
                shape [batch, pos, d_model]. Inclusive - ie, start_at_layer = 0 skips the embedding
                then runs the rest of the model. Supports negative indexing. start_at_layer = -1
                only runs the final block and the unembedding. Defaults to None (run the full
                model).
            tokens: Optional[Int[torch.Tensor, "batch pos"]]: Tokenized input. Only use if
                start_at_layer is not None and return type is "loss" or "both".
            stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer.
                Exclusive - ie, stop_at_layer = 0 will only run the embedding layer, stop_at_layer =
                1 will run the embedding layer and the first transformer block, etc. Supports
                negative indexing. Useful for analysis of intermediate layers, eg finding neuron
                activations in layer 3 of a 24 layer model. Defaults to None (run the full model).
                If not None, we return the last residual stream computed.
            only_use_these_layers Optional[List[Int]]: If not none, will process the layers provided only,
                in the given order
            fast_conv: Optional[bool]: If False, uses unoptimized pytorch code for the conv1d. If true,
                uses the custom cuda kernel from causal_conv1d (from https://github.com/Dao-AILab/causal-conv1d)
                must be installed seperately by using `pip install causal-conv1d>=1.1.0`
                Note that setting fast_conv=True will disable disable blocks.*.hook_conv (* represents a layer index)
            fast_ssm: Optional[bool]: If False, uses unoptimized pytorch code for the ssm loop. If true,
                uses the custom cuda kernel from mamba-ssm (from https://github.com/state-spaces/mamba),
                must be installed seperately by using `pip install mamba-ssm`
                Note that this will disable blocks.*.hook_delta, blocks.*.hook_A_bar, blocks.*.hook_B_bar,
                blocks.*.hook_y and blocks.*.hook_h.* (* represents an index)
            warn_disabled_hooks: Optional[bool]: When using fast_conv=True or fast_ssm=True, if you try and use one
                of the disabled hooks, warnings will be printed showing they are disabled. You can set this
                to False to disable those checks/warnings
        """

        if not tokens is None:
            input = tokens
        else:
            # make sure input is ids and not a str
            if type(input) is str:
                input = self.to_tokens(input=input, prepend_bos=prepend_bos, padding_side=padding_side)
            tokens = input
        
        input = input.to(self.cfg.device)
        
        given_tokens = len(input.size()) <= 2
                
        Batch,L = input.size()

        if given_tokens:
            # [B,L,D]                         [B,L]
            input_embed         = self.embedding(input)
        else: #[B,L,D]      [B,L,D]
            input_embed         = input
        resid         = self.hook_embed(input_embed)
        
        stopping = False
        if start_at_layer is None:
            start_at_layer = 0
        else:
            stopping = True
        if stop_at_layer is None:
            stop_at_layer = self.cfg.n_layers
        else:
            stopping = True
        
        if only_use_these_layers is None:
           only_use_these_layers = range(start_at_layer, stop_at_layer) 

        for layer_i in only_use_these_layers:
            # [B,L,D]         [B,L,D]
            resid     = self.blocks[layer_i](resid, fast_conv=fast_conv, fast_ssm=fast_ssm, warn_disabled_hooks=warn_disabled_hooks)
        
        # we stop early, just return the resid
        if stopping:
            return resid
        
        # [B,L,D]                   [B,L,D]
        resid_normed     = self.norm( resid )
        resid_normed     = self.hook_norm(resid_normed) # [B,L,D]
        
        # [B,L,V]          [D->V]    [B,L,D]
        logits    = self.lm_head( resid_normed ) # no bias
        logits    = self.hook_logits(logits) # [B,L,V]
        
        if return_type is None:
            return None
        else:
            if return_type == "logits":
                return logits
            else:
                assert (
                    tokens is not None
                ), "tokens must be passed in if return_type is 'loss' or 'both'"
                loss = self.loss_fn(logits, tokens, per_token=loss_per_token)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
    
    @staticmethod
    def from_pretrained(pretrained_model_name, device='cuda', tokenizer=None):
        '''
        Loads a pretrained model from huggingface or local
        '''
        cfg, state_dict = get_converted_model_from_hf(pretrained_model_name=pretrained_model_name, device=device)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MAMBA_TOKENIZER)
        model = HookedMamba(cfg=cfg, initialize_params=False, tokenizer=tokenizer) # no need to initialize since we will be overriding
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model # from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L54

class HookedMambaBlock(nn.Module):
    def __init__(self,
        cfg: MambaCfg,
        initialize_params: bool = True):
        """
        Args:
            cfg MambaCfg: Model config
            initialize_params Optional[bool]: If True, will do proper initialization of parameters
        """

        super().__init__()
        self.cfg : MambaCfg = cfg
        device : Union[torch.device,str] = self.cfg.device

        ## Variables
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        
        
        self.hook_resid_pre = HookPoint() # [B,L,D]
        self.hook_layer_input = HookPoint() # [B,L,D]
        
        ## Process inputs
        self.norm      = RMSNorm(D, device=device)
        self.hook_normalized_input = HookPoint() # [B,L,D]
        
        self.skip_proj = nn.Linear(D, E, bias=False, device=device)
        self.hook_skip = HookPoint() # [B,L,E]
        self.in_proj   = nn.Linear(D, E, bias=False, device=device)
        self.hook_in_proj = HookPoint() # [B,L,E]
        
        ## Conv
        self.conv1d    = nn.Conv1d(
            in_channels=E,
            out_channels=E,
            bias=True,
            kernel_size=D_conv,
            groups=E,
            padding=D_conv - 1,
            device=device
        )
        self.hook_conv = HookPoint()  # [B,L,E]
        self.hook_ssm_input = HookPoint() # [B,L,E]
        
        ## SSM Params
        self.W_delta_1 = nn.Linear(E, D_delta, bias=False, device=device)
        self.W_delta_2 = nn.Linear(D_delta, E, bias=True, device=device)
        self.W_B = nn.Linear(E, N, bias=False, device=device)
        self.W_C = nn.Linear(E, N, bias=False, device=device)
        
        if initialize_params:
            # from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L82

            # special W_delta_2 initialization
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = D_delta**-0.5 * cfg.initializer_cfg.dt_scale
            if cfg.initializer_cfg.dt_init == "constant":
                nn.init.constant_(self.W_delta_2.weight, dt_init_std)
            elif cfg.initializer_cfg.dt_init == "random":
                nn.init.uniform_(self.W_delta_2.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError
            
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(E, device=device) * (math.log(cfg.initializer_cfg.dt_max) - math.log(cfg.initializer_cfg.dt_min))
                + math.log(cfg.initializer_cfg.dt_min)
            ).clamp(min=cfg.initializer_cfg.dt_init_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.W_delta_2.bias.copy_(inv_dt)
            
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.W_delta_2.bias._no_reinit = True

        # "S4D real initialization", from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L103
        # A_log is a [E,N] matrix
        # E rows, each N-sized row is [log(1), log(2), log(3), ..., log(N)]
        A = repeat(
            torch.arange(1, N + 1, dtype=torch.float32, device=device),
            "N -> E N",
            E=E,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.hook_h_start = HookPoint()     # [B,E,N]
        
        self.hook_delta_1 = HookPoint() # [B,L,D_delta]
        self.hook_delta_2 = HookPoint() # [B,L,E]
        self.hook_delta = HookPoint() # [B,L,E]
        
        self.hook_A = HookPoint() # [E,N]
        self.hook_A_bar = HookPoint() # [B,L,E,N]
        self.hook_B = HookPoint()     # [B,L,N]
        self.hook_B_bar = HookPoint() # [B,L,E,N]
        
        self.hook_C = HookPoint()     # [B,L,N]
        
        def input_dependent_postfixes(input):
            Batch, L = input.size()
            for b,l in itertools.product(range(Batch), range(L)):
                postfix = make_postfix(l=l)
                yield postfix

        self.hook_h = InputDependentHookPoint(input_dependent_postfixes)     # [B,E,N], with l param

        self.hook_y = HookPoint() # [B,L,E]

        self.W_D = nn.Parameter(torch.ones(E))
        self.W_D._no_weight_decay = True

        self.hook_ssm_output = HookPoint() # [B,L,E]

        self.hook_after_skip = HookPoint() # [B,L,E]
        
        ## Project back out
        self.out_proj  = nn.Linear(E, D, bias=False)
        self.hook_out_proj = HookPoint() # [B,L,D]
        self.hook_resid_post = HookPoint() # [B,L,D]
    
    def forward(self,
                resid,
                fast_conv: bool = False,
                fast_ssm: bool =False,
                warn_disabled_hooks: bool =True):
        """
        Args:
            resid: the input to this block
            fast_conv: Optional[bool]: If False, uses unoptimized pytorch code for the conv1d. If true,
                uses the custom cuda kernel from causal_conv1d (from https://github.com/Dao-AILab/causal-conv1d)
                must be installed seperately by using `pip install causal-conv1d>=1.1.0`
                Note that setting fast_conv=True will disable disable blocks.*.hook_conv (* represents a layer index)
            fast_ssm: Optional[bool]: If False, uses unoptimized pytorch code for the ssm loop. If true,
                uses the custom cuda kernel from mamba-ssm (from https://github.com/state-spaces/mamba),
                must be installed seperately by using `pip install mamba-ssm`
                Note that this will disable blocks.*.hook_delta, blocks.*.hook_A_bar, blocks.*.hook_B_bar,
                blocks.*.hook_y and blocks.*.hook_h.* (* represents an index)
            warn_disabled_hooks: Optional[bool]: When using fast_conv=True or fast_ssm=True, if you try and use one
                of the disabled hooks, warnings will be printed showing they are disabled. You can set this
                to False to disable those checks/warnings
        Returns:
            resid: the updated residual after applying this block
        """
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L,D = resid.size()
        
        ###### Process inputs ######
        resid      = self.hook_resid_pre(resid) # [B,L,D]

        resid_input = resid
        # this hook is necessary if we want to override on only this layer's inputs
        if hook_has_hooks(self.hook_layer_input):
            # this clone prevents modifying layer input from modifying the residual stream
            resid_input = resid.clone() # clones are expensive, only do it if we need to
        resid_input = self.hook_layer_input(resid_input) # [B,L,D]
        
        # [B,L,D]             [B,L,D]
        resid_norm = self.norm(  resid_input  )
        resid_norm = self.hook_normalized_input(resid_norm) # [B,L,E]
        
        # [B,L,E]          [D->E]     [B,L,D]
        skip       = self.skip_proj( resid_norm ) # no bias
        skip       = self.hook_skip(skip) # [B,L,E]
        
        # [B,L,E]          [D->E]   [B,L,D]
        x_in       = self.in_proj( resid_norm ) # no bias
        x_in       = self.hook_in_proj(x_in) # [B,L,E]
        
        ###### Conv ######
        # [B,E,L]
        x_conv     = rearrange(x_in, 'B L E -> B E L')
        if fast_conv:
            if warn_disabled_hooks:
                for hook in [self.hook_conv]:
                    if hook_has_hooks(hook):
                        print(f"warning: hook {hook.name} will not be called because fast_conv=True, pass warn_disabled_hooks=True to disable this warning")

            from causal_conv1d import causal_conv1d_fn
            # this does the silu and conv at same time
            # so sadly we miss some hooks if we do this
            # [B,E,L]
            x_conv_out = causal_conv1d_fn(
                x=x_conv,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )
            # [B,L,E]
            x         = rearrange(x_conv_out, 'B E L -> B L E')
        else:
            # [B,E,L+3]                 [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
            x_conv_out = self.conv1d(   x_conv   )
            # [B,L+3,E]            [B,E,L+3]
            x_conv_out = rearrange(x_conv_out, 'B E L -> B L E')
            # [B,L,E]
            x_conv_out_cutoff = x_conv_out[:,:L,:]
            x_conv_out_cutoff = self.hook_conv(x_conv_out_cutoff) # [B,L,E]

            ###### Nonlinearity  ######
            # [B,L,E]               [B,L,E]
            x         = F.silu( x_conv_out_cutoff )
        x         = self.hook_ssm_input(x) # [B,L,E]
        
        ###### SSM ######
       
        # [E,N]
        self.A = -torch.exp(self.A_log)
        self.A = self.hook_A(self.A) # [E,N]
        
        ### Compute the delta, A_bar, B_bar, and C ahead of time,
        ### since none of them depend on h
        
        ## Compute Delta ##
        # [B,L,D_delta] [E->D_delta]  [B,E]
        delta_1        = self.W_delta_1( x ) # no bias
        delta_1        = self.hook_delta_1(delta_1) # [B,L,D_delta]
        
        # [B,L,E]         [D_delta->E] [B,L,D_delta] 
        delta_2        = self.W_delta_2(  delta_1  ) # with bias
        delta_2        = self.hook_delta_2(delta_2) # [B,L,E]

        # [B,L,N]     [E->N]   [B,L,E]
        B           = self.W_B(   x   )
        B           = self.hook_B(B) # [B,L,N]

        ## C
        # this just applies E->N projection to each E-sized vector
        # [B,L,N]      [E->N]  [B,L,E]     
        C           = self.W_C(   x   ) # no bias
        C           = self.hook_C(C) # [B,L,N]

        inner_loop_hooks = [
                self.hook_delta,
                self.hook_A_bar,
                self.hook_B_bar,
                self.hook_y,
        ] + list(self.hook_h.hooks.values())
        

        if fast_ssm:
            import selective_scan_cuda

            if warn_disabled_hooks:
                for hook in inner_loop_hooks:
                    if hook_has_hooks(hook):
                        print(f"warning: hook {hook.name} will not be called because fast_ssm=True, pass warn_disabled_hooks=True to disable this warning")

            # the cuda kernel is picky about shapes, rearrange things to make it happy
            
            # [B,E,L]
            skip_ssm_input = rearrange(skip, "B L E -> B E L")
            # [B,E,L]
            x_ssm_input = rearrange(x, "B L E -> B E L")
            # [B,E,L]
            delta_2_ssm_input = rearrange(delta_2, 'B L E -> B E L')
            # [B,1,N,L]
            B_ssm_input = rearrange(B, 'B L N -> B 1 N L')
            # [B,1,N,L]
            C_ssm_input = rearrange(C, "B L N -> B 1 N L")

            # hack because we applied bias above
            # it's a little slower but that's ok
            if not hasattr(self, "empty_bias"):
                self.empty_bias = torch.zeros(self.W_delta_2.bias.size(), device=self.cfg.device)

            # this does softplus(delta), discretization, inner loop, add x*D, and multiply softplus(skip)
            # all the stuff you see in the else clause below 
            # the contiguous is needed for cuda reasons
            y_apply_D_ssm_output, scan_intermediates, y_skip_ssm_output = SelectiveScanFn.apply(
                                    x_ssm_input.contiguous(), # u
                                    delta_2_ssm_input.contiguous(), # delta
                                    self.A.contiguous(), # A 
                                    B_ssm_input.contiguous(), # B
                                    C_ssm_input.contiguous(), # C
                                    self.W_D.float(), # D
                                    skip_ssm_input.contiguous(), # z
                                    self.empty_bias, # delta_bias
                                    True) # delta_softplus
            
            ssm_output_has_hooks = len(self.hook_ssm_output.fwd_hooks) > 0 or len(self.hook_ssm_output.bwd_hooks) > 0
           
            # recompute this if there was a hook,
            # this is in case the hook modifies it
            if ssm_output_has_hooks:
                # [B,L,E]
                y_ssm_output = rearrange(y_apply_D_ssm_output, "B E L -> B L E")
                y_ssm_output = self.hook_ssm_output(y_ssm_output) # [B,L,E]

                # [B,L,E]   [B,L,E]             [B,L,E]
                y_after_skip    = y_ssm_output * F.silu(  skip  )
            else:
                # [B,L,E]
                y_after_skip = rearrange(y_skip_ssm_output, "B E L -> B L E")
        else:
           
            # [B,L,E]           [B,L,E]
            delta  = F.softplus(delta_2) 
            delta  = self.hook_delta(delta) # [B,L,E]
            
            ## Discretize A
            # [B,L,E,N]                    [B,L,E] [E,N]
            A_bar       = torch.exp(einsum(delta, self.A, 'b l e, e n -> b l e n'))
            A_bar       = self.hook_A_bar(A_bar) # [B,L,E,N]
            
            ## Discretize B
            # [B,L,E,N]          [B,L,E]  [B,L,N] 
            B_bar       = einsum( delta,    B,     'b l e, b l n -> b l e n')
            B_bar       = self.hook_B_bar(B_bar) # [B,L,E,N]
            
            # Now we do the recurrence
            ys = []
            
            # latent state, init to zeros
            h = torch.zeros([Batch,E,N], device=self.cfg.device)
            h = self.hook_h_start(h) # [B,E,N]
            for l in range(L):
                # [B,E,N]   [B,E,N]     [B,E,N]          [B,E,N]          [B,E]
                h        =    h    *  A_bar[:,l,:,:]  + B_bar[:,l,:,:] * x[:,l].view(Batch, E, 1)
                
                postfix = make_postfix(l=l)
                h        = self.hook_h(h, postfix=postfix) # [B,E,N]
                
                # [B,E]    [B,E,N]       [B,N,1]   # this is like [E,N] x [N,1] for each batch
                y_l       =   h     @   C[:,l,:].view(Batch,N,1)
                # [B,E]              [B,E,1]
                y_l      =    y_l.view(Batch,E)
                ys.append(y_l)
                
            # we have lots of [B,E]
            # we need to stack them along the 1 dimension to get [B,L,E]
            y = torch.stack(ys, dim=1)
            y = self.hook_y(y) # [B,L,E]
            
            ###### Finish block ######
            
            # [B,L,E]  [B,L,E]    [B,L,E]       [E]
            y_ssm_output =   y      +   x     *  self.W_D
            y_ssm_output =  self.hook_ssm_output(y_ssm_output) # [B,L,E]
                
            # [B,L,E]   [B,L,E]             [B,L,E]
            y_after_skip    = y_ssm_output * F.silu(  skip  )

        y_after_skip    =  self.hook_after_skip(y_after_skip) # [B,L,E]
            
        # [B,L,D]         [E->D]   [B,L,E]
        y_out     = self.out_proj( y_after_skip ) # no bias
        y_out     = self.hook_out_proj(y_out) # [B,L,D]
    
        # [B,L,D]   [B,L,D]   [B,L,D]
        resid     = resid +  y_out
        resid     = self.hook_resid_post(resid) # [B,L,D]
        
        return resid

# modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py
class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, delta, A, B, C, D=None, skip=None, delta_bias=None, delta_softplus=False):
        import selective_scan_cuda
        y_apply_D_ssm_output, scan_intermediates, y_skip_ssm_output = selective_scan_cuda.fwd(x, delta, A, B, C, D, skip, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(x, delta, A, B, C, D, skip, delta_bias, scan_intermediates, y_apply_D_ssm_output)
        return y_apply_D_ssm_output, scan_intermediates, y_skip_ssm_output

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        import selective_scan_cuda
        x, delta, A, B, C, D, skip, delta_bias, scan_intermediates, y_apply_D_ssm_output = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        dx, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            x, delta, A, B, C, D, skip, delta_bias, dout, scan_intermediates, y_apply_D_ssm_output, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dskip = rest[0]
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (dx, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dskip,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)

# fromn https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L54
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)









### Tests ###


DEFAULT_TEST_MODEL_PATH = "state-spaces/mamba-370m"
def default_test_model(device='cuda'):
    return HookedMamba.from_pretrained(DEFAULT_TEST_MODEL_PATH, device=device)

def test_no_modify_inplace(model=None, device='cuda'):
    import itertools
    if model is None:
        model = default_test_model(device=device)
    # This test compares cached things with outputs from run_with_cache
    # it detects if there was in-place modification changing the output of run_with_hooks after things are stored
    # (a bug that causes issues when doing patching)

    for fast_conv, fast_ssm in itertools.product([True, False], [True, False]):
        manual_cache = {}
        
        def save_hook(tensor, hook):
            manual_cache[hook.name] = tensor.detach().to(model.cfg.device).clone()
            return tensor.clone()
        
        
        input = torch.tensor([[1,2,3]], device=model.cfg.device)
        model_args = [input]
        fwd_hooks = []
        bwd_hooks = []
        # fwd_hooks=None expands all input_dependent hooks
        # we need this context to get the input dependent hooks
        with model.input_dependent_hooks_context(fwd_hooks=None, bwd_hooks=None, model_args=model_args, model_kwargs={}):
            for name, hp in model.hook_dict.items():
                fwd_hooks.append((name, save_hook))
        
        manual_cache_logit = model.run_with_hooks(
            input,
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            fast_ssm=fast_ssm,
            fast_conv=fast_conv,
            warn_disabled_hooks=False,
        )
        
        run_with_cache_logit, run_with_cache_cache = model.run_with_cache(input, fast_ssm=fast_ssm, fast_conv=fast_conv,warn_disabled_hooks=False)
        
        for k in manual_cache.keys():
            manual_value = manual_cache[k]
            run_with_cache_value = run_with_cache_cache[k]
            did_modify_inplace = torch.any(manual_value != run_with_cache_value)
            if did_modify_inplace:
                print(f"on test for fast_conv {fast_conv} and fast_ssm {fast_ssm}")
                print("mismatch for key", k, "do you modify it in place?")
                assert(not did_modify_inplace)
        
def test_state_dict_convert(device='cuda'):

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    cfg, converted_state_dict = get_converted_model_from_hf(DEFAULT_TEST_MODEL_PATH, device=device)

    their_model = MambaLMHeadModel.from_pretrained(DEFAULT_TEST_MODEL_PATH)
    their_model = their_model.to(cfg.device)
    test_inputs = torch.tensor([[1,2]], device=cfg.device)

    their_logits = their_model.forward(test_inputs)
    our_model = HookedMamba(cfg)
    our_model.load_state_dict(converted_state_dict)
    our_model = our_model.to(cfg.device)
    our_logits = our_model.forward(test_inputs)
    
    for i in range(10):
        original = convert_hooked_state_dict_to_original_state_dict(cfg, converted_state_dict)
        converted = convert_original_state_dict_to_hooked_state_dict(cfg, original)
    our_model_again = HookedMamba(cfg)
    our_model_again.load_state_dict(converted)
    our_model_again = our_model_again.to(cfg.device)
    our_logits_again = our_model_again.forward(test_inputs)


    their_model.load_state_dict(original)
    #their_model = their_model.to(cfg.device)
    their_again_logits = their_model.forward()

    #assert(torch.allclose(their_logits, our_logits))
    assert(torch.allclose(their_logits, their_again_logits))
    
    assert(torch.allclose(our_logits, our_logits_again))
    

def test_cfg_conversion():
    # some extra terms are added by default, so this will display those
    # the issue is if terms are removed
    original_cfg = {
        'd_model': 3,
        'n_layer': 3,
        'vocab_size': 48,
        'ssm_config': {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
        },
        'initializer_cfg': {
            'initializer_range': 0.5,  # Now only used for embedding layer.
            'rescale_prenorm_residual': False,
            'n_residuals_per_layer': 3,  # Change to 2 if we have MLP
            'dt_init': 'totes', # other option is "constant"
            'dt_scale': 3.5,
            'dt_min': 0.022,
            'dt_max': 0.4,
            'dt_init_floor': 1e-1
        }
    }
    import hooked_mamba
    from importlib import reload
    reload(hooked_mamba)
    hooked_cfg = hooked_mamba.convert_original_config_to_hooked_mamba_config(original_cfg)
    print(hooked_cfg)
    original_back_cfg = hooked_mamba.convert_hooked_mamba_config_to_original_config(hooked_cfg)

    def compare_dicts(prefix, original, converted):
        for k in set(list(original.keys()) + list(converted.keys())):
            if not k in original:
                print(f"Warning: added new key {prefix}{k} that was not in original config")
            if not k in converted:
                print(f"Error: lost key {prefix}{k} which was in original config but is not after convert and convert back")
                assert(k in converted)
            if k in original and k in converted and original[k] != converted[k]:
                print(f"Error: key {prefix}{k} was originally value {original[k]} but after convert is now {converted[k]}")
                assert(original[k] == converted[k])
    compare_dicts("", original_cfg, original_back_cfg)
    compare_dicts(INITIALIZER_CFG + ".", original_cfg[INITIALIZER_CFG], original_back_cfg[INITIALIZER_CFG])
    compare_dicts(SSM_CONFIG + ".", original_cfg[SSM_CONFIG], original_back_cfg[SSM_CONFIG])


# they are different due to numerical differences so this will print a lot
def test_fast_grads(model=None, device='cuda'):
    import itertools
    from collections import defaultdict
    grads = defaultdict(lambda: {})
    torch.set_grad_enabled(True)

    if model is None:
        model = default_test_model(device=device)
    
    for param_name, param in model.named_parameters():
        reference_value = None
        for fast_conv, fast_ssm in itertools.product([False, True], [False, True]):    
            a = model.forward(torch.tensor([[1,2]]).to(model.cfg.device), fast_conv=fast_conv, fast_ssm=fast_ssm)
            loss = torch.mean(a)
            loss.backward()
            if reference_value is None:
                if hasattr(param, "grad") and not param.grad is None:
                    reference_value = param.grad.clone()
                else:
                    print("param", name, "does not have grad, skipping")
                    break
            else:
                if not torch.allclose(reference_value, param.grad, atol=0.05):
                    print(f"Error: grads not the same for param {param_name}")
                    print("Non fast grads:", reference_value)
                    print(f"grads for fast_conv={fast_conv}, fast_ssm={fast_ssm}", param.grad)
